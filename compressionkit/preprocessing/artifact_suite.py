"""Role-routing artifact suite runtime for PPG augmentation.

Consumes :class:`compressionkit.configs.artifact_suite.ArtifactSuiteConfig` and
applies PPG artifacts to ``(input, target)`` pairs according to each artifact's
*role* (where it lands relative to the input/target fork):

- ``recover``  -> applied to input AND target (model must reconstruct it faithfully)
- ``remove``   -> applied to input only, target stays clean (denoising objective)
- ``abstain``  -> same span zeroed in input AND target (dropout detection / abstention)

The suite operates on **raw** windows of shape ``(T,)`` (single) or ``(B, T)``
(batch) under the apply-before-norm policy: recover/remove corruption is applied in
raw amplitude space, then each branch is layer-normalized by its own per-window
stats (``normalize_after=True``). Abstain masks are applied after normalization so
dropout regions stay exactly zero. The config schema lives in
``compressionkit.configs.artifact_suite``; see
``experiments/ppg_augmentation_layering_design.md`` for the full design.
"""

from __future__ import annotations

import numpy as np

from compressionkit.configs.artifact_suite import (
    ArtifactParam,
    ArtifactRole,
    ArtifactSpec,
    ArtifactSuiteConfig,
    CurriculumConfig,
    NoiseBudgetConfig,
)
from compressionkit.preprocessing.augmentations import (
    add_baseline_wander,
    add_empirical_noise,
    add_motion_artifact,
    scale_beat_amplitudes,
    time_warp,
)


# ---------------------------------------------------------------------------
# Role-routing augmenter
# ---------------------------------------------------------------------------


class RoleRoutingAugmenter:
    """Apply a role-routed artifact suite to ``(input, target)`` pairs.

    Args:
        cfg: Artifact suite configuration.
        noise_bank: Optional empirical noise bank ``(N, W)`` for the
            ``empirical_noise`` artifact. If ``None``, that artifact is skipped.
        seed: RNG seed.
    """

    def __init__(
        self,
        cfg: ArtifactSuiteConfig,
        *,
        noise_bank: np.ndarray | None = None,
        seed: int = 42,
    ) -> None:
        self.cfg = cfg
        self.noise_bank = noise_bank
        self.rng = np.random.default_rng(seed)
        self.severity_scale = 1.0
        self._specs = cfg.effective_specs()

    def set_severity_scale(self, scale: float) -> None:
        """Set the global curriculum severity scale (called by the callback)."""
        self.severity_scale = float(min(max(scale, 0.0), 1.0))

    # -- severity sampling -------------------------------------------------

    def _sample_severity(self, spec: ArtifactSpec) -> float:
        sampled = self.rng.uniform(spec.severity_min, spec.severity_max)
        easy = spec.easy_anchor
        return float(easy + self.severity_scale * (sampled - easy))

    # -- per-artifact realizations ----------------------------------------

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Per-window layer normalization matching ``LayerNormalization1D``."""
        mean = float(np.mean(x))
        var = float(np.mean((x - mean) ** 2))
        return ((x - mean) / np.sqrt(var + self.cfg.epsilon)).astype(np.float32)

    def _additive_noise(self, spec: ArtifactSpec, signal: np.ndarray, severity: float) -> np.ndarray:
        """Return the additive noise vector for a ``remove``/additive artifact."""
        if spec.name == "gaussian":
            # Severity is a fraction of the signal's own std so it is meaningful in
            # both raw (pre-norm) and normalized space (std approx 1).
            sig_std = float(np.std(signal)) + 1e-8
            return (self.rng.standard_normal(signal.shape[0]).astype(np.float32) * severity * sig_std)
        if spec.name == "baseline_wander":
            corrupted = add_baseline_wander(
                signal,
                sample_rate=self.cfg.sample_rate,
                amplitude_range=(severity, severity),
                rng=self.rng,
            )
            return (corrupted - signal).astype(np.float32)
        if spec.name == "motion":
            corrupted = add_motion_artifact(
                signal, sample_rate=self.cfg.sample_rate,
                snr_range=(severity, severity), rng=self.rng,
            )
            return (corrupted - signal).astype(np.float32)
        if spec.name == "empirical_noise":
            if self.noise_bank is None or len(self.noise_bank) == 0:
                return np.zeros_like(signal)
            corrupted = add_empirical_noise(
                signal, self.noise_bank,
                snr_range=(severity, severity), rng=self.rng,
            )
            return (corrupted - signal).astype(np.float32)
        return np.zeros_like(signal)

    def _apply_recover(self, spec: ArtifactSpec, signal: np.ndarray, severity: float) -> np.ndarray:
        """Apply a recover-role transform to the shared signal (in place semantics)."""
        if spec.name == "baseline_wander":
            return add_baseline_wander(
                signal, sample_rate=self.cfg.sample_rate,
                amplitude_range=(severity, severity), rng=self.rng,
            ).astype(np.float32)
        if spec.name == "time_warp":
            return time_warp(
                signal, sample_rate=self.cfg.sample_rate,
                max_warp_fraction=severity, rng=self.rng,
            ).astype(np.float32)
        if spec.name == "beat_scale":
            spread = severity
            return scale_beat_amplitudes(
                signal, sample_rate=self.cfg.sample_rate,
                scale_range=(1.0 - spread, 1.0 + spread), rng=self.rng,
            ).astype(np.float32)
        # Additive recover artifacts (e.g. gaussian/motion under faithful_all).
        return (signal + self._additive_noise(spec, signal, severity)).astype(np.float32)

    def _span_mask(self, length: int, fraction: float) -> np.ndarray:
        span = int(round(min(max(fraction, 0.0), 1.0) * length))
        mask = np.ones(length, dtype=np.float32)
        if span <= 0:
            return mask
        if span >= length:
            return np.zeros(length, dtype=np.float32)
        start = int(self.rng.integers(0, length - span + 1))
        mask[start : start + span] = 0.0
        return mask

    # -- main entry points -------------------------------------------------

    def apply_pair(self, window: np.ndarray) -> dict:
        """Apply the suite to a single raw window.

        Args:
            window: Clean raw signal of shape ``(T,)``. With ``normalize_after``
                set (default), the returned arrays are layer-normalized.

        Returns:
            Dict with ``input``, ``target``, ``clean`` arrays and ``meta`` info
            (fired artifacts, sampled severities, achieved remove-SNR).
        """
        clean = np.asarray(window, dtype=np.float32).reshape(-1)
        length = clean.shape[0]

        # 1. Decide which artifacts fire, respecting the simultaneity cap.
        fired: list[tuple[ArtifactSpec, float]] = []
        for spec in self._specs:
            if spec.prob > 0.0 and self.rng.random() < spec.prob:
                fired.append((spec, self._sample_severity(spec)))
        cap = self.cfg.noise_budget.max_simultaneous
        if cap >= 0 and len(fired) > cap:
            keep_idx = self.rng.choice(len(fired), size=cap, replace=False)
            fired = [fired[i] for i in sorted(keep_idx)]

        # 2. Recover artifacts: transform the shared signal (input + target).
        shared = clean.copy()
        recover_names: list[str] = []
        for spec, sev in fired:
            if spec.role == "recover":
                shared = self._apply_recover(spec, shared, sev)
                recover_names.append(f"{spec.name}={sev:.3f}")

        # 3. Remove artifacts: accumulate additive noise on the input branch.
        remove_noise = np.zeros(length, dtype=np.float32)
        remove_names: list[str] = []
        for spec, sev in fired:
            if spec.role == "remove":
                remove_noise = remove_noise + self._additive_noise(spec, shared, sev)
                remove_names.append(f"{spec.name}={sev:.3f}")

        # 4. Destruction guard on the aggregate remove-noise.
        achieved_snr = float("inf")
        budget = self.cfg.noise_budget
        if np.any(remove_noise):
            p_sig = float(np.mean(shared**2) + 1e-12)
            p_noise = float(np.mean(remove_noise**2) + 1e-12)
            achieved_snr = 10.0 * np.log10(p_sig / p_noise)
            if achieved_snr < budget.min_post_corruption_snr_db and budget.enforce != "off":
                if budget.enforce == "skip":
                    remove_noise = np.zeros(length, dtype=np.float32)
                    achieved_snr = float("inf")
                else:  # scale_down
                    target_p_noise = p_sig / (10 ** (budget.min_post_corruption_snr_db / 10.0))
                    remove_noise = remove_noise * np.sqrt(target_p_noise / p_noise)
                    achieved_snr = budget.min_post_corruption_snr_db

        x_in = (shared + remove_noise).astype(np.float32)
        x_tgt = shared.astype(np.float32)
        clean_out = clean

        # 5. Apply-before-norm policy: normalize each branch by its own stats
        #    after corruption (recover + remove live in raw space above).
        if self.cfg.normalize_after:
            x_in = self._layer_norm(x_in)
            x_tgt = self._layer_norm(x_tgt)
            clean_out = self._layer_norm(clean)

        # 6. Abstain artifacts: zero the same span in input AND target. Applied
        #    after normalization so masked regions stay exactly zero.
        abstain_names: list[str] = []
        for spec, sev in fired:
            if spec.role == "abstain":
                mask = self._span_mask(length, sev)
                x_in = x_in * mask
                x_tgt = x_tgt * mask
                abstain_names.append(f"{spec.name}={sev:.3f}")

        return {
            "clean": clean_out,
            "input": x_in,
            "target": x_tgt,
            "meta": {
                "recover": recover_names,
                "remove": remove_names,
                "abstain": abstain_names,
                "remove_snr_db": achieved_snr,
                "severity_scale": self.severity_scale,
            },
        }

    def apply_batch(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply the suite to a batch of windows ``(B, T)``.

        Returns:
            ``(inputs, targets)`` each of shape ``(B, T)``.
        """
        batch = np.asarray(batch, dtype=np.float32)
        inputs = np.empty_like(batch)
        targets = np.empty_like(batch)
        for i in range(batch.shape[0]):
            res = self.apply_pair(batch[i])
            inputs[i] = res["input"]
            targets[i] = res["target"]
        return inputs, targets


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------


def make_curriculum_callback(augmenter: RoleRoutingAugmenter):
    """Build a Keras callback that ramps the augmenter severity scale per epoch.

    Returns ``None`` when the curriculum is disabled so callers can skip it.
    """
    cfg = augmenter.cfg.curriculum
    if not cfg.enabled:
        return None

    import keras

    class ArtifactCurriculumCallback(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):  # noqa: D401
            augmenter.set_severity_scale(cfg.scale_at(epoch))

    return ArtifactCurriculumCallback()


__all__ = [
    "ArtifactRole",
    "ArtifactSpec",
    "NoiseBudgetConfig",
    "CurriculumConfig",
    "ArtifactSuiteConfig",
    "RoleRoutingAugmenter",
    "make_curriculum_callback",
]
