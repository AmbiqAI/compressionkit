"""Wide, realistic artifact augmentation for training robust ECG models.

The earlier denoiser training used a *bimodal* corruption distribution — a hard
spike of exactly-clean windows plus a uniform SNR band — and only a single noise
type (empirical residual + optional baseline wander). This module replaces that
with a wide, continuous distribution over the full contact-artifact family set
(:func:`compressionkit.synthetic.ecg_contact_artifacts.simulate_contact_artifact`):

* **Many families, mixed per window** — colored, mains, motion, lead_off,
  weak_leak — so the model sees the same diversity the benchmark scores on.
* **Continuous severity** drawn from a Beta distribution (no clean spike). A
  small, *continuous* near-clean tail keeps an identity anchor without the
  bimodality, so the model learns to back off smoothly as SNR rises.
* **Optional empirical noise bank** threaded through to the families that use
  real PTB-XL residuals, for realism.

The augmenter returns ``(clean_target, noisy_input)`` pairs, both unit-normalized
so a signal-domain loss is scale-consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compressionkit.synthetic.ecg_contact_artifacts import (
    normalize_window,
    simulate_contact_artifact,
    synthesize_artifact_waveform,
)

_DEFAULT_FAMILIES: tuple[str, ...] = ("colored", "mains", "motion", "lead_off", "weak_leak")


def build_artifact_bank(
    n_total: int,
    length: int,
    sample_rate: float,
    *,
    families: tuple[str, ...] = _DEFAULT_FAMILIES,
    severity_range: tuple[float, float] = (0.2, 1.0),
    noise_bank: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Precompute a bank of normalized artifact waveforms across families.

    The expensive family synthesis (colored noise, mains, motion, gating) is
    done once here so that training can sample + mix rows cheaply in-graph,
    mirroring the empirical noise-bank pattern. Each row is a unit-normalized
    artifact waveform with a randomly drawn family and shape severity.

    Args:
        n_total: Number of artifact waveforms to synthesize.
        length: Waveform length in samples.
        sample_rate: Sampling rate in Hz.
        families: Artifact families to draw from.
        severity_range: Range for the per-row *shape* severity (the mixing
            power fraction is sampled separately at train time).
        noise_bank: Optional empirical residual bank for families that use it.
        seed: RNG seed.

    Returns:
        Float32 array of shape ``(n_total, length)``.
    """
    rng = np.random.default_rng(seed)
    bank = np.empty((n_total, length), dtype=np.float32)
    fam_arr = np.asarray(families)
    for i in range(n_total):
        family = str(rng.choice(fam_arr))
        severity = float(rng.uniform(*severity_range))
        bank[i] = synthesize_artifact_waveform(
            family, length, sample_rate=sample_rate, severity=severity, rng=rng, noise_bank=noise_bank
        )
    return bank


@dataclass
class WideArtifactAugmenter:
    """Sample wide, realistic contact-artifact corruptions for training.

    Attributes:
        sample_rate: Sampling rate in Hz.
        families: Artifact families to draw from.
        family_weights: Optional sampling weights (defaults to uniform).
        severity_beta: ``(a, b)`` of the Beta distribution over severity in
            ``[0, 1]``. ``(0.9, 1.3)`` skews toward milder corruption (realistic:
            most windows are lightly contaminated, a tail is severe).
        clean_prob: Probability of a *near-clean* window. Its severity is drawn
            continuously from ``[0, clean_severity_max]`` (NOT a hard zero), so
            the identity anchor is part of the continuous distribution rather
            than a separate mode.
        clean_severity_max: Upper severity of the near-clean tail.
        noise_bank: Optional empirical residual bank for the families that use it.
    """

    sample_rate: float
    families: tuple[str, ...] = _DEFAULT_FAMILIES
    family_weights: tuple[float, ...] | None = None
    severity_beta: tuple[float, float] = (0.9, 1.3)
    clean_prob: float = 0.08
    clean_severity_max: float = 0.05
    noise_bank: np.ndarray | None = None
    _weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.family_weights is None:
            self._weights = np.ones(len(self.families), dtype=np.float64) / len(self.families)
        else:
            w = np.asarray(self.family_weights, dtype=np.float64)
            if w.shape[0] != len(self.families):
                raise ValueError("family_weights length must match families")
            self._weights = w / w.sum()

    def _sample_severity(self, rng: np.random.Generator) -> float:
        if rng.random() < self.clean_prob:
            return float(rng.uniform(0.0, self.clean_severity_max))
        a, b = self.severity_beta
        return float(rng.beta(a, b))

    def corrupt_window(self, clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Corrupt a single clean window with a randomly drawn family/severity."""
        family = str(rng.choice(self.families, p=self._weights))
        severity = self._sample_severity(rng)
        return simulate_contact_artifact(
            clean,
            family=family,
            severity=severity,
            sample_rate=self.sample_rate,
            rng=rng,
            noise_bank=self.noise_bank,
            normalize=True,
        )

    def corrupt_batch(self, clean: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(clean_target, noisy_input)`` for a clean batch.

        Both outputs are unit-normalized per window so a signal-domain loss is
        scale-consistent.
        """
        rng = np.random.default_rng(seed)
        clean = np.asarray(clean, dtype=np.float32)
        target = np.empty_like(clean)
        noisy = np.empty_like(clean)
        for i, window in enumerate(clean):
            target[i] = normalize_window(window)
            noisy[i] = self.corrupt_window(window, rng)
        return target, noisy


__all__ = ["WideArtifactAugmenter", "build_artifact_bank"]
