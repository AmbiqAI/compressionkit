"""Per-frame quality-of-service (QoS) indicator for RVQ codecs.

The RVQ adapter (:class:`compressionkit.evaluation.RvqCodec`) attaches the
following to :attr:`EncodedFrame.side` on every encode:

    - ``quant_distances`` — per-token L2 between stage residual and chosen
                            codeword (list of length L = #RVQ levels).
    - ``residual_norms``  — latent norm after each stage (length L + 1).
    - ``token_ids``       — list of token-ID arrays.

This module turns those raw quantities into a per-frame **confidence**
score in ``[0, 1]`` that downstream consumers can gate on:

    confidence ≈ 1.0   → frame is well-represented by the codebook
    confidence ≈ 0.0   → frame is out-of-distribution; treat reconstruction
                         as untrusted (fall back to raw / SPIHT / drop).

The mapping is **calibrated** on in-distribution validation frames so the
confidence has a defensible percentile interpretation:

    1.  Fit :class:`RvqQoSCalibrator` on N validation frames.
    2.  At inference, ``calibrator.score(qos)`` returns the confidence.

Customer pitch: this is the killer differentiator vs SPIHT. We don't just
*claim* the neural codec is trustworthy frame-by-frame — we ship the QoS
flag with the bitstream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from compressionkit.evaluation.codec import EncodedFrame

__all__ = [
    "RvqQoS",
    "RvqQoSCalibrator",
    "compute_rvq_qos",
]


# ---------------------------------------------------------------------------
# Per-frame raw QoS
# ---------------------------------------------------------------------------


@dataclass
class RvqQoS:
    """Raw per-frame RVQ quality indicators (uncalibrated).

    Attributes:
        mean_quant_distance: Mean over tokens AND levels of the L2 distance
            between each stage residual and the chosen codeword.
        max_quant_distance: Maximum of the same.
        per_level_mean_distance: Mean quant distance at each RVQ level.
        residual_norm_per_stage: Latent residual norm after each stage
            (length L + 1; the first entry is the initial latent norm).
        final_residual_norm: ``residual_norm_per_stage[-1]``.
        relative_residual: ``final_residual_norm / max(initial_norm, eps)``.
            ``0`` → perfect quantization; ``1`` → codebook didn't help at all.
        codebook_perplexity_per_level: Per-level exp(entropy of token usage).
            Low values flag mode collapse / OOD inputs.
        initial_latent_norm: ``residual_norm_per_stage[0]``. Exposed because
            anomalously low latent norm (e.g. zero input) trivially yields
            a small residual and would otherwise fool the calibrator.
        confidence: Calibrated confidence in ``[0, 1]``. ``None`` until a
            calibrator has scored this frame.
    """

    mean_quant_distance: float
    max_quant_distance: float
    per_level_mean_distance: list[float]
    residual_norm_per_stage: list[float]
    final_residual_norm: float
    relative_residual: float
    codebook_perplexity_per_level: list[float]
    initial_latent_norm: float = 0.0
    confidence: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_perplexity(ids: np.ndarray, codebook_size: int) -> float:
    """Per-frame token-usage perplexity. Length-1 frames return 1.0."""
    if ids.size == 0:
        return 0.0
    counts = np.bincount(ids.astype(np.int64), minlength=int(codebook_size))
    probs = counts.astype(np.float64) / float(ids.size)
    nonzero = probs > 0
    if not np.any(nonzero):
        return 0.0
    entropy = float(-np.sum(probs[nonzero] * np.log2(probs[nonzero])))
    return float(2.0**entropy)


def compute_rvq_qos(encoded: EncodedFrame) -> RvqQoS:
    """Compute raw (uncalibrated) per-frame RVQ QoS from an encoded frame.

    Args:
        encoded: Result of :meth:`RvqCodec.encode`. Must have
            ``quant_distances``, ``residual_norms``, ``token_ids``, and
            ``codebook_sizes`` populated in :attr:`EncodedFrame.side`.

    Raises:
        ValueError: When the required side-channel keys are missing.
    """
    side = encoded.side
    required = ("quant_distances", "residual_norms", "token_ids", "codebook_sizes")
    missing = [k for k in required if k not in side]
    if missing:
        raise ValueError(f"compute_rvq_qos requires side keys {required}; missing {missing}")

    dists: list[np.ndarray] = [np.asarray(d, dtype=np.float64) for d in side["quant_distances"]]
    norms: list[float] = [float(n) for n in side["residual_norms"]]
    token_ids: list[np.ndarray] = [np.asarray(i, dtype=np.int64) for i in side["token_ids"]]
    codebook_sizes: list[int] = [int(k) for k in side["codebook_sizes"]]

    per_level_mean = [float(np.mean(d)) if d.size else 0.0 for d in dists]
    all_dists = np.concatenate(dists) if dists else np.zeros(0)
    mean_d = float(np.mean(all_dists)) if all_dists.size else 0.0
    max_d = float(np.max(all_dists)) if all_dists.size else 0.0

    initial_norm = norms[0] if norms else 0.0
    final_norm = norms[-1] if norms else 0.0
    rel = float(final_norm / initial_norm) if initial_norm > 1e-12 else 0.0

    perp = [_safe_perplexity(ids, k) for ids, k in zip(token_ids, codebook_sizes)]

    return RvqQoS(
        mean_quant_distance=mean_d,
        max_quant_distance=max_d,
        per_level_mean_distance=per_level_mean,
        residual_norm_per_stage=norms,
        final_residual_norm=float(final_norm),
        relative_residual=rel,
        codebook_perplexity_per_level=perp,
        initial_latent_norm=float(initial_norm),
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass
class RvqQoSCalibrator:
    """Maps raw QoS quantities to a calibrated confidence in ``[0, 1]``.

    Fit on a representative in-distribution validation set. Score at
    inference. The confidence is the geometric mean of three percentile-based
    sub-scores:

        score_dist = 1 - percentile(mean_quant_distance)
        score_res  = 1 - percentile(relative_residual)
        score_norm = 1 - 2 * |percentile(initial_latent_norm) - 0.5|
        confidence = (score_dist * score_res * score_norm) ** (1/3)

    The first two metrics measure how well the codebook explains the frame.
    The third is a **low-/high-energy guard**: anomalously low latent norm
    (e.g. zero input) trivially yields a small residual and would otherwise
    score as "perfect"; anomalously high norm signals saturation / OOD.
    Using a two-sided rank around the median catches both tails.

    Attributes:
        mean_distance_sorted: Sorted in-dist mean quant distances.
        relative_residual_sorted: Sorted in-dist relative residual values.
        initial_latent_norm_sorted: Sorted in-dist initial latent norms.
    """

    mean_distance_sorted: np.ndarray = field(default_factory=lambda: np.zeros(0))
    relative_residual_sorted: np.ndarray = field(default_factory=lambda: np.zeros(0))
    initial_latent_norm_sorted: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def fitted(self) -> bool:
        return (
            self.mean_distance_sorted.size > 0
            and self.relative_residual_sorted.size > 0
            and self.initial_latent_norm_sorted.size > 0
        )

    def fit(self, qos_samples: list[RvqQoS]) -> RvqQoSCalibrator:
        """Fit calibration distributions from in-distribution QoS samples."""
        if not qos_samples:
            raise ValueError("RvqQoSCalibrator.fit requires at least one sample")
        self.mean_distance_sorted = np.sort(np.asarray([q.mean_quant_distance for q in qos_samples], dtype=np.float64))
        self.relative_residual_sorted = np.sort(
            np.asarray([q.relative_residual for q in qos_samples], dtype=np.float64)
        )
        self.initial_latent_norm_sorted = np.sort(
            np.asarray([q.initial_latent_norm for q in qos_samples], dtype=np.float64)
        )
        return self

    @staticmethod
    def _rank(value: float, sorted_arr: np.ndarray) -> float:
        """Empirical CDF: fraction of *sorted_arr* ≤ *value*, in ``[0, 1]``."""
        if sorted_arr.size == 0:
            return 0.0
        idx = int(np.searchsorted(sorted_arr, value, side="right"))
        return float(idx) / float(sorted_arr.size)

    def score(self, qos: RvqQoS) -> float:
        """Return calibrated confidence in ``[0, 1]`` for *qos*.

        Also sets ``qos.confidence`` as a side effect for convenience.
        """
        if not self.fitted:
            raise RuntimeError("RvqQoSCalibrator must be fit() before score()")
        p_d = self._rank(qos.mean_quant_distance, self.mean_distance_sorted)
        p_r = self._rank(qos.relative_residual, self.relative_residual_sorted)
        p_n = self._rank(qos.initial_latent_norm, self.initial_latent_norm_sorted)
        # Lower percentile == better for distance / residual.
        score_d = 1.0 - p_d
        score_r = 1.0 - p_r
        # Two-sided: penalise both tails of initial latent norm.
        score_n = 1.0 - 2.0 * abs(p_n - 0.5)
        conf = float((max(0.0, score_d) * max(0.0, score_r) * max(0.0, score_n)) ** (1.0 / 3.0))
        qos.confidence = conf
        return conf

    def score_many(self, qos_samples: list[RvqQoS]) -> list[float]:
        return [self.score(q) for q in qos_samples]
