"""Frame-stitching strategies for reconstructing long recordings.

This module provides signal-agnostic helpers for reconstructing a long
1-D signal by sliding a trained compression model over it, frame-by-frame,
and stitching the reconstructed frames back together. Several strategies
are implemented so we can compare how sensitive downstream analyses
(HR/HRV, morphology) are to the choice of stitching — a useful lens for
driving future work on edge-friendly inference.

Methods
-------
``hard_concat``
    Non-overlapping frames concatenated end-to-end. Fastest and simplest;
    introduces visible seams at every frame boundary. Serves as the
    discontinuity baseline.
``overlap_add``
    Hann-windowed overlap-add at configurable hop ratio. Satisfies COLA
    at 50 % overlap; the de-facto default and what most papers report.
``linear_crossfade``
    Triangular-window OLA — gives a linear blend in the overlap region.
    Cheaper than Hann in embedded contexts (no cosine LUT) and also
    COLA-exact at 50 % overlap.
``tukey_overlap_add``
    Tukey (cosine-tapered rectangular) window with configurable flat
    fraction. A knob between ``hard_concat`` (taper = 0) and ``overlap_add``
    (taper = 1) useful for studying how much tapering is actually needed.

Each strategy accepts a callable ``predict_fn(frames) -> frames`` that
wraps the trained model. The stitching logic is decoupled from the model
interface so callers can plug in any encoder/decoder shape that accepts
``(B, 1, T, 1)`` inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------


PredictFn = Callable[[np.ndarray], np.ndarray]
"""``predict(frames) -> reconstructions``. Both arrays have shape ``(N, 1, T, 1)``."""


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------


def _hann_window(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def _triangular_window(n: int) -> np.ndarray:
    return np.bartlett(n).astype(np.float32)


def _tukey_window(n: int, alpha: float) -> np.ndarray:
    """Tukey window of length *n* with cosine-tapered fraction *alpha*.

    ``alpha=0`` → rectangular (no taper), ``alpha=1`` → Hann.
    """
    alpha = float(max(0.0, min(1.0, alpha)))
    if alpha <= 0.0:
        return np.ones(n, dtype=np.float32)
    if alpha >= 1.0:
        return _hann_window(n)
    w = np.ones(n, dtype=np.float32)
    taper = int(np.floor(alpha * (n - 1) / 2.0))
    if taper <= 0:
        return w
    t = np.arange(taper, dtype=np.float32)
    ramp = 0.5 * (1.0 + np.cos(np.pi * (t / taper - 1.0)))
    w[:taper] = ramp
    w[-taper:] = ramp[::-1]
    return w


# ---------------------------------------------------------------------------
# Frame extraction & model evaluation
# ---------------------------------------------------------------------------


def _frame_positions(total_len: int, frame_size: int, hop_size: int) -> list[int]:
    """Return start indices of frames that fit fully inside ``total_len``."""
    if frame_size > total_len:
        return []
    positions: list[int] = []
    pos = 0
    while pos + frame_size <= total_len:
        positions.append(pos)
        pos += hop_size
    return positions


def _extract_normalised_frames(
    signal: np.ndarray,
    positions: list[int],
    frame_size: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice frames, per-frame layer-normalise, return ``(frames, means, stds)``."""
    n = len(positions)
    frames = np.empty((n, frame_size), dtype=np.float32)
    means = np.empty(n, dtype=np.float32)
    stds = np.empty(n, dtype=np.float32)
    for i, start in enumerate(positions):
        f = signal[start : start + frame_size]
        m = float(np.mean(f))
        s = float(np.std(f)) + epsilon
        frames[i] = (f - m) / s
        means[i] = m
        stds[i] = s
    return frames, means, stds


def _run_model(frames: np.ndarray, predict_fn: PredictFn) -> np.ndarray:
    """Run the model on ``(N, frame_size)`` normalised frames."""
    batch = frames[:, np.newaxis, :, np.newaxis]
    recon = predict_fn(batch)
    return np.asarray(recon).reshape(frames.shape[0], frames.shape[1])


# ---------------------------------------------------------------------------
# Stitching strategies
# ---------------------------------------------------------------------------


def reconstruct_hard_concat(
    predict_fn: PredictFn,
    signal: np.ndarray,
    frame_size: int,
    *,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Non-overlapping, no-blend reconstruction (baseline)."""
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    total = len(sig)
    positions = _frame_positions(total, frame_size, hop_size=frame_size)
    if not positions:
        return sig.copy()

    frames, means, stds = _extract_normalised_frames(sig, positions, frame_size, epsilon)
    recon = _run_model(frames, predict_fn)

    out = sig.astype(np.float32).copy()
    for i, start in enumerate(positions):
        out[start : start + frame_size] = recon[i] * stds[i] + means[i]
    return out


def _windowed_overlap_add(
    predict_fn: PredictFn,
    signal: np.ndarray,
    frame_size: int,
    window: np.ndarray,
    hop_ratio: float,
    epsilon: float,
) -> np.ndarray:
    """Shared overlap-add core parameterised by the synthesis window."""
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    total = len(sig)
    hop_size = max(1, int(frame_size * hop_ratio))
    positions = _frame_positions(total, frame_size, hop_size)
    if not positions:
        return sig.copy()

    frames, means, stds = _extract_normalised_frames(sig, positions, frame_size, epsilon)
    recon = _run_model(frames, predict_fn)

    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    for i, start in enumerate(positions):
        denormed = recon[i].astype(np.float64) * stds[i] + means[i]
        out[start : start + frame_size] += window * denormed
        norm[start : start + frame_size] += window

    covered = norm > 1e-8
    out[covered] /= norm[covered]
    out[~covered] = sig[~covered]
    return out.astype(np.float32)


def reconstruct_overlap_add(
    predict_fn: PredictFn,
    signal: np.ndarray,
    frame_size: int,
    *,
    epsilon: float = 1e-3,
    hop_ratio: float = 0.5,
) -> np.ndarray:
    """Hann-window overlap-add — the canonical stitching method."""
    return _windowed_overlap_add(
        predict_fn, signal, frame_size,
        window=_hann_window(frame_size),
        hop_ratio=hop_ratio, epsilon=epsilon,
    )


def reconstruct_linear_crossfade(
    predict_fn: PredictFn,
    signal: np.ndarray,
    frame_size: int,
    *,
    epsilon: float = 1e-3,
    hop_ratio: float = 0.5,
) -> np.ndarray:
    """Triangular-window overlap-add (linear blend at seams)."""
    return _windowed_overlap_add(
        predict_fn, signal, frame_size,
        window=_triangular_window(frame_size),
        hop_ratio=hop_ratio, epsilon=epsilon,
    )


def reconstruct_tukey_overlap_add(
    predict_fn: PredictFn,
    signal: np.ndarray,
    frame_size: int,
    *,
    epsilon: float = 1e-3,
    hop_ratio: float = 0.5,
    alpha: float = 0.25,
) -> np.ndarray:
    """Tukey-window overlap-add with configurable taper fraction *alpha*."""
    return _windowed_overlap_add(
        predict_fn, signal, frame_size,
        window=_tukey_window(frame_size, alpha),
        hop_ratio=hop_ratio, epsilon=epsilon,
    )


# ---------------------------------------------------------------------------
# Stitching method registry
# ---------------------------------------------------------------------------


STITCH_METHODS: dict[str, Callable[..., np.ndarray]] = {
    "hard_concat": reconstruct_hard_concat,
    "overlap_add": reconstruct_overlap_add,
    "linear_crossfade": reconstruct_linear_crossfade,
    "tukey_overlap_add": reconstruct_tukey_overlap_add,
}
"""Mapping of method name → reconstruction function.

Keys are stable strings safe to persist in YAML configs / summary.json.
"""


def stitch(
    method: str,
    predict_fn: PredictFn,
    signal: np.ndarray,
    frame_size: int,
    **kwargs: Any,
) -> np.ndarray:
    """Dispatch to a named stitching method.

    Args:
        method: One of :data:`STITCH_METHODS`.
        predict_fn: Model-wrapping callable accepting ``(N, 1, T, 1)`` frames.
        signal: Long 1-D signal to reconstruct.
        frame_size: Model frame size in samples.
        **kwargs: Forwarded to the stitching function (e.g. ``hop_ratio``).

    Returns:
        Reconstructed signal of the same length as *signal*.
    """
    if method not in STITCH_METHODS:
        raise ValueError(f"Unknown stitching method {method!r}. Known: {sorted(STITCH_METHODS)}")
    return STITCH_METHODS[method](predict_fn, signal, frame_size, **kwargs)


# ---------------------------------------------------------------------------
# Stitching-quality metrics
# ---------------------------------------------------------------------------


def seam_discontinuity_ratio(
    signal: np.ndarray,
    *,
    frame_size: int,
    hop_ratio: float,
    radius: int = 4,
) -> dict[str, float]:
    """Quantify how visible frame seams are in a reconstructed signal.

    For each seam location (every ``hop_size`` samples after the first
    frame) we compute the RMS of the first difference in a small
    neighbourhood. We then compare that to the first-difference RMS far
    from any seam. A ratio near 1.0 means the seams are indistinguishable
    from the surrounding signal; larger values indicate visible
    discontinuities introduced by stitching.

    Args:
        signal: Reconstructed 1-D signal.
        frame_size: Model frame size used during reconstruction.
        hop_ratio: Hop ratio used during reconstruction.
        radius: Half-width (samples) of the seam neighbourhood.

    Returns:
        ``{"seam_rms", "non_seam_rms", "ratio", "num_seams"}``. Empty
        values are returned as ``nan`` when a signal is too short.
    """
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    n = sig.size
    if n < 2 * frame_size:
        return {"seam_rms": float("nan"), "non_seam_rms": float("nan"),
                "ratio": float("nan"), "num_seams": 0}

    diff = np.abs(np.diff(sig))
    hop = max(1, int(frame_size * hop_ratio))
    seam_positions = list(range(hop, n - 1, hop))
    seam_mask = np.zeros(diff.size, dtype=bool)
    for pos in seam_positions:
        lo = max(0, pos - radius)
        hi = min(diff.size, pos + radius)
        seam_mask[lo:hi] = True

    if not seam_mask.any() or seam_mask.all():
        return {"seam_rms": float("nan"), "non_seam_rms": float("nan"),
                "ratio": float("nan"), "num_seams": len(seam_positions)}

    seam_rms = float(np.sqrt(np.mean(diff[seam_mask] ** 2)))
    non_seam_rms = float(np.sqrt(np.mean(diff[~seam_mask] ** 2)))
    ratio = seam_rms / non_seam_rms if non_seam_rms > 0 else float("nan")
    return {
        "seam_rms": seam_rms,
        "non_seam_rms": non_seam_rms,
        "ratio": float(ratio),
        "num_seams": len(seam_positions),
    }


__all__ = [
    "STITCH_METHODS",
    "PredictFn",
    "reconstruct_hard_concat",
    "reconstruct_linear_crossfade",
    "reconstruct_overlap_add",
    "reconstruct_tukey_overlap_add",
    "seam_discontinuity_ratio",
    "stitch",
]
