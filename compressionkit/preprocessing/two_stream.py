"""Two-stream PPG decomposition: baseline (trend) + pulsatile (residual).

The idea is that PPG signals have two components with very different
characteristics:

1. **Baseline** — slow vasomotor drift, respiration-induced modulation, and
   contact-pressure artefacts. Varies on the order of seconds, has most energy
   below ~0.5 Hz. Easy to compress (smooth, low-bandwidth) and needs smooth
   stitching at frame boundaries.

2. **Pulsatile** — the actual cardiac pulse waveform after removing the
   baseline. Quasi-periodic, bounded amplitude, and the component from which
   HR/HRV is extracted.

Separating them allows:
- A *tiny* codec for the baseline (very high CR, only a few latent codes).
- A standard RVQ for the pulsatile channel that doesn't waste capacity on
  modelling drift.
- Trivial stitching for the baseline (linear interpolation at boundaries).
- Standard overlap-add for the pulsatile channel (which is bounded and
  mean-zero, so seams are minimal).

All operations are pure NumPy and use fixed-size buffers so they can be
ported to embedded C.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def decompose_baseline_pulsatile(
    signal: np.ndarray,
    *,
    sample_rate: int,
    baseline_cutoff_hz: float = 0.5,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a 1-D PPG signal into baseline + pulsatile components.

    Uses a zero-phase Butterworth low-pass filter to extract the baseline,
    then subtracts it to get the pulsatile residual.

    Args:
        signal: 1-D float array of PPG samples.
        sample_rate: Sampling rate in Hz.
        baseline_cutoff_hz: Low-pass cutoff for baseline extraction.
        order: Butterworth filter order.

    Returns:
        ``(baseline, pulsatile)`` both shaped like ``signal``.
    """
    sig = np.asarray(signal, dtype=np.float32).ravel()
    if sig.size < 2 * order + 1:
        # Degenerate: return zero baseline
        return np.zeros_like(sig), sig.copy()

    nyq = sample_rate / 2.0
    wn = min(baseline_cutoff_hz / nyq, 0.99)
    sos = butter(order, wn, btype="low", output="sos")
    baseline = sosfiltfilt(sos, sig).astype(np.float32)
    pulsatile = sig - baseline
    return baseline, pulsatile


def normalize_robust(
    signal: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, float, float]:
    """Robust normalization using median and MAD.

    Args:
        signal: 1-D float array.
        epsilon: Floor for scale to avoid division by zero.

    Returns:
        ``(normalized, center, scale)`` where
        ``normalized = (signal - center) / scale``.
    """
    sig = np.asarray(signal, dtype=np.float32).ravel()
    center = float(np.median(sig))
    mad = float(np.median(np.abs(sig - center)))
    # 1.4826 converts MAD to std-equivalent for Gaussian distributions
    scale = max(1.4826 * mad, epsilon)
    normalized = (sig - center) / scale
    return normalized.astype(np.float32), center, scale


def decompose_and_normalize(
    signal: np.ndarray,
    *,
    sample_rate: int,
    baseline_cutoff_hz: float = 0.5,
    order: int = 3,
    epsilon: float = 1e-6,
) -> dict[str, np.ndarray | float]:
    """Full preprocessing: decompose then normalize each stream independently.

    Args:
        signal: Raw 1-D PPG signal.
        sample_rate: Sample rate in Hz.
        baseline_cutoff_hz: Low-pass cutoff for baseline extraction.
        order: Filter order.
        epsilon: Normalization floor.

    Returns:
        Dictionary with keys:
        - ``baseline_norm``: Normalized baseline (float32).
        - ``pulsatile_norm``: Normalized pulsatile (float32).
        - ``baseline_center``, ``baseline_scale``: Normalization params.
        - ``pulsatile_center``, ``pulsatile_scale``: Normalization params.
    """
    baseline, pulsatile = decompose_baseline_pulsatile(
        signal,
        sample_rate=sample_rate,
        baseline_cutoff_hz=baseline_cutoff_hz,
        order=order,
    )
    baseline_norm, b_center, b_scale = normalize_robust(baseline, epsilon=epsilon)
    pulsatile_norm, p_center, p_scale = normalize_robust(pulsatile, epsilon=epsilon)
    return {
        "baseline_norm": baseline_norm,
        "pulsatile_norm": pulsatile_norm,
        "baseline_center": b_center,
        "baseline_scale": b_scale,
        "pulsatile_center": p_center,
        "pulsatile_scale": p_scale,
    }


def reconstruct_from_streams(
    baseline_norm: np.ndarray,
    pulsatile_norm: np.ndarray,
    *,
    baseline_center: float,
    baseline_scale: float,
    pulsatile_center: float,
    pulsatile_scale: float,
) -> np.ndarray:
    """Reconstruct the full PPG signal from normalized streams.

    Args:
        baseline_norm: Normalized baseline.
        pulsatile_norm: Normalized pulsatile.
        baseline_center: Baseline normalization center.
        baseline_scale: Baseline normalization scale.
        pulsatile_center: Pulsatile normalization center.
        pulsatile_scale: Pulsatile normalization scale.

    Returns:
        Reconstructed signal (denormalized baseline + pulsatile).
    """
    baseline = np.asarray(baseline_norm, dtype=np.float32) * baseline_scale + baseline_center
    pulsatile = np.asarray(pulsatile_norm, dtype=np.float32) * pulsatile_scale + pulsatile_center
    return baseline + pulsatile


def downsample_baseline(
    baseline: np.ndarray,
    *,
    factor: int,
) -> np.ndarray:
    """Downsample baseline by simple averaging (box filter).

    Since the baseline is band-limited to ~0.5 Hz, aggressive downsampling
    is safe and produces the tiny representation we need.

    Args:
        baseline: 1-D normalized baseline signal.
        factor: Downsampling factor (e.g., 8 or 16).

    Returns:
        Downsampled baseline of length ``ceil(len(baseline) / factor)``.
    """
    sig = np.asarray(baseline, dtype=np.float32).ravel()
    # Pad to multiple of factor
    pad_len = (factor - sig.size % factor) % factor
    if pad_len > 0:
        sig = np.concatenate([sig, np.full(pad_len, sig[-1], dtype=np.float32)])
    return sig.reshape(-1, factor).mean(axis=1).astype(np.float32)


def upsample_baseline(
    baseline_ds: np.ndarray,
    *,
    factor: int,
    target_len: int,
) -> np.ndarray:
    """Upsample baseline via linear interpolation.

    Args:
        baseline_ds: Downsampled baseline.
        factor: Downsampling factor used.
        target_len: Desired output length.

    Returns:
        Upsampled baseline of length ``target_len``.
    """
    ds = np.asarray(baseline_ds, dtype=np.float32).ravel()
    n_ds = ds.size
    # Source sample positions (center of each averaging bin)
    src_x = np.arange(n_ds, dtype=np.float32) * factor + (factor - 1) / 2.0
    # Target sample positions
    tgt_x = np.arange(target_len, dtype=np.float32)
    return np.interp(tgt_x, src_x, ds).astype(np.float32)
