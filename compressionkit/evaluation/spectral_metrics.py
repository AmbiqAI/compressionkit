"""Frequency-domain reconstruction quality metrics.

These metrics complement time-domain PRD/MSE: they answer "did the codec
preserve the *spectral content we care about*" rather than "did it match
sample-for-sample". They are especially useful when the ground-truth
contains noise that the codec correctly removes — time-domain metrics
penalize that, but band-power and coherence metrics on the clinical band
will not.

All functions are evaluation-only (numpy + scipy). Input signals must be
1-D float arrays with the same length and sample rate.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps


def _welch_psd(
    x: np.ndarray, fs: int, *, nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(freqs, psd)`` for a 1-D signal using Welch's method."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if nperseg is None:
        nperseg = min(x.size, max(64, int(fs * 2)))
    f, pxx = sps.welch(x, fs=fs, nperseg=nperseg, detrend="constant")
    return f, pxx


def psd_band_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    fs: int,
    bands: list[tuple[float, float]],
    nperseg: int | None = None,
) -> dict[str, float]:
    """Per-band relative power error between two signals.

    For each band ``(low, high)`` returns the absolute relative error of the
    integrated PSD: ``|P_recon - P_orig| / max(P_orig, eps)``. A flat result
    of ~0 across bands means spectral content is preserved; a large error in
    a clinical band signals real signal loss; a large error only in
    out-of-band ranges signals (correct) noise removal.

    Args:
        original: 1-D ground-truth signal.
        reconstructed: 1-D reconstructed signal (same length).
        fs: Sample rate in Hz.
        bands: List of (low_hz, high_hz) bands. Use ``high_hz=None`` for
            "up to Nyquist".
        nperseg: Welch window length; default uses up to 2 s.

    Returns:
        ``{"band_<low>_<high>_orig_power", "..._recon_power",
        "..._rel_error"}`` for each band, plus an aggregate
        ``"band_total_rel_error"`` (mean over bands).
    """
    f_o, pxx_o = _welch_psd(original, fs, nperseg=nperseg)
    f_r, pxx_r = _welch_psd(reconstructed, fs, nperseg=nperseg)
    out: dict[str, float] = {}
    rel_errs: list[float] = []
    nyq = fs / 2.0
    for low, high in bands:
        lo = float(low)
        hi = float(high) if high is not None else nyq
        hi = min(hi, nyq * 0.999)
        mask_o = (f_o >= lo) & (f_o <= hi)
        mask_r = (f_r >= lo) & (f_r <= hi)
        p_o = float(np.trapezoid(pxx_o[mask_o], f_o[mask_o])) if np.any(mask_o) else 0.0
        p_r = float(np.trapezoid(pxx_r[mask_r], f_r[mask_r])) if np.any(mask_r) else 0.0
        rel = abs(p_r - p_o) / max(p_o, 1e-12)
        tag = f"band_{lo:g}_{hi:g}"
        out[f"{tag}_orig_power"] = p_o
        out[f"{tag}_recon_power"] = p_r
        out[f"{tag}_rel_error"] = float(rel)
        rel_errs.append(float(rel))
    out["band_total_rel_error"] = float(np.mean(rel_errs)) if rel_errs else 0.0
    return out


def weighted_freq_prd(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    fs: int,
    weights: list[tuple[float, float, float]],
) -> dict[str, float]:
    """Frequency-weighted PRD using per-band importance weights.

    Computes ``100 * sqrt( sum_b w_b |X_b - X̂_b|^2 / sum_b w_b |X_b|^2 )``
    where the sum is over FFT bins inside band ``b`` with weight ``w_b``.
    Bands that overlap accumulate their weights additively. Bins outside
    every band get weight 0 — i.e. they are ignored entirely.

    Args:
        original: 1-D ground-truth signal.
        reconstructed: 1-D reconstructed signal (same length).
        fs: Sample rate.
        weights: List of ``(low_hz, high_hz, weight)`` triplets. ``high_hz``
            may be ``None`` for Nyquist.

    Returns:
        ``{"weighted_freq_prd_percent"}``.
    """
    x = np.asarray(original, dtype=np.float64).ravel()
    y = np.asarray(reconstructed, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 4:
        return {"weighted_freq_prd_percent": 0.0}
    n = x.size
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    nyq = fs / 2.0
    w = np.zeros_like(f)
    for low, high, weight in weights:
        lo = float(low)
        hi = float(high) if high is not None else nyq
        hi = min(hi, nyq * 0.999)
        mask = (f >= lo) & (f <= hi)
        w[mask] += float(weight)
    num = float(np.sum(w * np.abs(X - Y) ** 2))
    den = float(np.sum(w * np.abs(X) ** 2)) + 1e-12
    return {
        "weighted_freq_prd_percent": float(100.0 * np.sqrt(max(num / den, 0.0))),
    }


def spectral_coherence(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    fs: int,
    band: tuple[float, float],
    nperseg: int | None = None,
) -> dict[str, float]:
    """Mean magnitude-squared coherence inside a band of interest.

    Returns 1.0 when reconstruction is a noise-free linear copy of the
    original on that band; drops sharply when the codec injects independent
    distortion. Insensitive to scale.

    Args:
        original: 1-D ground-truth signal.
        reconstructed: 1-D reconstructed signal (same length).
        fs: Sample rate.
        band: (low_hz, high_hz) band to integrate.
        nperseg: Welch window length.

    Returns:
        ``{"coherence_<low>_<high>": float}``.
    """
    x = np.asarray(original, dtype=np.float64).ravel()
    y = np.asarray(reconstructed, dtype=np.float64).ravel()
    nyq = fs / 2.0
    lo = float(band[0])
    hi = float(band[1]) if band[1] is not None else nyq
    hi = min(hi, nyq * 0.999)
    if x.size != y.size or x.size < 16 or hi <= lo:
        return {f"coherence_{lo:g}_{hi:g}": 0.0}
    if nperseg is None:
        nperseg = min(x.size, max(64, int(fs * 2)))
    f, cxy = sps.coherence(x, y, fs=fs, nperseg=nperseg)
    mask = (f >= lo) & (f <= hi)
    val = float(np.mean(cxy[mask])) if np.any(mask) else 0.0
    return {f"coherence_{lo:g}_{hi:g}": val}


# ---------------------------------------------------------------------------
# Default band sets
# ---------------------------------------------------------------------------


ECG_DEFAULT_BANDS: list[tuple[float, float]] = [
    (0.5, 5.0),    # baseline / P-wave / T-wave
    (5.0, 15.0),   # QRS body
    (15.0, 40.0),  # QRS edges, fast morphology
    (40.0, 80.0),  # noise / EMG (out of clinical band)
]

# Boost the QRS body and edges; deweight low baseline drift; ignore HF noise.
ECG_DEFAULT_FREQ_WEIGHTS: list[tuple[float, float, float]] = [
    (0.5, 5.0, 0.5),
    (5.0, 15.0, 2.0),
    (15.0, 40.0, 1.5),
    # 40+ Hz: implicitly weight 0 (omitted)
]

ECG_DEFAULT_COHERENCE_BAND: tuple[float, float] = (5.0, 40.0)

PPG_DEFAULT_BANDS: list[tuple[float, float]] = [
    (0.5, 3.0),   # fundamental + first harmonic
    (3.0, 8.0),   # higher harmonics
]

PPG_DEFAULT_FREQ_WEIGHTS: list[tuple[float, float, float]] = [
    (0.5, 3.0, 2.0),
    (3.0, 8.0, 1.0),
]

PPG_DEFAULT_COHERENCE_BAND: tuple[float, float] = (0.5, 8.0)


__all__ = [
    "ECG_DEFAULT_BANDS",
    "ECG_DEFAULT_COHERENCE_BAND",
    "ECG_DEFAULT_FREQ_WEIGHTS",
    "PPG_DEFAULT_BANDS",
    "PPG_DEFAULT_COHERENCE_BAND",
    "PPG_DEFAULT_FREQ_WEIGHTS",
    "psd_band_error",
    "spectral_coherence",
    "weighted_freq_prd",
]
