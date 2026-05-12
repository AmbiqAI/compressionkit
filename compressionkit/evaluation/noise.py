"""Noise-floor estimation for ECG/PPG signals.

Customer-facing reconstruction metrics like PRD are confounded by noise on
the ground-truth signal: a model that *removes* high-frequency noise looks
worse than one that preserves it, even though it is clinically more useful.
This module provides several lightweight estimators of the per-recording
noise floor so we can normalize quality metrics and stratify reporting.

All functions are pure-numpy / scipy (LiteRT portability is not required —
these are evaluation-only) and operate on 1-D float signals. Each returns a
small dictionary so multiple estimators can be combined without name
collisions.
"""

from __future__ import annotations

import numpy as np
import physiokit as pk
from scipy import signal as sps

# ---------------------------------------------------------------------------
# Estimator 1 — High-frequency band power (above clinical bands of interest)
# ---------------------------------------------------------------------------


def estimate_hf_noise_power(
    x: np.ndarray, fs: int, *, hf_band: tuple[float, float] = (40.0, None)
) -> dict[str, float]:
    """Estimate noise power as the band power above the clinical band of interest.

    For ECG, the QRS complex and most clinical morphology lie below 40 Hz; the
    band above is dominated by EMG, mains, and instrumentation noise.

    Args:
        x: 1-D signal.
        fs: Sample rate in Hz.
        hf_band: (low_hz, high_hz) — high_hz=None means up to Nyquist.

    Returns:
        ``{"hf_noise_rms", "hf_noise_power", "hf_band_low", "hf_band_high"}``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    nyq = fs / 2.0
    low = float(hf_band[0])
    high = float(hf_band[1]) if hf_band[1] is not None else nyq
    high = min(high, nyq * 0.999)
    if high <= low or x.size < 8:
        return {
            "hf_noise_rms": 0.0, "hf_noise_power": 0.0,
            "hf_band_low": low, "hf_band_high": high,
        }
    # Welch PSD: trapezoidal integration over [low, high]
    nperseg = min(x.size, max(64, int(fs * 2)))
    f, pxx = sps.welch(x, fs=fs, nperseg=nperseg, detrend="constant")
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        power = 0.0
    else:
        power = float(np.trapezoid(pxx[mask], f[mask]))
    return {
        "hf_noise_rms": float(np.sqrt(max(power, 0.0))),
        "hf_noise_power": float(max(power, 0.0)),
        "hf_band_low": low,
        "hf_band_high": high,
    }


# ---------------------------------------------------------------------------
# Estimator 2 — Residual-after-bandpass (out-of-band energy)
# ---------------------------------------------------------------------------


def estimate_bandpass_residual_noise(
    x: np.ndarray,
    fs: int,
    *,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> dict[str, float]:
    """Estimate noise as the energy of ``x - bandpass(x)``.

    Captures both above-band noise (mains, EMG) and below-band drift (motion,
    respiration baseline). Uses physiokit's zero-phase Butterworth so the
    residual is not spectrally biased by phase distortion.

    Args:
        x: 1-D signal.
        fs: Sample rate in Hz.
        lowcut: Low cutoff in Hz (signal-of-interest band).
        highcut: High cutoff in Hz.
        order: Butterworth order.

    Returns:
        ``{"bp_noise_rms", "bp_noise_power", "bp_signal_rms",
        "bp_signal_power", "bp_lowcut", "bp_highcut"}``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 8:
        return {
            "bp_noise_rms": 0.0, "bp_noise_power": 0.0,
            "bp_signal_rms": 0.0, "bp_signal_power": 0.0,
            "bp_lowcut": lowcut, "bp_highcut": highcut,
        }
    clean = pk.signal.filter_signal(
        x.astype(np.float32),
        sample_rate=fs,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
        forward_backward=True,
    ).astype(np.float64)
    residual = x - clean
    return {
        "bp_noise_rms": float(np.sqrt(np.mean(residual**2))),
        "bp_noise_power": float(np.mean(residual**2)),
        "bp_signal_rms": float(np.sqrt(np.mean(clean**2))),
        "bp_signal_power": float(np.mean(clean**2)),
        "bp_lowcut": float(lowcut),
        "bp_highcut": float(highcut),
    }


# ---------------------------------------------------------------------------
# Estimator 3 — R-peak-locked SNR (ECG only)
# ---------------------------------------------------------------------------


def estimate_ecg_qrs_snr(
    x: np.ndarray,
    fs: int,
    *,
    template_window_ms: tuple[float, float] = (-100.0, 100.0),
    min_peaks: int = 4,
) -> dict[str, float]:
    """Estimate ECG SNR by averaging beats into a QRS template.

    Detects R-peaks, time-aligns a window around each, averages them into a
    template, then defines noise as the mean residual of each beat against
    the template. Falls back gracefully when too few beats are detected.

    Args:
        x: 1-D ECG signal.
        fs: Sample rate in Hz.
        template_window_ms: (pre, post) window in ms around each R-peak.
        min_peaks: Minimum number of detected peaks to compute SNR.

    Returns:
        ``{"qrs_snr_db", "qrs_template_rms", "qrs_residual_rms",
        "qrs_num_beats"}``. Returns NaN-filled dict if estimation fails.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    nan_result = {
        "qrs_snr_db": float("nan"),
        "qrs_template_rms": float("nan"),
        "qrs_residual_rms": float("nan"),
        "qrs_num_beats": 0,
    }
    if x.size < fs:
        return nan_result
    try:
        peaks = pk.ecg.find_peaks(x, sample_rate=fs)
        peaks = np.asarray(peaks).ravel()
    except Exception:
        return nan_result
    if peaks.size < min_peaks:
        return nan_result
    pre = round(template_window_ms[0] / 1000.0 * fs)
    post = round(template_window_ms[1] / 1000.0 * fs)
    win_len = post - pre
    if win_len <= 0:
        return nan_result
    valid: list[np.ndarray] = []
    for p in peaks:
        a, b = p + pre, p + post
        if a >= 0 and b <= x.size:
            valid.append(x[a:b].astype(np.float64))
    if len(valid) < min_peaks:
        return nan_result
    beats = np.stack(valid, axis=0)
    template = beats.mean(axis=0)
    residual = beats - template
    template_rms = float(np.sqrt(np.mean(template**2)))
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    if residual_rms <= 0 or template_rms <= 0:
        snr_db = float("nan")
    else:
        snr_db = float(20.0 * np.log10(template_rms / residual_rms))
    return {
        "qrs_snr_db": snr_db,
        "qrs_template_rms": template_rms,
        "qrs_residual_rms": residual_rms,
        "qrs_num_beats": int(beats.shape[0]),
    }


# ---------------------------------------------------------------------------
# Public composite API
# ---------------------------------------------------------------------------


def estimate_ecg_noise_floor(
    x: np.ndarray,
    fs: int,
    *,
    hf_band: tuple[float, float] = (40.0, None),
    bandpass: tuple[float, float] = (0.5, 40.0),
    bp_order: int = 4,
    qrs_window_ms: tuple[float, float] = (-100.0, 100.0),
) -> dict[str, float]:
    """Return all three ECG noise-floor estimators as a flat dict.

    The three estimates are deliberately not combined — each captures a
    different noise mode and we want callers (scorecard, stratification) to
    pick whichever best correlates with their downstream use.
    """
    out: dict[str, float] = {}
    out.update(estimate_hf_noise_power(x, fs, hf_band=hf_band))
    out.update(estimate_bandpass_residual_noise(
        x, fs, lowcut=bandpass[0], highcut=bandpass[1], order=bp_order,
    ))
    out.update(estimate_ecg_qrs_snr(x, fs, template_window_ms=qrs_window_ms))
    return out


def estimate_ppg_noise_floor(
    x: np.ndarray,
    fs: int,
    *,
    hf_band: tuple[float, float] = (8.0, None),
    bandpass: tuple[float, float] = (0.5, 8.0),
    bp_order: int = 3,
) -> dict[str, float]:
    """Return PPG noise-floor estimates (HF + bandpass-residual)."""
    out: dict[str, float] = {}
    out.update(estimate_hf_noise_power(x, fs, hf_band=hf_band))
    out.update(estimate_bandpass_residual_noise(
        x, fs, lowcut=bandpass[0], highcut=bandpass[1], order=bp_order,
    ))
    return out


__all__ = [
    "estimate_bandpass_residual_noise",
    "estimate_ecg_noise_floor",
    "estimate_ecg_qrs_snr",
    "estimate_hf_noise_power",
    "estimate_ppg_noise_floor",
]
