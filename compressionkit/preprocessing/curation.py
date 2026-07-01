"""SQI-based curation of real signal windows for denoiser training targets.

The wavelet gain denoiser is trained (known-corruption style) to reproduce a
real "clean" proxy that itself carries residual noise ``n0``. Curating the
training targets down to the cleanest windows shrinks ``n0`` and lifts the
achievable denoising floor (quantified in
``experiments/scripts/diagnose_reference_noise.py``).

Two per-window quality scores are provided:

* **template SNR** — each beat's residual against the ensemble template.
  Captures in-band noise (the honest ``sigma0``) but needs several beats, so it
  is fragile on short windows / low-SNR pulses.
* **bandpass-residual SNR** — signal vs out-of-band residual. Robust on any
  window length; used as the fallback when too few windows yield a valid
  template SNR (e.g. short ECG training frames with <4 beats).

``curate_windows`` ranks windows by the chosen score and keeps the cleanest
``keep_frac`` fraction.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from compressionkit.evaluation.noise import (
    estimate_ecg_noise_floor,
    estimate_ecg_qrs_snr,
    estimate_ppg_noise_floor,
)

__all__ = [
    "bandpass_snr_db",
    "curate_windows",
    "ppg_pulse_snr_db",
    "template_snr_db",
]


def ppg_pulse_snr_db(
    x: np.ndarray,
    fs: float,
    *,
    pre_ms: float = 150.0,
    post_ms: float = 350.0,
    min_beats: int = 4,
) -> float:
    """Ensemble-template SNR (dB) for one PPG window via systolic peaks.

    Returns ``nan`` when fewer than ``min_beats`` pulses are detected (too
    noisy / flat to estimate reliably).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    std = float(x.std())
    if std < 1e-6:
        return float("nan")
    peaks, _ = find_peaks(x, distance=round(0.4 * fs), prominence=0.4 * std)
    pre = round(pre_ms / 1000.0 * fs)
    post = round(post_ms / 1000.0 * fs)
    if pre + post <= 0:
        return float("nan")
    beats = [x[p - pre : p + post] for p in peaks if p - pre >= 0 and p + post <= x.size]
    if len(beats) < min_beats:
        return float("nan")
    stack = np.stack(beats, axis=0)
    template = stack.mean(axis=0)
    residual = stack - template
    t_rms = float(np.sqrt(np.mean(template**2)))
    r_rms = float(np.sqrt(np.mean(residual**2)))
    if t_rms <= 0 or r_rms <= 0:
        return float("nan")
    return float(20.0 * np.log10(t_rms / r_rms))


def template_snr_db(x: np.ndarray, modality: str, fs: float) -> float:
    """Beat/pulse ensemble-template SNR (dB). ``nan`` if not estimable."""
    if modality == "ecg":
        return float(estimate_ecg_qrs_snr(np.asarray(x, dtype=np.float64), int(fs)).get("qrs_snr_db", float("nan")))
    return ppg_pulse_snr_db(x, fs)


def bandpass_snr_db(x: np.ndarray, modality: str, fs: float) -> float:
    """Out-of-band (bandpass-residual) SNR in dB. Robust on short windows."""
    nf = estimate_ecg_noise_floor(x, int(fs)) if modality == "ecg" else estimate_ppg_noise_floor(x, int(fs))
    sig = float(nf.get("bp_signal_power", 0.0)) + 1e-12
    noise = float(nf.get("bp_noise_power", 0.0)) + 1e-12
    return float(10.0 * np.log10(sig / noise))


def curate_windows(
    windows: np.ndarray,
    *,
    modality: str,
    fs: float,
    keep_frac: float,
    method: str = "auto",
    min_finite_frac: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Keep the cleanest ``keep_frac`` of windows by SQI.

    Args:
        windows: ``(N, T)`` array of real signal windows.
        modality: ``"ecg"`` or ``"ppg"``.
        fs: Sample rate in Hz.
        keep_frac: Fraction of windows to retain (cleanest first). ``>=1`` keeps all.
        method: ``"template"``, ``"bp"``, or ``"auto"`` (template unless too few
            windows yield a valid template SNR, then fall back to bandpass).
        min_finite_frac: For ``"auto"``, switch to bandpass when the fraction of
            windows with a finite template SNR is below this.

    Returns:
        ``(kept_windows, kept_indices, info)``. ``info`` records the method used
        and the kept-subset SNR summary.
    """
    n = len(windows)
    if keep_frac >= 1.0:
        return windows, np.arange(n), {"method": "none", "kept": n}

    method_used = method
    if method in ("template", "auto"):
        snr = np.array([template_snr_db(w, modality, fs) for w in windows], dtype=np.float64)
        finite_frac = float(np.isfinite(snr).mean())
        if method == "auto" and finite_frac < min_finite_frac:
            snr = np.array([bandpass_snr_db(w, modality, fs) for w in windows], dtype=np.float64)
            method_used = "bp"
        else:
            method_used = "template"
    else:
        snr = np.array([bandpass_snr_db(w, modality, fs) for w in windows], dtype=np.float64)
        method_used = "bp"

    ranked = np.argsort(np.where(np.isfinite(snr), snr, -np.inf))[::-1]
    n_keep = max(1, round(keep_frac * n))
    kept_idx = ranked[:n_keep]
    kept_snr = snr[kept_idx]
    info = {
        "method": method_used,
        "n_in": int(n),
        "n_kept": int(n_keep),
        "kept_median_snr_db": float(np.nanmedian(kept_snr)) if np.isfinite(kept_snr).any() else None,
        "kept_min_snr_db": float(np.nanmin(kept_snr)) if np.isfinite(kept_snr).any() else None,
    }
    return windows[kept_idx], kept_idx, info
