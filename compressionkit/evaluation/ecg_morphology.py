"""ECG beat-morphology metrics for compression evaluation.

Computes per-beat morphology measurements on paired (original,
reconstructed) ECG windows and reports orig/recon/delta aggregates so a
codec's effect on clinically-relevant beat shape can be quantified
independently of waveform PRD.

Designed to consume the same per-frame ``sample_*.csv`` artifacts the
scorecard already uses, and to mirror the structure of
:mod:`compressionkit.evaluation.ppg_morphology`.

Per-beat measurements (around each detected R-peak):

* ``r_amplitude``      — R height above local Q/S baseline.
* ``qrs_width_ms``     — duration from Q-onset (argmin in [-50, 0] ms)
  to S-offset (argmin in [0, +50] ms).
* ``st_deviation``     — ECG amplitude at J+80 ms minus PQ-segment
  baseline (median over [R-100, R-60] ms). Sign-preserving (positive =
  ST elevation, negative = depression).
* ``t_amplitude``      — peak deflection in [+150, +400] ms after R,
  relative to PQ baseline. Sign captures upright vs inverted T.
* ``baseline_drift``   — PQ-baseline level relative to the window mean
  (proxies for low-frequency wander; orig vs recon delta flags wander
  injected by the codec).

Notes on per-frame z-norm:
    Like the PPG morphology module, this evaluator operates on signals
    that may have been per-frame z-normalized in the scorecard
    pipeline. Absolute amplitudes are therefore in z-units; relative
    deltas between orig and recon remain meaningful because both sides
    receive identical normalization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import physiokit as pk


def _detect_peaks_clean(
    signal: np.ndarray,
    *,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bandpass + detect R-peaks via physiokit. Returns (cleaned, peaks)."""
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    try:
        cleaned = pk.ecg.clean(sig, sample_rate=sample_rate)
    except Exception:
        cleaned = sig
    try:
        peaks = pk.ecg.find_peaks(cleaned, sample_rate=sample_rate)
    except Exception:
        return cleaned, np.empty(0, dtype=int)
    if peaks is None:
        return cleaned, np.empty(0, dtype=int)
    return cleaned, np.asarray(peaks, dtype=int).reshape(-1)


def _ms_to_samples(ms: float, sample_rate: int) -> int:
    return max(1, int(round(ms * sample_rate / 1000.0)))


def _measure_beat(
    cleaned: np.ndarray,
    r_peak: int,
    *,
    sample_rate: int,
) -> dict[str, float | int | None] | None:
    """Compute per-beat morphology measurements around a single R-peak."""
    n = cleaned.size
    qrs_half = _ms_to_samples(50.0, sample_rate)        # ±50 ms QRS search
    pq_lo = _ms_to_samples(100.0, sample_rate)          # PQ baseline left
    pq_hi = _ms_to_samples(60.0, sample_rate)           # PQ baseline right
    j80 = _ms_to_samples(80.0, sample_rate)             # ST measurement offset
    t_lo = _ms_to_samples(150.0, sample_rate)
    t_hi = _ms_to_samples(400.0, sample_rate)

    q_start = max(0, r_peak - qrs_half)
    s_end = min(n, r_peak + qrs_half + 1)
    if r_peak - q_start < 3 or s_end - r_peak < 3:
        return None
    q_idx = int(q_start + np.argmin(cleaned[q_start : r_peak + 1]))
    s_idx = int(r_peak + np.argmin(cleaned[r_peak:s_end]))
    if s_idx <= q_idx:
        return None

    qrs_width_ms = float((s_idx - q_idx) / sample_rate * 1000.0)

    # PQ baseline: median over [R - pq_lo, R - pq_hi].
    pq_a = max(0, r_peak - pq_lo)
    pq_b = max(pq_a + 1, r_peak - pq_hi)
    if pq_b - pq_a < 2:
        pq_baseline = float(cleaned[pq_a])
    else:
        pq_baseline = float(np.median(cleaned[pq_a:pq_b]))

    # R amplitude over local Q/S baseline (less drift-sensitive than PQ baseline).
    local_baseline = 0.5 * (float(cleaned[q_idx]) + float(cleaned[s_idx]))
    r_amp = float(cleaned[r_peak] - local_baseline)

    # ST deviation at J+80 ms (J ≈ S point).
    j_plus = s_idx + j80
    if j_plus < n:
        st_dev = float(cleaned[j_plus] - pq_baseline)
    else:
        st_dev = float("nan")

    # T-wave: largest absolute deflection in [+t_lo, +t_hi] ms after R.
    t_a = min(n, r_peak + t_lo)
    t_b = min(n, r_peak + t_hi)
    if t_b - t_a >= 4:
        seg = cleaned[t_a:t_b] - pq_baseline
        idx = int(np.argmax(np.abs(seg)))
        t_amp = float(seg[idx])
    else:
        t_amp = float("nan")

    # Baseline drift proxy: PQ baseline relative to whole-window mean.
    drift = float(pq_baseline - float(cleaned.mean()))

    return {
        "r_peak": int(r_peak),
        "q_idx": q_idx,
        "s_idx": s_idx,
        "r_amplitude": r_amp,
        "qrs_width_ms": qrs_width_ms,
        "st_deviation": st_dev,
        "t_amplitude": t_amp,
        "pq_baseline": pq_baseline,
        "baseline_drift": drift,
    }


def _beat_shape_corr(
    orig_clean: np.ndarray,
    recon_clean: np.ndarray,
    r_peak: int,
    *,
    sample_rate: int,
    window_ms: tuple[float, float] = (-150.0, 400.0),
) -> float:
    """Pearson correlation of orig/recon over a fixed window around R."""
    n = orig_clean.size
    lo = max(0, r_peak + int(window_ms[0] * sample_rate / 1000.0))
    hi = min(n, r_peak + int(window_ms[1] * sample_rate / 1000.0))
    if hi - lo < 16:
        return float("nan")
    a = orig_clean[lo:hi]
    b = recon_clean[lo:hi]
    az = a - a.mean()
    bz = b - b.mean()
    denom = float(np.linalg.norm(az) * np.linalg.norm(bz))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(az, bz) / denom)


def _aggregate(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def evaluate_ecg_morphology(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    *,
    sample_rate: int,
    timing_tolerance_ms: float = 25.0,
    min_beats: int = 1,
) -> dict[str, Any]:
    """Compute paired ECG beat-morphology metrics on a batch of windows.

    Args:
        originals: Array of shape ``(N, T)`` with ground-truth ECG windows.
        reconstructions: Matched array of shape ``(N, T)``.
        sample_rate: Sample rate (Hz).
        timing_tolerance_ms: Half-window for matching orig/recon R-peaks.
        min_beats: Minimum beats a window must contain on both sides to
            contribute.

    Returns:
        Dict with per-metric paired blocks (``orig``, ``recon``,
        ``abs_delta``, ``delta``), a ``beat_shape_correlation`` block,
        and bookkeeping counts.
    """
    originals = np.asarray(originals)
    reconstructions = np.asarray(reconstructions)
    if originals.ndim == 1:
        originals = originals[None, :]
        reconstructions = reconstructions[None, :]
    if originals.shape != reconstructions.shape:
        raise ValueError(
            f"originals/reconstructions shape mismatch: {originals.shape} vs {reconstructions.shape}"
        )

    tol_samples = _ms_to_samples(timing_tolerance_ms, sample_rate)

    orig_ramp: list[float] = []
    recon_ramp: list[float] = []
    orig_qrs: list[float] = []
    recon_qrs: list[float] = []
    orig_st: list[float] = []
    recon_st: list[float] = []
    orig_t: list[float] = []
    recon_t: list[float] = []
    orig_drift: list[float] = []
    recon_drift: list[float] = []
    shape_corrs: list[float] = []

    num_windows = int(originals.shape[0])
    num_windows_with_beats = 0
    num_beats_matched = 0

    for orig, recon in zip(originals, reconstructions):
        o_clean, o_peaks = _detect_peaks_clean(orig, sample_rate=sample_rate)
        r_clean, r_peaks = _detect_peaks_clean(recon, sample_rate=sample_rate)
        if o_peaks.size < min_beats or r_peaks.size < min_beats:
            continue

        o_beats: list[dict[str, Any]] = []
        for p in o_peaks:
            m = _measure_beat(o_clean, int(p), sample_rate=sample_rate)
            if m is not None:
                o_beats.append(m)
        r_beats: list[dict[str, Any]] = []
        for p in r_peaks:
            m = _measure_beat(r_clean, int(p), sample_rate=sample_rate)
            if m is not None:
                r_beats.append(m)
        if not o_beats or not r_beats:
            continue
        num_windows_with_beats += 1

        used_r: set[int] = set()
        for ob in o_beats:
            deltas = np.array([abs(ob["r_peak"] - rb["r_peak"]) for rb in r_beats])
            valid = [int(i) for i in np.flatnonzero(deltas <= tol_samples) if int(i) not in used_r]
            if not valid:
                continue
            best = int(min(valid, key=lambda i: deltas[i]))
            used_r.add(best)
            rb = r_beats[best]
            num_beats_matched += 1

            orig_ramp.append(float(ob["r_amplitude"]))
            recon_ramp.append(float(rb["r_amplitude"]))
            orig_qrs.append(float(ob["qrs_width_ms"]))
            recon_qrs.append(float(rb["qrs_width_ms"]))
            if np.isfinite(ob["st_deviation"]) and np.isfinite(rb["st_deviation"]):
                orig_st.append(float(ob["st_deviation"]))
                recon_st.append(float(rb["st_deviation"]))
            if np.isfinite(ob["t_amplitude"]) and np.isfinite(rb["t_amplitude"]):
                orig_t.append(float(ob["t_amplitude"]))
                recon_t.append(float(rb["t_amplitude"]))
            orig_drift.append(float(ob["baseline_drift"]))
            recon_drift.append(float(rb["baseline_drift"]))

            # Beat-shape correlation uses orig's R-peak position for both
            # sides since matched recon R is within tol_samples.
            shape_corrs.append(
                _beat_shape_corr(
                    o_clean, r_clean, int(ob["r_peak"]), sample_rate=sample_rate
                )
            )

    def _paired_block(orig_vals: list[float], recon_vals: list[float]) -> dict[str, Any]:
        if not orig_vals:
            return {"orig": _aggregate([]), "recon": _aggregate([]), "abs_delta": _aggregate([]), "delta": _aggregate([])}
        deltas = [r - o for o, r in zip(orig_vals, recon_vals)]
        abs_deltas = [abs(d) for d in deltas]
        return {
            "orig": _aggregate(orig_vals),
            "recon": _aggregate(recon_vals),
            "abs_delta": _aggregate(abs_deltas),
            "delta": _aggregate(deltas),
        }

    return {
        "num_windows": num_windows,
        "num_windows_with_beats": num_windows_with_beats,
        "num_beats_matched": num_beats_matched,
        "r_amplitude": _paired_block(orig_ramp, recon_ramp),
        "qrs_width_ms": _paired_block(orig_qrs, recon_qrs),
        "st_deviation": _paired_block(orig_st, recon_st),
        "t_amplitude": _paired_block(orig_t, recon_t),
        "baseline_drift": _paired_block(orig_drift, recon_drift),
        "beat_shape_correlation": _aggregate(shape_corrs),
        "params": {
            "sample_rate": int(sample_rate),
            "timing_tolerance_ms": float(timing_tolerance_ms),
            "min_beats": int(min_beats),
        },
    }


__all__ = ["evaluate_ecg_morphology"]
