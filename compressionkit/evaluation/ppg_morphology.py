"""PPG pulse-morphology metrics for compression evaluation.

Computes per-pulse morphology measurements on paired (original,
reconstructed) PPG windows and reports orig/recon/delta aggregates so a
codec's effect on clinically-relevant pulse shape can be quantified
independently of waveform PRD.

Designed to work on the same per-frame ``sample_*.csv`` artifacts the
scorecard already consumes. Pulse detection uses the physiokit PPG
pipeline (matching :func:`compute_ppg_physiokit_metrics`) so morphology
numbers are directly comparable to the existing HR/HRV physiology block.

Note on DC / AC-DC ratio:
    Per-frame ``sample_*.csv`` artifacts are typically z-normalized 2-sec
    windows. DC level and absolute AC amplitude are therefore not
    physically meaningful on them. We still report ``ac_amplitude_norm``
    (peak − trough in z-units) because the *relative* change between
    orig and recon is preserved by the z-norm. True perfusion-index
    (AC/DC) requires raw-scale signals; see ``ppg_decimate``-style
    evaluators for that workflow.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import physiokit as pk


# Tolerance for matching orig/recon peaks (samples), set at call time.

def _detect_peaks_clean(
    signal: np.ndarray,
    *,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bandpass + detect peaks via physiokit. Returns (cleaned, peaks)."""
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    cleaned = pk.ppg.clean(
        sig,
        lowcut=low_hz,
        highcut=high_hz,
        sample_rate=sample_rate,
        order=order,
    )
    peaks = pk.ppg.find_peaks(cleaned, sample_rate=sample_rate)
    if peaks is None:
        return cleaned, np.empty(0, dtype=int)
    return cleaned, np.asarray(peaks, dtype=int).reshape(-1)


def _trough_before(signal: np.ndarray, peak: int, search_samples: int) -> int | None:
    lo = max(0, peak - search_samples)
    if peak - lo < 2:
        return None
    return int(lo + np.argmin(signal[lo:peak]))


def _trough_after(signal: np.ndarray, peak: int, search_samples: int) -> int | None:
    hi = min(signal.size, peak + search_samples + 1)
    if hi - peak < 2:
        return None
    return int(peak + np.argmin(signal[peak:hi]))


def _detect_dicrotic_notch(
    signal: np.ndarray,
    peak: int,
    trough_after: int,
) -> int | None:
    """Locate a dicrotic notch (local minimum on the diastolic decay).

    Returns the sample index of the notch, or ``None`` if no convincing
    inflection is found between ``peak`` and ``trough_after``. The
    detector requires a downstream local minimum followed by a small
    local maximum (the diastolic peak), which is a robust signature.
    """
    if trough_after - peak < 6:
        return None
    seg = signal[peak:trough_after]
    if seg.size < 6:
        return None
    # Search the middle 60% of the diastolic decay for a local min, then
    # require a subsequent local max within the remaining segment.
    lo = int(0.15 * seg.size)
    hi = int(0.80 * seg.size)
    if hi - lo < 3:
        return None
    inner = seg[lo:hi]
    candidates = np.flatnonzero((inner[1:-1] < inner[:-2]) & (inner[1:-1] < inner[2:]))
    if candidates.size == 0:
        return None
    # Pick the deepest local min.
    best = candidates[np.argmin(inner[candidates + 1])] + 1
    notch_idx = lo + int(best)
    # Require a local max after the notch (diastolic peak).
    tail = seg[notch_idx + 1 :]
    if tail.size < 2 or float(tail.max() - seg[notch_idx]) < 0.02 * float(np.ptp(signal) + 1e-12):
        return None
    return int(peak + notch_idx)


def _measure_pulse(
    cleaned: np.ndarray,
    peak: int,
    *,
    sample_rate: int,
    max_pulse_ms: float,
) -> dict[str, float | int | None] | None:
    """Measure morphology around a single systolic peak."""
    search = max(2, int(max_pulse_ms * sample_rate / 1000.0))
    t_before = _trough_before(cleaned, peak, search)
    t_after = _trough_after(cleaned, peak, search)
    if t_before is None or t_after is None:
        return None
    if peak <= t_before or t_after <= peak:
        return None

    upstroke = cleaned[t_before : peak + 1]
    if upstroke.size < 3:
        return None
    ac_amp = float(cleaned[peak] - cleaned[t_before])
    # Slope in (signal units) per second.
    upstroke_diffs = np.diff(upstroke) * float(sample_rate)
    upstroke_slope = float(upstroke_diffs.max())
    pulse_width_sec = float((t_after - t_before) / sample_rate)
    rise_time_sec = float((peak - t_before) / sample_rate)
    fall_time_sec = float((t_after - peak) / sample_rate)
    notch_idx = _detect_dicrotic_notch(cleaned, peak, t_after)
    return {
        "trough_before": int(t_before),
        "peak": int(peak),
        "trough_after": int(t_after),
        "ac_amp": ac_amp,
        "upstroke_slope": upstroke_slope,
        "pulse_width_sec": pulse_width_sec,
        "rise_time_sec": rise_time_sec,
        "fall_time_sec": fall_time_sec,
        "has_notch": notch_idx is not None,
        "notch_idx": notch_idx,
    }


def _pulse_shape_corr(
    orig_clean: np.ndarray,
    recon_clean: np.ndarray,
    orig_pulse: dict[str, Any],
    recon_pulse: dict[str, Any],
    *,
    template_len: int = 64,
) -> float:
    """Pearson correlation of the two matched pulses on a common length."""
    o = orig_clean[orig_pulse["trough_before"] : orig_pulse["trough_after"] + 1]
    r = recon_clean[recon_pulse["trough_before"] : recon_pulse["trough_after"] + 1]
    if o.size < 4 or r.size < 4:
        return float("nan")
    xo = np.linspace(0.0, 1.0, o.size)
    xr = np.linspace(0.0, 1.0, r.size)
    xt = np.linspace(0.0, 1.0, template_len)
    o_rs = np.interp(xt, xo, o)
    r_rs = np.interp(xt, xr, r)
    o_z = o_rs - o_rs.mean()
    r_z = r_rs - r_rs.mean()
    denom = float(np.linalg.norm(o_z) * np.linalg.norm(r_z))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(o_z, r_z) / denom)


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


def evaluate_ppg_morphology(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    *,
    sample_rate: int,
    low_hz: float = 0.5,
    high_hz: float = 8.0,
    order: int = 3,
    min_pulses: int = 1,
    timing_tolerance_ms: float = 125.0,
    max_pulse_ms: float = 1500.0,
) -> dict[str, Any]:
    """Compute paired PPG morphology metrics on a batch of windows.

    Args:
        originals: Array of shape ``(N, T)`` with ground-truth PPG windows.
        reconstructions: Matched array of shape ``(N, T)``.
        sample_rate: Sample rate (Hz) of both arrays.
        low_hz, high_hz, order: physiokit bandpass parameters.
        min_pulses: Minimum number of valid pulses a window must contain
            on *both* sides to contribute to the aggregates.
        timing_tolerance_ms: Window for matching orig/recon peaks
            (mirrors :func:`summarize_ppg_peak_alignment`).
        max_pulse_ms: Search half-window for locating the trough before/
            after each systolic peak.

    Returns:
        Dict with keys ``num_windows``, ``num_windows_with_pulses``,
        ``num_pulses_matched``, ``ac_amplitude_norm``,
        ``upstroke_slope``, ``pulse_width_sec``, ``rise_time_sec``,
        ``fall_time_sec``, ``pulse_shape_correlation``,
        ``dicrotic_notch``, ``params``.

        Each metric block aggregates orig, recon, abs_delta, and signed
        delta across all matched pulses.
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

    tol_samples = max(1, round(timing_tolerance_ms * sample_rate / 1000.0))

    orig_ac: list[float] = []
    recon_ac: list[float] = []
    orig_slope: list[float] = []
    recon_slope: list[float] = []
    orig_width: list[float] = []
    recon_width: list[float] = []
    orig_rise: list[float] = []
    recon_rise: list[float] = []
    orig_fall: list[float] = []
    recon_fall: list[float] = []
    shape_corrs: list[float] = []

    # Notch contingency table (orig_has, recon_has).
    notch_tt = 0  # both present
    notch_tf = 0  # orig only (false negative)
    notch_ft = 0  # recon only (false positive)
    notch_ff = 0  # both absent

    num_windows = int(originals.shape[0])
    num_windows_with_pulses = 0
    num_pulses_matched = 0

    for orig, recon in zip(originals, reconstructions):
        try:
            o_clean, o_peaks = _detect_peaks_clean(
                orig, sample_rate=sample_rate, low_hz=low_hz, high_hz=high_hz, order=order
            )
            r_clean, r_peaks = _detect_peaks_clean(
                recon, sample_rate=sample_rate, low_hz=low_hz, high_hz=high_hz, order=order
            )
        except Exception:
            continue
        if o_peaks.size < min_pulses or r_peaks.size < min_pulses:
            continue

        o_pulses: list[dict[str, Any]] = []
        for p in o_peaks:
            m = _measure_pulse(o_clean, int(p), sample_rate=sample_rate, max_pulse_ms=max_pulse_ms)
            if m is not None:
                o_pulses.append(m)
        r_pulses: list[dict[str, Any]] = []
        for p in r_peaks:
            m = _measure_pulse(r_clean, int(p), sample_rate=sample_rate, max_pulse_ms=max_pulse_ms)
            if m is not None:
                r_pulses.append(m)
        if not o_pulses or not r_pulses:
            continue
        num_windows_with_pulses += 1

        # Greedy nearest-peak match within tolerance.
        used_r: set[int] = set()
        for op in o_pulses:
            deltas = np.array([abs(op["peak"] - rp["peak"]) for rp in r_pulses])
            valid = np.flatnonzero(deltas <= tol_samples)
            valid = [i for i in valid if i not in used_r]
            if not valid:
                continue
            best = int(min(valid, key=lambda i: deltas[i]))
            used_r.add(best)
            rp = r_pulses[best]
            num_pulses_matched += 1

            orig_ac.append(float(op["ac_amp"]))
            recon_ac.append(float(rp["ac_amp"]))
            orig_slope.append(float(op["upstroke_slope"]))
            recon_slope.append(float(rp["upstroke_slope"]))
            orig_width.append(float(op["pulse_width_sec"]))
            recon_width.append(float(rp["pulse_width_sec"]))
            orig_rise.append(float(op["rise_time_sec"]))
            recon_rise.append(float(rp["rise_time_sec"]))
            orig_fall.append(float(op["fall_time_sec"]))
            recon_fall.append(float(rp["fall_time_sec"]))
            shape_corrs.append(_pulse_shape_corr(o_clean, r_clean, op, rp))

            o_has = bool(op["has_notch"])
            r_has = bool(rp["has_notch"])
            if o_has and r_has:
                notch_tt += 1
            elif o_has and not r_has:
                notch_tf += 1
            elif not o_has and r_has:
                notch_ft += 1
            else:
                notch_ff += 1

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

    total_notch = notch_tt + notch_tf + notch_ft + notch_ff
    if total_notch:
        agreement = (notch_tt + notch_ff) / total_notch
        orig_rate = (notch_tt + notch_tf) / total_notch
        recon_rate = (notch_tt + notch_ft) / total_notch
    else:
        agreement = float("nan")
        orig_rate = float("nan")
        recon_rate = float("nan")

    return {
        "num_windows": num_windows,
        "num_windows_with_pulses": num_windows_with_pulses,
        "num_pulses_matched": num_pulses_matched,
        "ac_amplitude_norm": _paired_block(orig_ac, recon_ac),
        "upstroke_slope": _paired_block(orig_slope, recon_slope),
        "pulse_width_sec": _paired_block(orig_width, recon_width),
        "rise_time_sec": _paired_block(orig_rise, recon_rise),
        "fall_time_sec": _paired_block(orig_fall, recon_fall),
        "pulse_shape_correlation": _aggregate(shape_corrs),
        "dicrotic_notch": {
            "agreement": float(agreement) if np.isfinite(agreement) else None,
            "orig_detection_rate": float(orig_rate) if np.isfinite(orig_rate) else None,
            "recon_detection_rate": float(recon_rate) if np.isfinite(recon_rate) else None,
            "both_present": notch_tt,
            "orig_only": notch_tf,
            "recon_only": notch_ft,
            "both_absent": notch_ff,
        },
        "params": {
            "sample_rate": int(sample_rate),
            "low_hz": float(low_hz),
            "high_hz": float(high_hz),
            "order": int(order),
            "min_pulses": int(min_pulses),
            "timing_tolerance_ms": float(timing_tolerance_ms),
            "max_pulse_ms": float(max_pulse_ms),
        },
    }


__all__ = ["evaluate_ppg_morphology"]
