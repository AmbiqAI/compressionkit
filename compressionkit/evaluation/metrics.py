"""Reconstruction quality metrics for signal compression evaluation."""

from __future__ import annotations

from typing import Any

import keras
import numpy as np
import physiokit as pk


@keras.saving.register_keras_serializable(package="compression_kit")
class PRD(keras.metrics.Metric):
    """Percent RMS difference metric with optional energy normalization."""

    def __init__(self, normalized: bool = True, name: str = "prd", **kwargs):
        super().__init__(name=name, **kwargs)
        self.normalized = normalized
        self._num = self.add_weight(name="num", initializer="zeros")
        self._den = self.add_weight(name="den", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = keras.ops.cast(y_true, "float32")
        err = y_true_f - keras.ops.cast(y_pred, "float32")
        self._num.assign_add(keras.ops.sum(keras.ops.square(err)))
        if self.normalized:
            self._den.assign_add(keras.ops.sum(keras.ops.square(y_true_f)))
        else:
            self._den.assign_add(keras.ops.cast(keras.ops.size(y_true_f), "float32"))

    def result(self):
        ratio = self._num / (self._den + keras.ops.cast(1e-8, "float32"))
        return keras.ops.cast(100.0, "float32") * keras.ops.sqrt(
            keras.ops.maximum(ratio, keras.ops.cast(0.0, "float32"))
        )

    def reset_state(self):
        self._num.assign(0.0)
        self._den.assign(0.0)


@keras.saving.register_keras_serializable(package="compression_kit")
class TruePRD(PRD):
    """Normalized PRD metric (convenience alias)."""

    def __init__(self, name: str = "prd", **kwargs):
        super().__init__(normalized=True, name=name, **kwargs)


def compute_signal_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    noise_power: float | None = None,
) -> dict[str, float]:
    """Compute scalar reconstruction metrics on two aligned signals.

    Args:
        original: Ground-truth signal (any shape; flattened).
        reconstructed: Reconstructed signal (same total size).
        noise_power: Optional estimate of the *noise* contribution to the
            ground-truth power, in the same scale as ``mean(original**2)``.
            When provided, ``prdn_noise_percent`` is added to the result —
            this is PRD normalized by the *clean* signal power
            ``max(mean(orig^2) - noise_power, eps)`` rather than the raw
            signal power. It removes the unfair penalty applied to codecs
            that correctly remove noise from a noisy ground-truth.
    """
    orig_flat = np.asarray(original, dtype=np.float32).reshape(-1)
    recon_flat = np.asarray(reconstructed, dtype=np.float32).reshape(-1)
    mse = float(np.mean((orig_flat - recon_flat) ** 2))
    mae = float(np.mean(np.abs(orig_flat - recon_flat)))
    denom = float(np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat) + 1e-8)
    cos_sim = float(np.dot(orig_flat, recon_flat) / denom)
    sse = float(np.sum((orig_flat - recon_flat) ** 2))
    sig_pow_sum = float(np.sum(orig_flat**2))
    prd_percent = float(100.0 * np.sqrt(max(sse / (sig_pow_sum + 1e-8), 0.0)))
    out = {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "cosine_similarity": cos_sim,
        "prd_percent": prd_percent,
    }
    if noise_power is not None:
        # PRDN-noise: assume the reconstruction error decomposes as
        #   recon = clean + (recon - clean)         where ||clean - orig|| ~= noise
        # so E[sse] >= N * noise_power even for a perfect denoiser. We
        # subtract that expected noise energy from sse, and divide by the
        # clean-signal power, so a codec that perfectly reconstructs the
        # underlying clean signal scores ~0 even if the ground-truth was
        # noisy.
        n_samples = max(orig_flat.size, 1)
        noise_energy = float(noise_power) * n_samples
        adjusted_sse = max(sse - noise_energy, 0.0)
        clean_pow_total = max(sig_pow_sum - noise_energy, 1e-12)
        prdn = 100.0 * float(np.sqrt(max(adjusted_sse / clean_pow_total, 0.0)))
        out["prdn_noise_percent"] = prdn
        out["clean_signal_power"] = float(clean_pow_total / n_samples)
        out["noise_power_estimate"] = float(noise_power)
    return out


def compute_ppg_physiokit_metrics(
    signal: np.ndarray,
    *,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    order: int,
    min_peaks: int,
) -> dict[str, float] | None:
    """Compute HR/HRV metrics for one PPG signal using physiokit."""
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    if sig.size < 4:
        return None
    try:
        cleaned = pk.ppg.clean(
            sig, lowcut=low_hz, highcut=high_hz, sample_rate=sample_rate, order=order,
        )
        hr_bpm, hr_qos = pk.ppg.compute_heart_rate(cleaned, sample_rate=sample_rate, method="peak")
        peaks = pk.ppg.find_peaks(cleaned, sample_rate=sample_rate)
        if peaks is None:
            return None
        peaks = np.asarray(peaks).reshape(-1)
        if peaks.size < min_peaks:
            return None
        # Filter peaks by physiological RR bounds + quotient filter
        filtered_peaks = pk.ppg.filter_peaks(peaks, sample_rate=sample_rate)
        filtered_peaks = np.asarray(filtered_peaks).reshape(-1)
        if filtered_peaks.size < min_peaks:
            return None
        rr_intervals = pk.ppg.compute_rr_intervals(filtered_peaks)
        rr_intervals = np.asarray(rr_intervals).reshape(-1)
        if rr_intervals.size < max(2, min_peaks - 1):
            return None
        hrv_time = pk.hrv.compute_hrv_time(rr_intervals, sample_rate=sample_rate)
        return {
            "hr_bpm": float(hr_bpm),
            "hr_qos": float(hr_qos),
            "sdnn_ms": float(hrv_time.sd_nn),
            "rmssd_ms": float(hrv_time.rms_sd),
            "mean_nn_ms": float(hrv_time.mean_nn),
            "num_peaks": int(filtered_peaks.size),
            "peak_locations": filtered_peaks.astype(int).tolist(),
        }
    except Exception:
        return None


def _match_peaks_one_to_one(
    target_peaks: np.ndarray,
    recon_peaks: np.ndarray,
    *,
    tolerance_samples: int,
) -> list[tuple[int, int]]:
    """Greedily match target/reconstruction peaks within a sample tolerance."""
    if target_peaks.size == 0 or recon_peaks.size == 0:
        return []
    candidates: list[tuple[int, int, int]] = []
    for target_idx, target_peak in enumerate(target_peaks):
        deltas = np.abs(recon_peaks - target_peak)
        for recon_idx in np.flatnonzero(deltas <= tolerance_samples):
            candidates.append((int(deltas[recon_idx]), int(target_idx), int(recon_idx)))
    candidates.sort(key=lambda item: item[0])

    used_targets: set[int] = set()
    used_recons: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _delta, target_idx, recon_idx in candidates:
        if target_idx in used_targets or recon_idx in used_recons:
            continue
        used_targets.add(target_idx)
        used_recons.add(recon_idx)
        matches.append((target_idx, recon_idx))
    matches.sort(key=lambda item: item[0])
    return matches


def summarize_ppg_peak_alignment(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    *,
    sample_rate: int,
    low_hz: float = 0.5,
    high_hz: float = 8.0,
    order: int = 3,
    min_peaks: int = 5,
    timing_tolerance_ms: float = 125.0,
) -> tuple[dict[str, float] | None, list[dict[str, Any] | None]]:
    """Compare PPG pulse peak timing between paired original/reconstructed signals.

    Peaks are detected through the same physiokit PPG path used for HR/HRV,
    then matched one-to-one within ``timing_tolerance_ms``. This reports pulse
    preservation directly: precision catches extra invented pulses, recall
    catches missed pulses, and timing/IBI errors catch peak shifts that can
    degrade HRV even when waveform PRD is low.
    """
    per_sample: list[dict[str, Any] | None] = []
    peak_count_diffs: list[int] = []
    precision_vals: list[float] = []
    recall_vals: list[float] = []
    f1_vals: list[float] = []
    timing_errors_ms: list[float] = []
    ibi_errors_ms: list[float] = []
    total_target_peaks = 0
    total_recon_peaks = 0
    total_matched_peaks = 0
    total_missed_peaks = 0
    total_extra_peaks = 0
    tolerance_samples = max(1, int(round(timing_tolerance_ms * sample_rate / 1000.0)))

    for target, recon in zip(originals, reconstructions):
        target_metrics = compute_ppg_physiokit_metrics(
            target,
            sample_rate=sample_rate,
            low_hz=low_hz,
            high_hz=high_hz,
            order=order,
            min_peaks=min_peaks,
        )
        recon_metrics = compute_ppg_physiokit_metrics(
            recon,
            sample_rate=sample_rate,
            low_hz=low_hz,
            high_hz=high_hz,
            order=order,
            min_peaks=min_peaks,
        )
        if target_metrics is None or recon_metrics is None:
            per_sample.append(None)
            continue

        target_peaks = np.asarray(target_metrics.get("peak_locations", []), dtype=int).reshape(-1)
        recon_peaks = np.asarray(recon_metrics.get("peak_locations", []), dtype=int).reshape(-1)
        if target_peaks.size < min_peaks or recon_peaks.size < min_peaks:
            per_sample.append(None)
            continue

        matches = _match_peaks_one_to_one(
            target_peaks,
            recon_peaks,
            tolerance_samples=tolerance_samples,
        )
        matched = len(matches)
        target_count = int(target_peaks.size)
        recon_count = int(recon_peaks.size)
        total_target_peaks += target_count
        total_recon_peaks += recon_count
        total_matched_peaks += matched

        missed = target_count - matched
        extra = recon_count - matched
        total_missed_peaks += missed
        total_extra_peaks += extra
        count_diff = recon_count - target_count
        peak_count_diffs.append(count_diff)

        precision = matched / recon_count if recon_count else 0.0
        recall = matched / target_count if target_count else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)

        sample_timing_errors: list[float] = []
        for target_idx, recon_idx in matches:
            err_ms = abs(int(recon_peaks[recon_idx]) - int(target_peaks[target_idx]))
            err_ms = err_ms / sample_rate * 1000.0
            sample_timing_errors.append(float(err_ms))
            timing_errors_ms.append(float(err_ms))

        sample_ibi_errors: list[float] = []
        for (prev_target_idx, prev_recon_idx), (target_idx, recon_idx) in zip(matches, matches[1:]):
            if target_idx != prev_target_idx + 1:
                continue
            target_ibi_ms = (target_peaks[target_idx] - target_peaks[prev_target_idx]) / sample_rate * 1000.0
            recon_ibi_ms = (recon_peaks[recon_idx] - recon_peaks[prev_recon_idx]) / sample_rate * 1000.0
            ibi_err_ms = abs(float(recon_ibi_ms - target_ibi_ms))
            sample_ibi_errors.append(ibi_err_ms)
            ibi_errors_ms.append(ibi_err_ms)

        per_sample.append({
            "target_num_peaks": target_count,
            "reconstructed_num_peaks": recon_count,
            "matched_num_peaks": matched,
            "missed_peaks": missed,
            "extra_peaks": extra,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "peak_timing_mae_ms": float(np.mean(sample_timing_errors)) if sample_timing_errors else None,
            "ibi_mae_ms": float(np.mean(sample_ibi_errors)) if sample_ibi_errors else None,
        })

    if not precision_vals:
        return None, per_sample

    exact = sum(1 for d in peak_count_diffs if d == 0)
    summary: dict[str, float] = {
        "num_total_pairs": float(len(per_sample)),
        "num_valid_pairs": float(len(precision_vals)),
        "timing_tolerance_ms": float(timing_tolerance_ms),
        "timing_tolerance_samples": float(tolerance_samples),
        "target_total_peaks": float(total_target_peaks),
        "reconstructed_total_peaks": float(total_recon_peaks),
        "matched_total_peaks": float(total_matched_peaks),
        "total_missed_peaks": float(total_missed_peaks),
        "total_extra_peaks": float(total_extra_peaks),
        "peak_count_exact_match_pct": 100.0 * exact / len(peak_count_diffs),
        "peak_precision_pct": float(100.0 * np.mean(precision_vals)),
        "peak_recall_pct": float(100.0 * np.mean(recall_vals)),
        "peak_f1_pct": float(100.0 * np.mean(f1_vals)),
    }
    if timing_errors_ms:
        within_one_sample = sum(
            1 for err in timing_errors_ms if err <= (1000.0 / sample_rate)
        )
        summary.update({
            "peak_timing_mae_ms": float(np.mean(timing_errors_ms)),
            "peak_timing_median_ms": float(np.median(timing_errors_ms)),
            "peak_timing_p90_ms": float(np.percentile(timing_errors_ms, 90)),
            "peak_timing_max_ms": float(np.max(timing_errors_ms)),
            "peak_timing_within_1sample_pct": float(100.0 * within_one_sample / len(timing_errors_ms)),
        })
    if ibi_errors_ms:
        summary.update({
            "ibi_mae_ms": float(np.mean(ibi_errors_ms)),
            "ibi_median_ae_ms": float(np.median(ibi_errors_ms)),
            "ibi_p90_ae_ms": float(np.percentile(ibi_errors_ms, 90)),
        })
    return summary, per_sample


def summarize_physiokit_alignment(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    *,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    order: int,
    min_peaks: int,
) -> tuple[dict[str, float] | None, list[dict[str, Any] | None]]:
    """Compare physiokit HR/HRV metrics between original and reconstructed signals."""
    per_sample: list[dict[str, Any] | None] = []
    hr_abs_errors: list[float] = []
    hr_biases: list[float] = []
    rmssd_abs_errors: list[float] = []
    sdnn_abs_errors: list[float] = []
    target_hr_values: list[float] = []
    recon_hr_values: list[float] = []
    target_rmssd_values: list[float] = []
    recon_rmssd_values: list[float] = []
    target_sdnn_values: list[float] = []
    recon_sdnn_values: list[float] = []

    for target, recon in zip(originals, reconstructions):
        target_metrics = compute_ppg_physiokit_metrics(
            target, sample_rate=sample_rate, low_hz=low_hz, high_hz=high_hz,
            order=order, min_peaks=min_peaks,
        )
        recon_metrics = compute_ppg_physiokit_metrics(
            recon, sample_rate=sample_rate, low_hz=low_hz, high_hz=high_hz,
            order=order, min_peaks=min_peaks,
        )
        if target_metrics is None or recon_metrics is None:
            per_sample.append(None)
            continue

        hr_diff = recon_metrics["hr_bpm"] - target_metrics["hr_bpm"]
        rmssd_diff = recon_metrics["rmssd_ms"] - target_metrics["rmssd_ms"]
        sdnn_diff = recon_metrics["sdnn_ms"] - target_metrics["sdnn_ms"]
        target_hr_values.append(target_metrics["hr_bpm"])
        recon_hr_values.append(recon_metrics["hr_bpm"])
        target_rmssd_values.append(target_metrics["rmssd_ms"])
        recon_rmssd_values.append(recon_metrics["rmssd_ms"])
        target_sdnn_values.append(target_metrics["sdnn_ms"])
        recon_sdnn_values.append(recon_metrics["sdnn_ms"])
        hr_abs_errors.append(abs(hr_diff))
        hr_biases.append(hr_diff)
        rmssd_abs_errors.append(abs(rmssd_diff))
        sdnn_abs_errors.append(abs(sdnn_diff))
        per_sample.append({
            "target": target_metrics,
            "reconstructed": recon_metrics,
            "delta": {
                "hr_bpm": float(hr_diff),
                "rmssd_ms": float(rmssd_diff),
                "sdnn_ms": float(sdnn_diff),
            },
        })

    valid_pairs = len(hr_abs_errors)
    if valid_pairs == 0:
        return None, per_sample

    summary = {
        "num_total_pairs": len(per_sample),
        "num_valid_pairs": int(valid_pairs),
        "target_mean_hr_bpm": float(np.mean(target_hr_values)),
        "reconstructed_mean_hr_bpm": float(np.mean(recon_hr_values)),
        "hr_mae_bpm": float(np.mean(hr_abs_errors)),
        "hr_median_ae_bpm": float(np.median(hr_abs_errors)),
        "hr_bias_bpm": float(np.mean(hr_biases)),
        "target_mean_rmssd_ms": float(np.mean(target_rmssd_values)),
        "reconstructed_mean_rmssd_ms": float(np.mean(recon_rmssd_values)),
        "rmssd_mae_ms": float(np.mean(rmssd_abs_errors)),
        "rmssd_median_ae_ms": float(np.median(rmssd_abs_errors)),
        "target_mean_sdnn_ms": float(np.mean(target_sdnn_values)),
        "reconstructed_mean_sdnn_ms": float(np.mean(recon_sdnn_values)),
        "sdnn_mae_ms": float(np.mean(sdnn_abs_errors)),
        "sdnn_median_ae_ms": float(np.median(sdnn_abs_errors)),
    }
    return summary, per_sample


# ---------------------------------------------------------------------------
# ECG HR/HRV metrics (mirrors PPG flow but uses pk.ecg.* peak detection)
# ---------------------------------------------------------------------------


def compute_ecg_hr_hrv(
    signal: np.ndarray,
    *,
    sample_rate: int,
    min_peaks_for_hr: int = 2,
    min_peaks_for_hrv: int = 3,
) -> dict[str, Any] | None:
    """Detect R-peaks and compute HR/HRV from a single ECG signal.

    Returns ``None`` when there are too few peaks to compute even HR.
    Otherwise returns ``num_peaks``, ``hr_bpm``, ``mean_rr_ms``,
    ``peak_locations`` (sample indices), and — when at least
    ``min_peaks_for_hrv`` peaks are found — ``sdnn_ms`` and ``rmssd_ms``.
    """
    sig = np.asarray(signal, dtype=np.float32).ravel()
    if sig.size < sample_rate:
        return None
    try:
        peaks = pk.ecg.find_peaks(sig, sample_rate=sample_rate)
        peaks = np.asarray(peaks).ravel()
        if peaks.size < min_peaks_for_hr:
            return None
        rr = pk.ecg.compute_rr_intervals(peaks)
        rr = np.asarray(rr, dtype=np.float64).ravel()
        if rr.size < 1:
            return None
        mean_rr_s = float(np.mean(rr) / sample_rate)
        hr_bpm = float(60.0 / mean_rr_s) if mean_rr_s > 0 else 0.0
        result: dict[str, Any] = {
            "num_peaks": int(peaks.size),
            "hr_bpm": hr_bpm,
            "mean_rr_ms": float(mean_rr_s * 1000.0),
            "peak_locations": peaks.astype(int).tolist(),
        }
        if peaks.size >= min_peaks_for_hrv and rr.size >= 2:
            hrv = pk.hrv.compute_hrv_time(rr, sample_rate=sample_rate)
            result["sdnn_ms"] = float(hrv.sd_nn)
            result["rmssd_ms"] = float(hrv.rms_sd)
        return result
    except Exception:
        return None


def summarize_ecg_alignment(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    *,
    sample_rate: int,
    min_peaks_for_hr: int = 2,
    min_peaks_for_hrv: int = 3,
    timing_tolerance_ms: float = 10.0,
) -> tuple[dict[str, float] | None, list[dict[str, Any] | None]]:
    """Compare ECG HR/HRV/peak-timing between paired original and reconstructed signals.

    Args:
        originals: Iterable/array of ground-truth signals.
        reconstructions: Iterable/array of reconstructed signals (same N).
        sample_rate: Hz.
        min_peaks_for_hr: Minimum peaks required to report HR.
        min_peaks_for_hrv: Minimum peaks required to report HRV (SDNN/RMSSD).
        timing_tolerance_ms: Threshold for "peak timing within tolerance".

    Returns:
        ``(summary, per_sample)``. ``summary`` is ``None`` if no pair was
        valid. ``per_sample`` is aligned with the inputs and contains
        ``None`` for pairs where either side lacked peaks.
    """
    per_sample: list[dict[str, Any] | None] = []
    hr_abs_errors: list[float] = []
    hr_biases: list[float] = []
    peak_count_diffs: list[int] = []
    peak_timing_errors: list[float] = []
    sdnn_abs_errors: list[float] = []
    rmssd_abs_errors: list[float] = []
    missed_peaks = 0
    extra_peaks = 0

    for target, recon in zip(originals, reconstructions):
        tm = compute_ecg_hr_hrv(
            target, sample_rate=sample_rate,
            min_peaks_for_hr=min_peaks_for_hr,
            min_peaks_for_hrv=min_peaks_for_hrv,
        )
        rm = compute_ecg_hr_hrv(
            recon, sample_rate=sample_rate,
            min_peaks_for_hr=min_peaks_for_hr,
            min_peaks_for_hrv=min_peaks_for_hrv,
        )
        if tm is None or rm is None:
            per_sample.append(None)
            continue

        hr_diff = float(rm["hr_bpm"] - tm["hr_bpm"])
        hr_abs_errors.append(abs(hr_diff))
        hr_biases.append(hr_diff)

        count_diff = int(rm["num_peaks"] - tm["num_peaks"])
        peak_count_diffs.append(count_diff)
        if count_diff < 0:
            missed_peaks += abs(count_diff)
        elif count_diff > 0:
            extra_peaks += count_diff

        orig_locs = np.asarray(tm["peak_locations"])
        recon_locs = np.asarray(rm["peak_locations"])
        if orig_locs.size > 0 and recon_locs.size > 0:
            for op in orig_locs:
                idx = int(np.argmin(np.abs(recon_locs - op)))
                err_ms = abs(int(recon_locs[idx]) - int(op)) / sample_rate * 1000.0
                peak_timing_errors.append(float(err_ms))

        delta: dict[str, float] = {"hr_bpm": hr_diff}
        if "sdnn_ms" in tm and "sdnn_ms" in rm:
            sdnn_d = float(rm["sdnn_ms"] - tm["sdnn_ms"])
            rmssd_d = float(rm["rmssd_ms"] - tm["rmssd_ms"])
            sdnn_abs_errors.append(abs(sdnn_d))
            rmssd_abs_errors.append(abs(rmssd_d))
            delta["sdnn_ms"] = sdnn_d
            delta["rmssd_ms"] = rmssd_d

        per_sample.append({"target": tm, "reconstructed": rm, "delta": delta})

    if not hr_abs_errors:
        return None, per_sample

    summary: dict[str, float] = {
        "num_total_pairs": float(len(per_sample)),
        "num_valid_pairs": float(len(hr_abs_errors)),
        "hr_mae_bpm": float(np.mean(hr_abs_errors)),
        "hr_median_ae_bpm": float(np.median(hr_abs_errors)),
        "hr_std_ae_bpm": float(np.std(hr_abs_errors)),
        "hr_p90_ae_bpm": float(np.percentile(hr_abs_errors, 90)),
        "hr_max_ae_bpm": float(np.max(hr_abs_errors)),
        "hr_bias_bpm": float(np.mean(hr_biases)),
        "hr_bias_std_bpm": float(np.std(hr_biases)),
    }
    if peak_count_diffs:
        exact = sum(1 for d in peak_count_diffs if d == 0)
        summary["peak_count_exact_match_pct"] = 100.0 * exact / len(peak_count_diffs)
        summary["total_missed_peaks"] = float(missed_peaks)
        summary["total_extra_peaks"] = float(extra_peaks)
    if peak_timing_errors:
        within = sum(1 for t in peak_timing_errors if t <= timing_tolerance_ms)
        summary["peak_timing_mae_ms"] = float(np.mean(peak_timing_errors))
        summary["peak_timing_median_ms"] = float(np.median(peak_timing_errors))
        summary["peak_timing_std_ms"] = float(np.std(peak_timing_errors))
        summary["peak_timing_p90_ms"] = float(np.percentile(peak_timing_errors, 90))
        summary["peak_timing_max_ms"] = float(np.max(peak_timing_errors))
        summary[f"peak_timing_within_{timing_tolerance_ms:g}ms_pct"] = (
            100.0 * within / len(peak_timing_errors)
        )
    if sdnn_abs_errors:
        summary["sdnn_mae_ms"] = float(np.mean(sdnn_abs_errors))
        summary["sdnn_std_ae_ms"] = float(np.std(sdnn_abs_errors))
        summary["sdnn_p90_ae_ms"] = float(np.percentile(sdnn_abs_errors, 90))
        summary["rmssd_mae_ms"] = float(np.mean(rmssd_abs_errors))
        summary["rmssd_std_ae_ms"] = float(np.std(rmssd_abs_errors))
        summary["rmssd_p90_ae_ms"] = float(np.percentile(rmssd_abs_errors, 90))
    return summary, per_sample


__all__ = [
    "PRD",
    "TruePRD",
    "compute_ecg_hr_hrv",
    "compute_ppg_physiokit_metrics",
    "compute_signal_metrics",
    "summarize_ecg_alignment",
    "summarize_physiokit_alignment",
    "summarize_ppg_peak_alignment",
]
