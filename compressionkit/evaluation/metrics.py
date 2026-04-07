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
) -> dict[str, float]:
    """Compute scalar reconstruction metrics on two aligned signals."""
    orig_flat = np.asarray(original, dtype=np.float32).reshape(-1)
    recon_flat = np.asarray(reconstructed, dtype=np.float32).reshape(-1)
    mse = float(np.mean((orig_flat - recon_flat) ** 2))
    mae = float(np.mean(np.abs(orig_flat - recon_flat)))
    denom = float(np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat) + 1e-8)
    cos_sim = float(np.dot(orig_flat, recon_flat) / denom)
    prd_num = float(np.sum((orig_flat - recon_flat) ** 2))
    prd_den = float(np.sum(orig_flat**2) + 1e-8)
    prd_percent = float(100.0 * np.sqrt(max(prd_num / prd_den, 0.0)))
    return {
        "mse": mse,
        "mae": mae,
        "cosine_similarity": cos_sim,
        "prd_percent": prd_percent,
    }


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
        rr_intervals = pk.ppg.compute_rr_intervals(peaks)
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
            "num_peaks": int(peaks.size),
        }
    except Exception:
        return None


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
        "num_total_pairs": int(len(per_sample)),
        "num_valid_pairs": int(valid_pairs),
        "target_mean_hr_bpm": float(np.mean(target_hr_values)),
        "reconstructed_mean_hr_bpm": float(np.mean(recon_hr_values)),
        "hr_mae_bpm": float(np.mean(hr_abs_errors)),
        "hr_bias_bpm": float(np.mean(hr_biases)),
        "target_mean_rmssd_ms": float(np.mean(target_rmssd_values)),
        "reconstructed_mean_rmssd_ms": float(np.mean(recon_rmssd_values)),
        "rmssd_mae_ms": float(np.mean(rmssd_abs_errors)),
        "target_mean_sdnn_ms": float(np.mean(target_sdnn_values)),
        "reconstructed_mean_sdnn_ms": float(np.mean(recon_sdnn_values)),
        "sdnn_mae_ms": float(np.mean(sdnn_abs_errors)),
    }
    return summary, per_sample


__all__ = [
    "PRD",
    "TruePRD",
    "compute_ppg_physiokit_metrics",
    "compute_signal_metrics",
    "summarize_physiokit_alignment",
]
