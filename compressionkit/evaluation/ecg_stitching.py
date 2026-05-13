"""ECG long-recording stitching evaluation.

Runs each stitching strategy in :mod:`compressionkit.evaluation.stitching`
over a handful of full-length PTB-XL recordings, then reports both
signal-level reconstruction quality (PRD / cosine) and seam-discontinuity
metrics — the latter quantify how visible frame boundaries are in the
stitched output, which often matters more than whole-frame PRD for
downstream analyses like beat detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import keras
import numpy as np

from compressionkit.datasets.ecg import load_ecg_file_splits, load_ecg_signal
from compressionkit.evaluation.metrics import (
    compute_ecg_hr_hrv,
    compute_signal_metrics,
)
from compressionkit.evaluation.stitching import (
    STITCH_METHODS,
    seam_discontinuity_ratio,
    stitch,
)

logger = logging.getLogger("ecg-rvq-trainer")


def _make_predict_fn(model: keras.Model, batch_size: int):
    def _predict(batch: np.ndarray) -> np.ndarray:
        return model.predict(batch, batch_size=batch_size, verbose=0)

    return _predict


def evaluate_stitching(
    model: keras.Model,
    *,
    datasets_dir: Path,
    dataset_glob: str,
    frame_size: int,
    sample_rate: int,
    duration_sec: float,
    epsilon: float,
    methods: list[str],
    hop_ratio: float,
    num_recordings: int,
    batch_size: int = 32,
    seed: int = 42,
    lead_index: int = 1,
    seam_radius: int = 4,
    hr_hrv: bool = True,
) -> dict[str, Any]:
    """Evaluate every *method* over *num_recordings* val files.

    Returns a dict of per-method aggregate metrics plus a per-recording
    breakdown. The absence of a ``"stitching"`` key in the summary JSON
    is the signal that this evaluation was disabled.

    When ``hr_hrv`` is true, R-peak-derived HR/HRV are computed on each
    stitched trace and the original recording, and per-method aggregates
    (HR MAE/bias, SDNN/RMSSD MAE) are reported alongside the signal-level
    metrics. This is the long-recording counterpart to per-window
    :func:`summarize_ecg_alignment` and addresses issue #3.
    """
    _, val_files, _ = load_ecg_file_splits(datasets_dir, dataset_glob, seed=seed)
    if not val_files:
        logger.warning("No validation files found for stitching evaluation.")
        return {"methods": {}, "per_recording": []}

    rng = np.random.default_rng(seed)
    if len(val_files) > num_recordings:
        indices = rng.choice(len(val_files), size=num_recordings, replace=False)
        selected = [val_files[i] for i in sorted(indices)]
    else:
        selected = val_files

    unknown = [m for m in methods if m not in STITCH_METHODS]
    if unknown:
        raise ValueError(f"Unknown stitching methods: {unknown}. Known: {sorted(STITCH_METHODS)}")

    target_samples = int(duration_sec * sample_rate)
    predict_fn = _make_predict_fn(model, batch_size)

    # Per-method accumulators
    _quality_keys = ["prd_percent", "cosine_similarity", "mse", "seam_ratio", "seam_rms", "non_seam_rms"]
    _hr_keys = ["hr_abs_err", "hr_bias", "sdnn_abs_err", "rmssd_abs_err", "peak_count_diff"]
    per_method: dict[str, dict[str, list[float]]] = {
        m: {k: [] for k in _quality_keys + (_hr_keys if hr_hrv else [])} for m in methods
    }
    per_recording: list[dict[str, Any]] = []

    for fpath in selected:
        try:
            signal = load_ecg_signal(Path(fpath), lead_index=lead_index)
        except Exception as exc:
            logger.debug("Skipping %s: %s", fpath.name, exc)
            continue
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        if signal.size < frame_size * 2:
            continue
        if signal.size > target_samples:
            signal = signal[:target_samples]

        rec_entry: dict[str, Any] = {"file": Path(fpath).name, "length": int(signal.size), "methods": {}}

        # HR/HRV on the original signal is shared across stitching methods.
        orig_hr = compute_ecg_hr_hrv(signal, sample_rate=sample_rate) if hr_hrv else None
        if hr_hrv:
            rec_entry["original_ecg"] = (
                {
                    "hr_bpm": orig_hr["hr_bpm"],
                    "num_peaks": orig_hr["num_peaks"],
                    "sdnn_ms": orig_hr.get("sdnn_ms"),
                    "rmssd_ms": orig_hr.get("rmssd_ms"),
                }
                if orig_hr is not None
                else None
            )

        for method in methods:
            kwargs: dict[str, Any] = {"epsilon": epsilon}
            if method != "hard_concat":
                kwargs["hop_ratio"] = hop_ratio

            recon = stitch(method, predict_fn, signal, frame_size, **kwargs)
            sig_metrics = compute_signal_metrics(signal, recon)
            # Seams only make sense for the actual stride used during reconstruction
            effective_hop = 1.0 if method == "hard_concat" else hop_ratio
            seam = seam_discontinuity_ratio(
                recon,
                frame_size=frame_size,
                hop_ratio=effective_hop,
                radius=seam_radius,
            )
            entry = {
                "prd_percent": sig_metrics["prd_percent"],
                "cosine_similarity": sig_metrics["cosine_similarity"],
                "mse": sig_metrics["mse"],
                "seam_ratio": seam["ratio"],
                "seam_rms": seam["seam_rms"],
                "non_seam_rms": seam["non_seam_rms"],
                "num_seams": seam["num_seams"],
            }

            if hr_hrv:
                recon_hr = compute_ecg_hr_hrv(recon, sample_rate=sample_rate)
                hr_block: dict[str, Any] = {
                    "recon": (
                        {
                            "hr_bpm": recon_hr["hr_bpm"],
                            "num_peaks": recon_hr["num_peaks"],
                            "sdnn_ms": recon_hr.get("sdnn_ms"),
                            "rmssd_ms": recon_hr.get("rmssd_ms"),
                        }
                        if recon_hr is not None
                        else None
                    ),
                    "delta": None,
                }
                if orig_hr is not None and recon_hr is not None:
                    hr_diff = float(recon_hr["hr_bpm"] - orig_hr["hr_bpm"])
                    peak_count_diff = int(recon_hr["num_peaks"] - orig_hr["num_peaks"])
                    delta: dict[str, float] = {"hr_bpm": hr_diff, "peak_count_diff": float(peak_count_diff)}
                    per_method[method]["hr_abs_err"].append(abs(hr_diff))
                    per_method[method]["hr_bias"].append(hr_diff)
                    per_method[method]["peak_count_diff"].append(float(peak_count_diff))
                    if "sdnn_ms" in orig_hr and "sdnn_ms" in recon_hr:
                        sdnn_d = float(recon_hr["sdnn_ms"] - orig_hr["sdnn_ms"])
                        rmssd_d = float(recon_hr["rmssd_ms"] - orig_hr["rmssd_ms"])
                        delta["sdnn_ms"] = sdnn_d
                        delta["rmssd_ms"] = rmssd_d
                        per_method[method]["sdnn_abs_err"].append(abs(sdnn_d))
                        per_method[method]["rmssd_abs_err"].append(abs(rmssd_d))
                    hr_block["delta"] = delta
                entry["hr_hrv"] = hr_block

            rec_entry["methods"][method] = entry

            acc = per_method[method]
            acc["prd_percent"].append(entry["prd_percent"])
            acc["cosine_similarity"].append(entry["cosine_similarity"])
            acc["mse"].append(entry["mse"])
            if np.isfinite(entry["seam_ratio"]):
                acc["seam_ratio"].append(entry["seam_ratio"])
                acc["seam_rms"].append(entry["seam_rms"])
                acc["non_seam_rms"].append(entry["non_seam_rms"])

        per_recording.append(rec_entry)

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    method_summary: dict[str, dict[str, float | int]] = {}
    for m, acc in per_method.items():
        block: dict[str, float | int] = {
            "num_recordings": len(acc["prd_percent"]),
            "mean_prd_percent": _mean(acc["prd_percent"]),
            "mean_cosine_similarity": _mean(acc["cosine_similarity"]),
            "mean_mse": _mean(acc["mse"]),
            "mean_seam_ratio": _mean(acc["seam_ratio"]),
            "mean_seam_rms": _mean(acc["seam_rms"]),
            "mean_non_seam_rms": _mean(acc["non_seam_rms"]),
        }
        if hr_hrv:
            block["hr_mae_bpm"] = _mean(acc["hr_abs_err"])
            block["hr_bias_bpm"] = _mean(acc["hr_bias"])
            block["sdnn_mae_ms"] = _mean(acc["sdnn_abs_err"])
            block["rmssd_mae_ms"] = _mean(acc["rmssd_abs_err"])
            block["mean_peak_count_diff"] = _mean(acc["peak_count_diff"])
            block["num_hr_valid"] = len(acc["hr_abs_err"])
        method_summary[m] = block

    return {
        "duration_sec": float(duration_sec),
        "hop_ratio": float(hop_ratio),
        "frame_size": int(frame_size),
        "num_recordings_eval": len(per_recording),
        "methods": method_summary,
        "per_recording": per_recording,
    }


__all__ = ["evaluate_stitching"]
