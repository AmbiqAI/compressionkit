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
from compressionkit.evaluation.metrics import compute_signal_metrics
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
) -> dict[str, Any]:
    """Evaluate every *method* over *num_recordings* val files.

    Returns a dict of per-method aggregate metrics plus a per-recording
    breakdown. The absence of a ``"stitching"`` key in the summary JSON
    is the signal that this evaluation was disabled.
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
    per_method: dict[str, dict[str, list[float]]] = {
        m: {"prd_percent": [], "cosine_similarity": [], "mse": [], "seam_ratio": [], "seam_rms": [], "non_seam_rms": []}
        for m in methods
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
        method_summary[m] = {
            "num_recordings": len(acc["prd_percent"]),
            "mean_prd_percent": _mean(acc["prd_percent"]),
            "mean_cosine_similarity": _mean(acc["cosine_similarity"]),
            "mean_mse": _mean(acc["mse"]),
            "mean_seam_ratio": _mean(acc["seam_ratio"]),
            "mean_seam_rms": _mean(acc["seam_rms"]),
            "mean_non_seam_rms": _mean(acc["non_seam_rms"]),
        }

    return {
        "duration_sec": float(duration_sec),
        "hop_ratio": float(hop_ratio),
        "frame_size": int(frame_size),
        "num_recordings_eval": len(per_recording),
        "methods": method_summary,
        "per_recording": per_recording,
    }


__all__ = ["evaluate_stitching"]
