"""Shared helpers used by PPG and ECG training recipes.

These are small, signal-agnostic pieces that every recipe needs: setting up
the run directory, picking best weights off disk, saving model artifacts,
extracting per-epoch metrics from a Keras ``History`` object, and writing a
summary JSON. Everything signal- or architecture-specific lives in the
per-signal trainer modules next to these helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

logger = logging.getLogger("compressionkit.trainers")


# ---------------------------------------------------------------------------
# Run directory / config persistence
# ---------------------------------------------------------------------------


def setup_run_dir(results_root: str | Path, run_name: str) -> Path:
    """Create (if needed) and return the canonical ``results/<run_name>/`` directory."""
    root = Path(results_root).resolve()
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(cfg_dump: dict[str, Any], run_dir: Path) -> None:
    """Write the validated configuration snapshot alongside other artifacts."""
    with (run_dir / "config.json").open("w") as f:
        json.dump(cfg_dump, f, indent=2)


# ---------------------------------------------------------------------------
# Post-fit model handling
# ---------------------------------------------------------------------------


BEST_CKPT_NAME = "best_model.weights.h5"


def reload_best_weights(model: keras.Model, run_dir: Path) -> None:
    """Restore the best checkpoint into *model* if one was written."""
    ckpt = run_dir / BEST_CKPT_NAME
    if ckpt.exists():
        model.load_weights(ckpt)
    else:
        logger.warning("Best checkpoint not found at %s; keeping in-memory weights.", ckpt)


def save_model_artifacts(model: keras.Model, run_dir: Path) -> dict[str, str]:
    """Save encoder, decoder, full weights, and RVQ codebooks to *run_dir*.

    Returns a dict of artifact basenames so callers can drop it straight into
    the run summary.
    """
    model.encoder.save(run_dir / "encoder.keras")
    model.decoder.save(run_dir / "decoder.keras")
    model.save_weights(run_dir / "model.weights.h5")
    rvq_weights = model.vq.get_weights()
    np.savez(run_dir / "rvq_weights.npz", *rvq_weights)
    return {
        "model": "model.keras",
        "encoder": "encoder.keras",
        "decoder": "decoder.keras",
        "best_model": BEST_CKPT_NAME,
        "rvq_weights": "rvq_weights.npz",
    }


# ---------------------------------------------------------------------------
# Representative-dataset helper (for TFLite INT8 calibration)
# ---------------------------------------------------------------------------


def collect_rep_dataset(
    val_ds: tf.data.Dataset,
    *,
    num_batches: int,
    fallback: np.ndarray,
) -> np.ndarray:
    """Pull *num_batches* batches from ``val_ds`` for INT8 calibration.

    Falls back to *fallback* (typically the evaluation sample inputs) if the
    dataset yields nothing.
    """
    batches: list[np.ndarray] = []
    for batch, _ in val_ds.take(max(1, num_batches)):
        batches.append(batch.numpy())
    if batches:
        return np.concatenate(batches, axis=0)
    return fallback.astype(np.float32)


def collect_disjoint_quantization_datasets(
    val_ds: tf.data.Dataset,
    *,
    calibration_frames: int,
    validation_frames: int,
    sampling_pool_frames: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reservoir-sample disjoint calibration and holdout frame partitions.

    The complete validation split is scanned up to ``sampling_pool_frames``.
    Reservoir sampling makes the retained frames representative without holding
    the full split in memory, then a seeded shuffle creates non-overlapping
    calibration and post-export validation partitions.

    Args:
        val_ds: Batched validation dataset yielding ``(inputs, targets)``.
        calibration_frames: Number of frames to provide to the TFLite converter.
        validation_frames: Number of different frames for post-export parity.
        sampling_pool_frames: Maximum source frames to scan before sampling.
        seed: Deterministic sampling seed.

    Returns:
        ``(calibration_frames, validation_frames)`` arrays.

    Raises:
        ValueError: If the validation stream cannot supply both partitions.
    """
    required = calibration_frames + validation_frames
    if calibration_frames <= 0 or validation_frames <= 0:
        raise ValueError("Calibration and validation frame counts must be positive")
    if sampling_pool_frames < required:
        raise ValueError("sampling_pool_frames must cover both quantization partitions")

    rng = np.random.default_rng(seed)
    reservoir: np.ndarray | None = None
    frames_seen = 0
    for inputs, _ in val_ds:
        batch = np.asarray(inputs.numpy(), dtype=np.float32)
        if reservoir is None:
            reservoir = np.empty((required, *batch.shape[1:]), dtype=np.float32)
        for frame in batch:
            if frames_seen < required:
                reservoir[frames_seen] = frame
            else:
                selected = int(rng.integers(0, frames_seen + 1))
                if selected < required:
                    reservoir[selected] = frame
            frames_seen += 1
            if frames_seen >= sampling_pool_frames:
                break
        if frames_seen >= sampling_pool_frames:
            break

    if reservoir is None or frames_seen < required:
        raise ValueError(
            f"Validation stream yielded {frames_seen} frames; need {required} disjoint quantization frames"
        )
    rng.shuffle(reservoir)
    return reservoir[:calibration_frames].copy(), reservoir[calibration_frames:].copy()


# ---------------------------------------------------------------------------
# Training history → scalar summary
# ---------------------------------------------------------------------------


def extract_history_metrics(
    history_dict: dict[str, list[float]],
    *,
    selection_metric: str,
) -> tuple[int, dict[str, float], dict[str, float], str]:
    """Pick the best epoch from a Keras ``History.history`` dict.

    Returns:
        ``(best_epoch_1_indexed, best_metrics, final_metrics, resolved_selection_metric)``.
        Falls back to ``"val_loss"`` if the requested selection metric is missing.
    """
    resolved = selection_metric
    if resolved not in history_dict:
        logger.warning("Selection metric '%s' not in history; falling back to 'val_loss'.", resolved)
        resolved = "val_loss"

    series = np.asarray(history_dict[resolved], dtype=np.float64)
    best_idx = int(np.argmin(series))
    best_epoch = best_idx + 1

    best_metrics = {
        key: float(values[best_idx])
        for key, values in history_dict.items()
        if isinstance(values, list) and len(values) > best_idx
    }
    final_metrics = {
        "final_loss": float(history_dict["loss"][-1]),
        "final_val_loss": float(history_dict["val_loss"][-1]),
        "final_val_mse": float(history_dict.get("val_mse", [0.0])[-1]),
    }
    return best_epoch, best_metrics, final_metrics, resolved


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


def write_summary(summary: dict[str, Any], run_dir: Path) -> Path:
    """Serialize *summary* to ``run_dir/summary.json`` and return the path."""
    path = run_dir / "summary.json"
    with path.open("w") as f:
        json.dump(summary, f, indent=2)
    return path


def write_long_recording_eval(payload: dict[str, Any], run_dir: Path) -> Path:
    """Serialize *payload* to ``run_dir/long_recording_eval.json``.

    Centralizes the long-recording report (HR/HRV on stitched traces)
    referenced by issue #3. Returns the written path.
    """
    path = run_dir / "long_recording_eval.json"
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=float)
    return path


__all__ = [
    "BEST_CKPT_NAME",
    "collect_disjoint_quantization_datasets",
    "collect_rep_dataset",
    "extract_history_metrics",
    "reload_best_weights",
    "save_config_snapshot",
    "save_model_artifacts",
    "setup_run_dir",
    "write_long_recording_eval",
    "write_summary",
]
