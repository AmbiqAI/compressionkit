"""Long-recording overlap-add reconstruction and evaluation.

Thin PPG-specific wrapper around :mod:`compressionkit.evaluation.stitching`.
The core stitching logic lives there and is shared with ECG; this module
drives it over real EDF recordings and computes HR/HRV metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import keras
import numpy as np

from compressionkit.datasets.ppg import load_ppg_file_splits, load_ppg_signal
from compressionkit.evaluation.metrics import (
    compute_ppg_physiokit_metrics,
    compute_signal_metrics,
)
from compressionkit.evaluation.stitching import (
    reconstruct_overlap_add as _stitch_overlap_add,
)

logger = logging.getLogger("ppg-rvq-trainer")


# ---------------------------------------------------------------------------
# Core overlap-add reconstruction
# ---------------------------------------------------------------------------


def reconstruct_overlap_add(
    model: keras.Model,
    signal: np.ndarray,
    frame_size: int,
    *,
    epsilon: float = 1e-3,
    hop_ratio: float = 0.5,
    batch_size: int = 32,
) -> np.ndarray:
    """Reconstruct a long 1D signal via Hann-window overlap-add.

    Thin wrapper around
    :func:`compressionkit.evaluation.stitching.reconstruct_overlap_add`
    that adapts a Keras model into a ``predict_fn`` callable. Preserved
    for backwards compatibility with existing PPG trainers and scripts.
    """

    def _predict(batch: np.ndarray) -> np.ndarray:
        return model.predict(batch, batch_size=batch_size, verbose=0)

    return _stitch_overlap_add(
        _predict,
        signal,
        frame_size,
        epsilon=epsilon,
        hop_ratio=hop_ratio,
    )


# ---------------------------------------------------------------------------
# Long-recording evaluation driver
# ---------------------------------------------------------------------------


def evaluate_long_recordings(
    model: keras.Model,
    *,
    datasets_dir: Path,
    dataset_glob: str,
    frame_size: int,
    sample_rate: int,
    target_label: str,
    offset_samples: int,
    duration_sec: float,
    epsilon: float,
    hop_ratio: float,
    num_recordings: int,
    physiokit_low_hz: float,
    physiokit_high_hz: float,
    physiokit_order: int,
    physiokit_min_peaks: int,
    orig_max_sdnn_ms: float = 300.0,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[dict[str, Any] | None, list[dict[str, Any] | None]]:
    """Evaluate HR/HRV on long overlap-add reconstructions from val EDF files.

    Args:
        model: Trained VQ autoencoder.
        datasets_dir: Root directory containing EDF files.
        dataset_glob: Glob pattern for EDF files.
        frame_size: Model frame size in samples.
        sample_rate: Target sample rate in Hz.
        target_label: EDF channel label (e.g. ``"Pleth"``).
        offset_samples: Samples to skip at the start of each recording.
        duration_sec: Duration of each evaluation segment in seconds.
        epsilon: LayerNorm epsilon.
        hop_ratio: Overlap-add hop ratio.
        num_recordings: Number of val recordings to evaluate.
        physiokit_low_hz: PhysioKit bandpass low frequency.
        physiokit_high_hz: PhysioKit bandpass high frequency.
        physiokit_order: PhysioKit bandpass filter order.
        physiokit_min_peaks: Minimum peaks required for valid HR/HRV.
        orig_max_sdnn_ms: Maximum SDNN (ms) on the **original** signal for a
            recording to be included in the summary.  Rejects recordings with
            poor peak detection / implausible HRV.  Set ``0`` to disable.
        batch_size: Inference batch size.
        seed: Random seed for file selection.

    Returns:
        ``(summary_dict, per_recording_list)`` — summary is ``None`` when no
        valid pairs could be evaluated.
    """
    _, val_files, _ = load_ppg_file_splits(
        datasets_dir,
        dataset_glob,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    if len(val_files) > num_recordings:
        indices = rng.choice(len(val_files), size=num_recordings, replace=False)
        selected_files = [val_files[i] for i in sorted(indices)]
    else:
        selected_files = val_files

    num_samples = int(duration_sec * sample_rate)
    # Scale min_peaks with duration: require at least 0.5 peaks/sec
    # (conservative — even 40 bpm gives ~0.67 peaks/sec)
    effective_min_peaks = max(physiokit_min_peaks, int(duration_sec * 0.5))
    pk_kwargs = {
        "sample_rate": sample_rate,
        "low_hz": physiokit_low_hz,
        "high_hz": physiokit_high_hz,
        "order": physiokit_order,
        "min_peaks": effective_min_peaks,
    }

    per_recording: list[dict[str, Any] | None] = []
    hr_abs_errors: list[float] = []
    hr_biases: list[float] = []
    sdnn_abs_errors: list[float] = []
    rmssd_abs_errors: list[float] = []

    for fpath in selected_files:
        try:
            raw_signal = load_ppg_signal(
                fpath,
                target_rate=sample_rate,
                offset_samples=offset_samples,
                num_samples=num_samples,
                target_label=target_label,
            )
        except Exception as exc:
            logger.debug("Skipping %s: %s", fpath.name, exc)
            per_recording.append(None)
            continue

        recon_signal = reconstruct_overlap_add(
            model,
            raw_signal,
            frame_size,
            epsilon=epsilon,
            hop_ratio=hop_ratio,
            batch_size=batch_size,
        )

        # Signal-level metrics
        sig_metrics = compute_signal_metrics(raw_signal, recon_signal)

        # PhysioKit HR/HRV on original and reconstructed
        orig_pk = compute_ppg_physiokit_metrics(raw_signal, **pk_kwargs)
        recon_pk = compute_ppg_physiokit_metrics(recon_signal, **pk_kwargs)

        if orig_pk is None or recon_pk is None:
            per_recording.append(
                {
                    "file": fpath.name,
                    "signal_metrics": sig_metrics,
                    "original_physiokit": orig_pk,
                    "reconstructed_physiokit": recon_pk,
                    "delta": None,
                    "quality_rejected": False,
                }
            )
            continue

        hr_diff = recon_pk["hr_bpm"] - orig_pk["hr_bpm"]
        sdnn_diff = recon_pk["sdnn_ms"] - orig_pk["sdnn_ms"]
        rmssd_diff = recon_pk["rmssd_ms"] - orig_pk["rmssd_ms"]

        # Quality gate: reject recordings where original HRV is implausible
        quality_ok = orig_max_sdnn_ms <= 0 or orig_pk["sdnn_ms"] <= orig_max_sdnn_ms
        if quality_ok:
            hr_abs_errors.append(abs(hr_diff))
            hr_biases.append(hr_diff)
            sdnn_abs_errors.append(abs(sdnn_diff))
            rmssd_abs_errors.append(abs(rmssd_diff))

        per_recording.append(
            {
                "file": fpath.name,
                "signal_metrics": sig_metrics,
                "original_physiokit": orig_pk,
                "reconstructed_physiokit": recon_pk,
                "delta": {
                    "hr_bpm": float(hr_diff),
                    "sdnn_ms": float(sdnn_diff),
                    "rmssd_ms": float(rmssd_diff),
                },
                "quality_rejected": not quality_ok,
            }
        )

    valid = len(hr_abs_errors)
    if valid == 0:
        return None, per_recording

    summary: dict[str, Any] = {
        "duration_sec": duration_sec,
        "hop_ratio": hop_ratio,
        "orig_max_sdnn_ms": orig_max_sdnn_ms,
        "num_total_recordings": len(per_recording),
        "num_valid_recordings": valid,
        "hr_mae_bpm": float(np.mean(hr_abs_errors)),
        "hr_median_ae_bpm": float(np.median(hr_abs_errors)),
        "hr_bias_bpm": float(np.mean(hr_biases)),
        "sdnn_mae_ms": float(np.mean(sdnn_abs_errors)),
        "sdnn_median_ae_ms": float(np.median(sdnn_abs_errors)),
        "rmssd_mae_ms": float(np.mean(rmssd_abs_errors)),
        "rmssd_median_ae_ms": float(np.median(rmssd_abs_errors)),
    }
    return summary, per_recording


__all__ = [
    "evaluate_long_recordings",
    "reconstruct_overlap_add",
]
