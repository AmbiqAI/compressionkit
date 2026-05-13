"""Tests for HR/HRV aggregation in the ECG long-recording stitching evaluator (#3).

The evaluator now reports per-method HR MAE / bias / SDNN / RMSSD on the
stitched trace in addition to signal-level PRD and seam metrics. These
tests exercise the aggregation logic with a synthetic ECG file and an
identity ``predict_fn`` (no model dependency).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from compressionkit.evaluation.ecg_stitching import evaluate_stitching
from compressionkit.trainers.common import write_long_recording_eval


def _synthetic_ecg(num_samples: int, sample_rate: int, hr_bpm: float = 72.0) -> np.ndarray:
    """Generate a periodic gaussian-pulse ECG-like trace with steady HR."""
    rng = np.random.default_rng(0)
    period = sample_rate * 60.0 / hr_bpm
    t = np.arange(num_samples, dtype=np.float64)
    centers = np.arange(period * 0.5, num_samples, period)
    signal = np.zeros(num_samples, dtype=np.float32)
    width = sample_rate * 0.02
    for c in centers:
        signal += np.exp(-0.5 * ((t - c) / width) ** 2).astype(np.float32)
    signal += rng.normal(0, 0.01, num_samples).astype(np.float32)
    return signal


def _write_h5(path: Path, signal: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=signal[None, :])  # (1, T) channel-first


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    sample_rate = 256
    duration = 12.0  # long enough to contain a robust HR estimate
    n = int(duration * sample_rate)
    for i in range(10):
        _write_h5(tmp_path / f"rec_{i}.h5", _synthetic_ecg(n, sample_rate, hr_bpm=72.0 + i))
    return tmp_path


def _identity_predict(batch: np.ndarray) -> np.ndarray:
    return batch.astype(np.float32)


def test_evaluate_stitching_reports_hr_hrv(synthetic_dataset: Path):
    sample_rate = 256
    # Patch the predict path: the evaluator builds it from a keras model, so
    # short-circuit ``_make_predict_fn`` to bypass keras.
    with patch(
        "compressionkit.evaluation.ecg_stitching._make_predict_fn",
        return_value=_identity_predict,
    ):
        results = evaluate_stitching(
            model=None,  # type: ignore[arg-type]
            datasets_dir=synthetic_dataset,
            dataset_glob="*.h5",
            frame_size=256,
            sample_rate=sample_rate,
            duration_sec=10.0,
            epsilon=1e-3,
            methods=["overlap_add", "hard_concat"],
            hop_ratio=0.5,
            num_recordings=2,
            batch_size=8,
            seed=0,
            lead_index=0,
            seam_radius=4,
            hr_hrv=True,
        )

    assert results["num_recordings_eval"] == 2
    for method in ("overlap_add", "hard_concat"):
        block = results["methods"][method]
        assert "hr_mae_bpm" in block
        assert "hr_bias_bpm" in block
        assert "sdnn_mae_ms" in block
        assert "rmssd_mae_ms" in block
        # Identity predict ⇒ HR error must be ~0.
        assert abs(block["hr_mae_bpm"]) < 1.0
        assert block["num_hr_valid"] >= 1

    # Per-recording entries carry the HR/HRV block.
    rec0 = results["per_recording"][0]
    assert "original_ecg" in rec0
    for method in ("overlap_add", "hard_concat"):
        assert "hr_hrv" in rec0["methods"][method]


def test_evaluate_stitching_hr_hrv_disabled(synthetic_dataset: Path):
    sample_rate = 256
    with patch(
        "compressionkit.evaluation.ecg_stitching._make_predict_fn",
        return_value=_identity_predict,
    ):
        results = evaluate_stitching(
            model=None,  # type: ignore[arg-type]
            datasets_dir=synthetic_dataset,
            dataset_glob="*.h5",
            frame_size=256,
            sample_rate=sample_rate,
            duration_sec=8.0,
            epsilon=1e-3,
            methods=["overlap_add"],
            hop_ratio=0.5,
            num_recordings=1,
            seed=0,
            lead_index=0,
            hr_hrv=False,
        )
    block = results["methods"]["overlap_add"]
    assert "hr_mae_bpm" not in block
    assert "original_ecg" not in results["per_recording"][0]


def test_write_long_recording_eval_round_trip(tmp_path: Path):
    payload = {
        "modality": "ecg",
        "stitching": {
            "methods": {
                "overlap_add": {"hr_mae_bpm": 0.1, "num_recordings": 3},
            },
            "per_recording": [],
        },
    }
    path = write_long_recording_eval(payload, tmp_path)
    assert path == tmp_path / "long_recording_eval.json"
    assert json.loads(path.read_text())["modality"] == "ecg"
