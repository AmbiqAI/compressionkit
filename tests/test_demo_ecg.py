"""Tests for browser-demo ECG sample generation."""

from __future__ import annotations

import json

import h5py
import numpy as np

from compressionkit.export.demo_ecg import export_ecg_demo_csvs, generate_ecg_demo_clips
from compressionkit.preprocessing.ecg import generate_synthetic_ecg_batch


def _write_mitdb_fixture(root, *, count: int = 2) -> None:
    """Write canonical MIT-BIH-shaped H5 records for source-selection tests."""
    for index in range(count):
        signal = generate_synthetic_ecg_batch(
            num_segments=1,
            signal_length=360 * 20,
            sample_rate=360,
            noise_multiplier=[0.05, 0.05],
            seed=index + 1,
        )[0]
        with h5py.File(root / f"{100 + index}.h5", "w") as handle:
            handle.create_dataset("data", data=np.stack([signal, signal]))
            handle.attrs["fs"] = 360
            handle.attrs["source"] = "mitdb"
            handle.attrs["patient_id"] = str(100 + index)


def test_generate_ecg_demo_clips_are_quality_gated(tmp_path) -> None:
    """Generated demo clips are continuous 256 Hz ECG signals with valid rhythm metrics."""
    _write_mitdb_fixture(tmp_path)
    clips = generate_ecg_demo_clips(tmp_path, num_clips=2, duration_seconds=10, sample_rate=256, seed=7)

    assert len(clips) == 2
    for clip in clips:
        assert clip.signal.shape == (2560,)
        assert clip.signal.dtype == np.float32
        assert 45.0 <= clip.quality.heart_rate_bpm <= 110.0
        assert clip.quality.num_r_peaks >= 8
        assert clip.quality.rr_cv <= 0.15
        assert clip.quality.clipping_fraction < 0.01
        assert clip.source_record in {"100", "101"}
        assert clip.source_sample_rate == 360


def test_export_ecg_demo_csvs_writes_manifest(tmp_path) -> None:
    """CSV export writes an inspectable clip manifest with quality metadata."""
    source_dir = tmp_path / "mitdb"
    source_dir.mkdir()
    _write_mitdb_fixture(source_dir)
    exported = export_ecg_demo_csvs(
        tmp_path / "output",
        dataset_dir=source_dir,
        num_clips=2,
        duration_seconds=10,
        sample_rate=256,
        seed=7,
    )

    assert len(exported.csv_paths) == 2
    rows = np.loadtxt(exported.csv_paths[0], delimiter=",", skiprows=1)
    assert rows.shape == (2560, 3)
    assert np.allclose(rows[:, 1], np.arange(2560) / 256.0)

    manifest = json.loads(exported.manifest_path.read_text())
    assert manifest["modality"] == "ecg"
    assert manifest["source"]["license"] == "ODC-By-1.0"
    assert manifest["sample_rate"] == 256
    assert len(manifest["clips"]) == 2
    assert manifest["clips"][0]["quality"]["num_r_peaks"] >= 8
