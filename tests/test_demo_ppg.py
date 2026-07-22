"""Tests for real BIDMC PPG browser-demo selection."""

from __future__ import annotations

import h5py
import numpy as np

from compressionkit.export.demo_ppg import generate_ppg_demo_clips


def _write_bidmc_fixture(root, *, count: int = 2) -> None:
    """Write canonical BIDMC-shaped H5 records with clean pulse waveforms."""
    sample_rate = 125
    times = np.arange(sample_rate * 20, dtype=np.float32) / sample_rate
    for index in range(count):
        frequency = 1.2 + index * 0.1
        signal = np.sin(2 * np.pi * frequency * times) + 0.3 * np.sin(4 * np.pi * frequency * times)
        with h5py.File(root / f"bidmc{index:02d}.h5", "w") as handle:
            handle.create_dataset("data", data=signal[np.newaxis, :])
            handle.attrs["fs"] = sample_rate
            handle.attrs["source"] = "bidmc"
            handle.attrs["patient_id"] = f"bidmc{index:02d}"


def test_generate_ppg_demo_clips_are_quality_gated(tmp_path) -> None:
    """Selected PPG clips are continuous, resampled, and physiologically plausible."""
    _write_bidmc_fixture(tmp_path)

    clips = generate_ppg_demo_clips(tmp_path, num_clips=2, duration_seconds=10, sample_rate=64, seed=7)

    assert len(clips) == 2
    for clip in clips:
        assert clip.signal.shape == (640,)
        assert clip.signal.dtype == np.float32
        assert 45.0 <= clip.quality.heart_rate_bpm <= 120.0
        assert clip.quality.heart_rate_qos >= 0.8
        assert clip.quality.pulse_snr_db >= 8.0
        assert clip.quality.rr_cv <= 0.2
        assert clip.source_sample_rate == 125
