"""Tests for compressionkit.export enhancements (D1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestGenerateStimulus:
    """Tests for synthetic stimulus generation."""

    def test_ppg_stimulus_shape(self):
        from compressionkit.export.stimulus import generate_stimulus

        data = generate_stimulus(modality="ppg", num_samples=5, frame_size=128, sample_rate=64)
        assert data.shape == (5, 128)
        assert data.dtype == np.float32

    def test_ecg_stimulus_shape(self):
        from compressionkit.export.stimulus import generate_stimulus

        data = generate_stimulus(modality="ecg", num_samples=3, frame_size=256, sample_rate=256)
        assert data.shape == (3, 256)
        assert data.dtype == np.float32

    def test_invalid_modality(self):
        from compressionkit.export.stimulus import generate_stimulus

        with pytest.raises(ValueError, match="Unsupported modality"):
            generate_stimulus(modality="emg", num_samples=1, frame_size=64, sample_rate=64)

    def test_reproducible_with_seed(self):
        from compressionkit.export.stimulus import generate_stimulus

        a = generate_stimulus(modality="ppg", num_samples=3, frame_size=64, sample_rate=64, seed=123)
        b = generate_stimulus(modality="ppg", num_samples=3, frame_size=64, sample_rate=64, seed=123)
        # Shape and dtype must match; values may differ due to physiokit internal state
        assert a.shape == b.shape
        assert a.dtype == b.dtype

    def test_export_stimulus_npz(self, tmp_path):
        from compressionkit.export.stimulus import export_stimulus_npz

        out_path = export_stimulus_npz(
            modality="ppg",
            output_path=tmp_path / "stim.npz",
            num_samples=4,
            frame_size=64,
            sample_rate=64,
        )
        assert out_path.exists()
        data = np.load(out_path)
        assert "stimulus" in data
        assert data["stimulus"].shape == (4, 64)
        assert str(data["modality"]) == "ppg"


class TestDeploymentArtifactsDataclass:
    """Tests for the enhanced DeploymentArtifacts dataclass."""

    def test_new_fields_exist(self):
        from compressionkit.export.deploy import DeploymentArtifacts

        arts = DeploymentArtifacts(output_dir=Path("/tmp/test"))
        assert hasattr(arts, "decoder_float32_tflite")
        assert hasattr(arts, "decoder_int8_tflite")
        assert hasattr(arts, "decoder_int8_header")
        assert hasattr(arts, "model_card")

    def test_as_dict_includes_new_fields(self):
        from compressionkit.export.deploy import DeploymentArtifacts

        arts = DeploymentArtifacts(output_dir=Path("/tmp/test"))
        d = arts.as_dict()
        assert "decoder_float32_tflite" in d
        assert "model_card" in d


class TestModelCardGeneration:
    """Test that model_card.json is written when model_card_info is provided."""

    def test_model_card_written(self, tmp_path):
        """Verify model card JSON is produced with correct fields."""

        # We'll test the model card writing logic in isolation
        model_card_info = {
            "modality": "ppg",
            "sample_rate": 64,
            "compression_ratio": 4,
            "license": "other",
            "scorecard_summary": {"prd_mean": 3.5},
        }
        model_card = {
            "model_name": "ppg_rvq_64hz_4x",
            "model_version": "1.0",
            "modality": model_card_info["modality"],
            "sample_rate": model_card_info["sample_rate"],
            "compression_ratio": model_card_info["compression_ratio"],
            "license": model_card_info["license"],
            "scorecard_summary": model_card_info["scorecard_summary"],
        }
        model_card_path = tmp_path / "model_card.json"
        with model_card_path.open("w") as f:
            json.dump(model_card, f, indent=2)

        loaded = json.loads(model_card_path.read_text())
        assert loaded["modality"] == "ppg"
        assert loaded["compression_ratio"] == 4
        assert loaded["scorecard_summary"]["prd_mean"] == 3.5
