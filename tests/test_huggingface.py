"""Tests for HuggingFace release pipeline (D3)."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

# ── Model card generation ──────────────────────────────────────────


@pytest.fixture()
def mock_deploy(tmp_path: Path) -> Path:
    """Create a minimal deploy directory for model card tests."""
    deploy = tmp_path / "deploy"
    deploy.mkdir()

    manifest = {
        "model_name": "ppg_rvq_64hz_04x_test",
        "model_version": "1.0",
        "quantization": "INT8",
        "io_type": "int8",
        "encoder": {
            "tflite": "encoder.tflite",
            "header": "encoder.h",
            "input_shape": [None, 1, 320, 1],
            "output_shape": [None, 1, 80, 16],
        },
        "decoder": {"keras": "decoder.keras"},
        "codebook": {
            "npz": "codebook.npz",
            "header": "codebook.h",
            "num_levels": 4,
            "num_embeddings": 256,
            "embedding_dim": 16,
        },
        "sample_data": {"npz": "sample_data.npz", "num_samples": 10, "arrays": ["inputs"]},
    }
    (deploy / "deploy_manifest.json").write_text(json.dumps(manifest))
    return deploy


@pytest.fixture()
def mock_scorecard(tmp_path: Path) -> Path:
    """Create a minimal scorecard JSON."""
    scorecard = {
        "modality": "ppg",
        "sample_rate": 64,
        "time_domain": {
            "prd_percent": {
                "n": 100,
                "mean": 5.0,
                "std": 2.0,
                "median": 3.0,
                "p10": 1.5,
                "p90": 10.0,
                "max": 50.0,
                "min": 0.5,
            },
            "rmse": {
                "n": 100,
                "mean": 0.03,
                "std": 0.02,
                "median": 0.025,
                "p10": 0.015,
                "p90": 0.05,
                "max": 0.3,
                "min": 0.01,
            },
            "cosine_similarity": {
                "n": 100,
                "mean": 0.99,
                "std": 0.01,
                "median": 0.995,
                "p10": 0.98,
                "p90": 0.999,
                "max": 0.9999,
                "min": 0.9,
            },
        },
        "spectral": {
            "band_total_rel_error": {
                "n": 100,
                "mean": 0.08,
                "std": 0.1,
                "median": 0.035,
                "p10": 0.01,
                "p90": 0.2,
                "max": 3.0,
                "min": 0.001,
            },
        },
        "bitrate": {
            "cr_codec_uniform": 4.0,
            "cr_codec_learned": 4.5,
        },
    }
    path = tmp_path / "quality_scorecard.json"
    path.write_text(json.dumps(scorecard))
    return path


class TestModelCardGeneration:
    """Test generate_model_card from compressionkit.export.model_card."""

    def test_basic_card(self, mock_deploy):
        from compressionkit.export.model_card import generate_model_card

        card = generate_model_card(mock_deploy)
        assert card.startswith("---\n")
        assert "license: other" in card
        assert "license_name: ambiq-model-weights-license" in card
        assert "compressionkit" in card
        assert "ppg" in card.lower()
        assert "RVQ" in card or "rvq" in card.lower()
        assert "from_pretrained" in card

    def test_card_with_scorecard(self, mock_deploy, mock_scorecard):
        from compressionkit.export.model_card import generate_model_card

        card = generate_model_card(mock_deploy, scorecard_path=mock_scorecard)
        assert "Quality Metrics" in card
        assert "PRD vs input — faithfulness (%)" in card
        assert "Cosine Similarity" in card
        assert "4.0x" in card  # CR uniform
        assert "4.50x" in card  # CR learned

    def test_card_yaml_frontmatter(self, mock_deploy):
        from compressionkit.export.model_card import generate_model_card

        card = generate_model_card(mock_deploy)
        # Should have valid YAML frontmatter
        parts = card.split("---")
        assert len(parts) >= 3
        frontmatter = parts[1]
        assert "license:" in frontmatter
        assert "library_name: compressionkit" in frontmatter
        assert "tags:" in frontmatter

    def test_card_compression_ratio_inferred(self, mock_deploy):
        from compressionkit.export.model_card import generate_model_card

        card = generate_model_card(mock_deploy)
        # 320 / 80 = 4x
        assert "4x" in card

    def test_card_custom_license(self, mock_deploy):
        from compressionkit.export.model_card import generate_model_card

        card = generate_model_card(mock_deploy, license_id="mit")
        assert "license: mit" in card


# ── Publish staging ────────────────────────────────────────────────


class TestPublishStaging:
    """Test publish() in dry-run mode (no HF upload)."""

    def test_dry_run_stages_files(self, mock_deploy, mock_scorecard):
        # Create some dummy files in the deploy dir
        (mock_deploy / "encoder.tflite").write_bytes(b"\x00" * 100)
        (mock_deploy / "codebook.npz").write_bytes(b"\x00" * 50)
        (mock_deploy / "encoder.h").write_text("// header")

        from scripts.publish_to_huggingface import publish

        staging_dir = publish(
            deploy_dir=mock_deploy,
            repo_id="test-org/test-model",
            scorecard_path=mock_scorecard,
            dry_run=True,
        )
        assert staging_dir is not None
        assert staging_dir.is_dir()

        # Check renamed files
        assert (staging_dir / "encoder_int8.tflite").exists()
        assert (staging_dir / "codebook.npz").exists()
        assert (staging_dir / "encoder.h").exists()
        assert (staging_dir / "config.json").exists()  # renamed from deploy_manifest.json
        assert (staging_dir / "quality_scorecard.json").exists()
        assert (staging_dir / "README.md").exists()

        # README should be a proper model card
        readme = (staging_dir / "README.md").read_text()
        assert readme.startswith("---\n")
        assert "compressionkit" in readme

        # Cleanup
        shutil.rmtree(staging_dir)

    def test_dry_run_no_manifest_exits(self, tmp_path):
        from scripts.publish_to_huggingface import publish

        with pytest.raises(FileNotFoundError):
            publish(deploy_dir=tmp_path, repo_id="test/test", dry_run=True)

    def test_dry_run_hybrid_stages_denoiser(self, tmp_path):
        """A hybrid deploy (DSP backend + learned denoiser) must publish the denoiser.

        The backend reports ``family='spiht'`` in its manifest; the presence of
        ``hybrid_manifest.json`` is what marks the package as hybrid. The learned
        ``denoiser_gain_model.keras`` would be silently dropped if it were staged
        with the plain SPIHT file map.
        """
        deploy = tmp_path / "deploy"
        deploy.mkdir()
        manifest = {
            "family": "spiht",
            "method": "dsp",
            "compression_ratio": 16.0,
            "codec": {"modality": "ecg", "sample_rate": 256, "frame_size": 512, "target_cr": 16.0},
        }
        (deploy / "deploy_manifest.json").write_text(json.dumps(manifest))
        (deploy / "hybrid_manifest.json").write_text(json.dumps({"pipeline": ["denoise", "spiht"]}))
        (deploy / "spiht_config.json").write_text("{}")
        (deploy / "denoiser_gain_model.keras").write_bytes(b"\x00" * 64)
        (deploy / "denoiser_train_config.json").write_text("{}")
        (deploy / "spiht_app_config.h").write_text("// header")
        (deploy / "checksums.json").write_text("{}")

        from scripts.publish_to_huggingface import publish

        staging_dir = publish(deploy_dir=deploy, repo_id="test-org/ecg-hybrid-16x", dry_run=True)
        assert staging_dir is not None
        try:
            # The AI denoiser and hybrid pipeline manifest must be staged.
            assert (staging_dir / "denoiser_gain_model.keras").exists()
            assert (staging_dir / "denoiser_train_config.json").exists()
            assert (staging_dir / "hybrid_manifest.json").exists()
            # Backend (SPIHT) contract is still present.
            assert (staging_dir / "config.json").exists()  # renamed deploy_manifest.json
            assert (staging_dir / "spiht_app_config.h").exists()
            assert (staging_dir / "README.md").exists()
        finally:
            shutil.rmtree(staging_dir)


# ── from_pretrained symlink logic ──────────────────────────────────


class TestFromPretrainedSymlinks:
    """Test the _ensure_symlink helper used by from_pretrained."""

    def test_ensure_symlink_creates(self, tmp_path):
        from compressionkit.runtime.codec import _ensure_symlink

        (tmp_path / "encoder_int8.tflite").write_bytes(b"\x00")
        _ensure_symlink(tmp_path, "encoder_int8.tflite", "encoder.tflite")
        assert (tmp_path / "encoder.tflite").is_symlink()
        assert (tmp_path / "encoder.tflite").exists()

    def test_ensure_symlink_noop_when_target_exists(self, tmp_path):
        from compressionkit.runtime.codec import _ensure_symlink

        (tmp_path / "encoder_int8.tflite").write_bytes(b"\x01")
        (tmp_path / "encoder.tflite").write_bytes(b"\x02")
        _ensure_symlink(tmp_path, "encoder_int8.tflite", "encoder.tflite")
        # Should NOT have replaced the existing file
        assert not (tmp_path / "encoder.tflite").is_symlink()
        assert (tmp_path / "encoder.tflite").read_bytes() == b"\x02"

    def test_ensure_symlink_noop_when_hf_missing(self, tmp_path):
        from compressionkit.runtime.codec import _ensure_symlink

        _ensure_symlink(tmp_path, "nonexistent.tflite", "encoder.tflite")
        assert not (tmp_path / "encoder.tflite").exists()


class TestFamilyAgnosticLoader:
    """Test HF compatibility aliases on the family-agnostic load_codec path."""

    def test_load_codec_creates_rvq_hf_aliases(self, tmp_path, monkeypatch):
        (tmp_path / "config.json").write_text(json.dumps({"family": "rvq", "spec": "codec_spec.json"}))
        (tmp_path / "codec_spec.json").write_text(
            json.dumps(
                {
                    "family": "rvq",
                    "encoder": {"tflite": "encoder.tflite"},
                    "decoder": {"int8_tflite": "decoder.tflite"},
                    "codebook": {"npz": "codebook.npz"},
                }
            )
        )
        (tmp_path / "encoder_int8.tflite").write_bytes(b"encoder")
        (tmp_path / "decoder_int8.tflite").write_bytes(b"decoder")
        (tmp_path / "sample_stimulus.npz").write_bytes(b"samples")
        (tmp_path / "codebook.npz").write_bytes(b"codebook")

        class FakeRVQCodec:
            def __init__(self, deploy_dir):
                deploy_path = Path(deploy_dir)
                assert (deploy_path / "deploy_manifest.json").exists()
                assert (deploy_path / "encoder.tflite").exists()
                assert (deploy_path / "decoder.tflite").exists()
                assert (deploy_path / "sample_data.npz").exists()

        fake_module = types.ModuleType("compressionkit.runtime.codec")
        fake_module.RVQCodec = FakeRVQCodec
        monkeypatch.setitem(sys.modules, "compressionkit.runtime.codec", fake_module)

        from compressionkit.runtime import load_codec

        assert isinstance(load_codec(tmp_path), FakeRVQCodec)


# ── Golden deploy model card (integration) ────────────────────────


class TestGoldenModelCard:
    """Integration test using golden deploy directory."""

    @pytest.fixture()
    def golden_deploy(self) -> Path | None:
        d = Path("results/ppg_rvq_64hz_04x_golden/deploy")
        if d.exists() and (d / "deploy_manifest.json").exists():
            return d
        return None

    @pytest.fixture()
    def golden_scorecard(self) -> Path | None:
        p = Path("results/ppg_rvq_64hz_04x_golden/quality_scorecard.json")
        return p if p.exists() else None

    def test_golden_model_card(self, golden_deploy, golden_scorecard):
        if golden_deploy is None:
            pytest.skip("Golden deploy directory not available")
        from compressionkit.export.model_card import generate_model_card

        card = generate_model_card(golden_deploy, scorecard_path=golden_scorecard)
        assert "compressionkit-ppg-4x" in card
        assert "PPG" in card
        assert "4x" in card
        if golden_scorecard:
            assert "Quality Metrics" in card
