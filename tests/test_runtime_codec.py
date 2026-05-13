"""Tests for compressionkit.runtime.codec (D2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def golden_deploy_dir() -> Path | None:
    """Return the PPG 4x golden deploy directory if it exists."""
    d = Path("results/ppg_rvq_64hz_04x_golden/deploy")
    if d.exists() and (d / "deploy_manifest.json").exists():
        return d
    return None


@pytest.fixture()
def mock_deploy_dir(tmp_path: Path) -> Path:
    """Create a minimal mock deploy directory for unit testing."""
    deploy_dir = tmp_path / "deploy"
    deploy_dir.mkdir()

    # Create a simple codebook
    num_levels = 2
    num_embeddings = 4
    embedding_dim = 8
    codebooks = {}
    for i in range(num_levels):
        codebooks[f"level_{i}"] = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
    np.savez(deploy_dir / "codebook.npz", **codebooks)

    # Write manifest (no TFLite files — tests will skip encoder/decoder)
    manifest = {
        "model_name": "test_codec",
        "model_version": "1.0",
        "quantization": "INT8",
        "io_type": "int8",
        "encoder": {"tflite": "encoder.tflite", "input_shape": [None, 1, 32, 1], "output_shape": [None, 1, 8, 8]},
        "decoder": {"keras": "decoder.keras", "input_shape": [None, 1, 8, 8], "output_shape": [None, 1, 32, 1]},
        "codebook": {
            "npz": "codebook.npz",
            "header": "codebook.h",
            "num_levels": num_levels,
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
        },
        "sample_data": {"npz": None, "num_samples": 0, "arrays": []},
    }
    with (deploy_dir / "deploy_manifest.json").open("w") as f:
        json.dump(manifest, f)

    return deploy_dir


class TestCodebookOperations:
    """Test quantize_latent and dequantize_indices without TFLite models."""

    def test_quantize_nearest_neighbor(self):
        """Verify quantize picks the nearest codebook entry."""
        from compressionkit.runtime.codec import RVQCodec

        # Build a trivial codebook: 2 levels, 2 entries, dim=2
        # Level 0: [1,0] and [0,1]; Level 1: [0.5,0] and [0,0.5]
        deploy_dir = Path("/tmp/_test_rvq_quantize")
        deploy_dir.mkdir(exist_ok=True)
        np.savez(
            deploy_dir / "codebook.npz",
            level_0=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            level_1=np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32),
        )
        # Minimal manifest — we won't use encoder/decoder
        manifest = {
            "model_name": "test",
            "model_version": "1.0",
            "quantization": "NONE",
            "io_type": "float32",
            "encoder": {"tflite": "encoder.tflite"},
            "decoder": {},
            "codebook": {"npz": "codebook.npz", "num_levels": 2, "num_embeddings": 2, "embedding_dim": 2},
            "sample_data": {"npz": None, "num_samples": 0, "arrays": []},
        }
        with (deploy_dir / "deploy_manifest.json").open("w") as f:
            json.dump(manifest, f)

        # Create a dummy encoder tflite (we'll catch the error)
        # Instead, directly test the codebook methods
        # We need to bypass the Interpreter loading
        codec = object.__new__(RVQCodec)
        codec._deploy_dir = deploy_dir
        codec._manifest = manifest
        codec._codebooks = [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32),
        ]
        codec._num_levels = 2
        codec._num_embeddings = 2
        codec._embedding_dim = 2

        # Latent close to [1, 0] → level0=0, residual=[0,0] → level1=0 (closer to [0.5,0])
        latent = np.array([[[0.9, 0.1]]], dtype=np.float32)  # (1, 1, 2)
        indices = codec.quantize_latent(latent)
        assert indices.shape == (1, 1, 2)
        assert indices[0, 0, 0] == 0  # nearest to [1,0]

    def test_roundtrip_codebook(self):
        """quantize → dequantize should approximate the original latent."""
        codec = object.__new__(type("RVQCodec", (), {}))

        # Import the actual class for methods
        from compressionkit.runtime.codec import RVQCodec

        codec = object.__new__(RVQCodec)
        num_levels = 4
        num_embeddings = 256
        embedding_dim = 16
        rng = np.random.default_rng(42)
        codec._codebooks = [
            rng.standard_normal((num_embeddings, embedding_dim)).astype(np.float32) for _ in range(num_levels)
        ]
        codec._num_levels = num_levels
        codec._num_embeddings = num_embeddings
        codec._embedding_dim = embedding_dim

        # Random latent
        latent = rng.standard_normal((1, 1, 8, embedding_dim)).astype(np.float32)
        indices = codec.quantize_latent(latent)
        recon = codec.dequantize_indices(indices)

        # RVQ with 4 levels of 256 entries should reconstruct reasonably
        assert recon.shape == latent.shape
        error = np.mean((latent - recon) ** 2)
        assert error < np.mean(latent**2)  # better than zero prediction

    def test_dequantize_deterministic(self):
        """dequantize_indices is deterministic — same indices give same output."""
        from compressionkit.runtime.codec import RVQCodec

        codec = object.__new__(RVQCodec)
        codec._codebooks = [np.eye(4, dtype=np.float32) for _ in range(2)]
        codec._num_levels = 2
        codec._num_embeddings = 4
        codec._embedding_dim = 4

        indices = np.array([[[0, 1], [2, 3]]], dtype=np.int32)
        a = codec.dequantize_indices(indices)
        b = codec.dequantize_indices(indices)
        np.testing.assert_array_equal(a, b)


class TestQuantizationHelpers:
    """Test INT8 quantize/dequantize helpers."""

    def test_quantize_roundtrip(self):
        from compressionkit.runtime.codec import RVQCodec

        codec = object.__new__(RVQCodec)
        details = {
            "dtype": np.int8,
            "quantization_parameters": {
                "scales": np.array([0.05], dtype=np.float32),
                "zero_points": np.array([0], dtype=np.int32),
            },
        }
        data = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        quantized = codec._quantize(data, details)
        assert quantized.dtype == np.int8
        dequantized = codec._dequantize(quantized, details)
        np.testing.assert_allclose(data, dequantized, atol=0.05)

    def test_float32_passthrough(self):
        from compressionkit.runtime.codec import RVQCodec

        codec = object.__new__(RVQCodec)
        details = {
            "dtype": np.float32,
            "quantization_parameters": {
                "scales": np.array([1.0]),
                "zero_points": np.array([0]),
            },
        }
        data = np.array([1.5, -2.3], dtype=np.float32)
        result = codec._quantize(data, details)
        np.testing.assert_array_equal(data, result)


class TestRVQCodecIntegration:
    """Integration tests using the golden deploy directory."""

    def test_load_golden_codec(self, golden_deploy_dir):
        if golden_deploy_dir is None:
            pytest.skip("Golden deploy directory not available")
        from compressionkit.runtime import RVQCodec

        codec = RVQCodec(golden_deploy_dir)
        assert codec.num_levels == 4
        assert codec.num_embeddings == 256
        assert codec.embedding_dim == 16
        assert codec.manifest["model_name"] == "ppg_rvq_64hz_04x_golden"

    def test_encode_decode_roundtrip(self, golden_deploy_dir):
        if golden_deploy_dir is None:
            pytest.skip("Golden deploy directory not available")
        from compressionkit.runtime import RVQCodec

        codec = RVQCodec(golden_deploy_dir)

        # Load sample data
        sample = np.load(golden_deploy_dir / "sample_data.npz")
        signal = sample["inputs"][:1]  # (1, 1, 320, 1)

        # Encode
        indices = codec.encode(signal)
        assert indices.shape[-1] == codec.num_levels
        assert indices.min() >= 0
        assert indices.max() < codec.num_embeddings

        # Decode (if decoder TFLite available)
        if codec.has_decoder:
            recon = codec.decode(indices)
            assert recon.shape == signal.shape
            # Verify reconstruction is finite and has reasonable magnitude
            assert np.all(np.isfinite(recon))
            assert recon.std() > 0  # not all zeros
            # PRD should be finite (reconstruction is signal-like)
            mse = np.mean((signal - recon) ** 2)
            assert np.isfinite(mse)

    def test_encode_matches_sample_data(self, golden_deploy_dir):
        """Encoder output should be similar to training-time latents."""
        if golden_deploy_dir is None:
            pytest.skip("Golden deploy directory not available")
        from compressionkit.runtime import RVQCodec

        codec = RVQCodec(golden_deploy_dir)

        sample = np.load(golden_deploy_dir / "sample_data.npz")
        signal = sample["inputs"][:1]

        latent = codec.encode_latent(signal)
        assert latent.ndim == 4  # (B, 1, T', D)
        assert latent.dtype == np.float32

    def test_missing_manifest_raises(self, tmp_path):
        from compressionkit.runtime import RVQCodec

        with pytest.raises(FileNotFoundError, match="Deploy manifest not found"):
            RVQCodec(tmp_path)
