"""Tests for multi-lead token extraction and decoding (issue #8)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from compressionkit.generative.token_extraction import extract_rvq_tokens


def _make_mock_model(
    *, frame_size: int = 64, tokens_per_frame: int = 4, embed_dim: int = 16, vocab_size: int = 256, num_levels: int = 1
):
    """Build a mock VQAutoencoder with deterministic encode behaviour."""
    model = MagicMock()

    def fake_encoder(x, training=False):
        # x shape: (N, 1, frame_size, num_leads)
        batch = x.shape[0]
        return np.random.default_rng(42).standard_normal((batch, 1, tokens_per_frame, embed_dim)).astype(np.float32)

    model.encoder.side_effect = fake_encoder

    def fake_vq_encode(z):
        batch = z.shape[0]
        rng = np.random.default_rng(7)
        return [
            rng.integers(0, vocab_size, size=(batch * tokens_per_frame,)).astype(np.int32) for _ in range(num_levels)
        ]

    model.vq.encode = fake_vq_encode
    return model


class TestExtractPerLead:
    def test_single_lead_backward_compat(self):
        model = _make_mock_model(frame_size=64, tokens_per_frame=4)
        signals = [np.random.default_rng(0).standard_normal(256).astype(np.float32)]
        tokens = extract_rvq_tokens(model, signals, frame_size=64, per_lead=False)
        # 256 / 64 = 4 frames
        assert tokens.shape == (4, 4, 1)
        assert tokens.dtype == np.int16

    def test_per_lead_shape(self):
        model = _make_mock_model(frame_size=64, tokens_per_frame=4)
        # 3-lead signal of length 192 → 192/64 = 3 frames per lead
        signals = [np.random.default_rng(1).standard_normal((192, 3)).astype(np.float32)]
        tokens = extract_rvq_tokens(model, signals, frame_size=64, per_lead=True)
        assert tokens.shape == (3, 3, 4, 1)  # (leads, frames, tpf, levels)
        assert tokens.dtype == np.int16

    def test_per_lead_1d_signal_gives_single_lead(self):
        model = _make_mock_model(frame_size=64, tokens_per_frame=4)
        signals = [np.random.default_rng(2).standard_normal(128).astype(np.float32)]
        tokens = extract_rvq_tokens(model, signals, frame_size=64, per_lead=True)
        # 1-D → treated as 1 lead, 2 frames
        assert tokens.shape == (1, 2, 4, 1)

    def test_per_lead_multiple_signals(self):
        model = _make_mock_model(frame_size=64, tokens_per_frame=4)
        # Two signals with 2 leads each, 128 samples each → 2 frames each → 4 total per lead
        signals = [
            np.random.default_rng(3).standard_normal((128, 2)).astype(np.float32),
            np.random.default_rng(4).standard_normal((128, 2)).astype(np.float32),
        ]
        tokens = extract_rvq_tokens(model, signals, frame_size=64, per_lead=True)
        assert tokens.shape == (2, 4, 4, 1)  # 2 leads, 4 frames total, 4 tpf, 1 level

    def test_empty_signals(self):
        model = _make_mock_model(frame_size=64, tokens_per_frame=4)
        tokens = extract_rvq_tokens(model, [], frame_size=64, per_lead=True)
        assert tokens.shape[0] == 0


class TestDecodePerLead:
    def _make_decode_model(
        self, *, frame_size: int = 64, tokens_per_frame: int = 4, embed_dim: int = 16, vocab_size: int = 256
    ):
        model = MagicMock()

        def fake_vq_decode(indices_list, shape):
            return np.random.default_rng(11).standard_normal(shape).astype(np.float32)

        model.vq.decode = fake_vq_decode

        def fake_decoder(z, training=False):
            batch = z.shape[0]
            return np.random.default_rng(22).standard_normal((batch, 1, frame_size, 1)).astype(np.float32)

        model.decoder.side_effect = fake_decoder
        return model

    def test_decode_per_lead(self):
        from compressionkit.generative.sampling import decode_tokens_to_signal

        model = self._make_decode_model(frame_size=64, tokens_per_frame=4, embed_dim=16)
        # 2 leads, 3 samples, 8 tokens each (2 frames)
        tokens = np.random.default_rng(5).integers(0, 256, size=(2, 3, 8)).astype(np.int32)
        out = decode_tokens_to_signal(model, tokens, frame_size=64, tokens_per_frame=4, embedding_dim=16, per_lead=True)
        assert out.shape == (2, 3, 128)  # 2 leads, 3 samples, 2 frames × 64

    def test_decode_single_lead_unchanged(self):
        from compressionkit.generative.sampling import decode_tokens_to_signal

        model = self._make_decode_model(frame_size=64, tokens_per_frame=4, embed_dim=16)
        tokens = np.random.default_rng(6).integers(0, 256, size=(2, 8)).astype(np.int32)
        out = decode_tokens_to_signal(
            model, tokens, frame_size=64, tokens_per_frame=4, embedding_dim=16, per_lead=False
        )
        assert out.shape == (2, 128)

    def test_decode_per_lead_wrong_ndim_raises(self):
        from compressionkit.generative.sampling import decode_tokens_to_signal

        model = self._make_decode_model()
        tokens = np.zeros((8,), dtype=np.int32)
        with pytest.raises(ValueError, match="per_lead=True requires"):
            decode_tokens_to_signal(model, tokens, frame_size=64, tokens_per_frame=4, embedding_dim=16, per_lead=True)


class TestCompareStrategies:
    def test_compare_basic(self):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "compare_12lead_strategies",
            str(
                __import__("pathlib").Path(__file__).resolve().parent.parent
                / "scripts"
                / "compare_12lead_strategies.py"
            ),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["compare_12lead_strategies"] = mod
        spec.loader.exec_module(mod)

        rng = np.random.default_rng(99)
        tokens_ind = rng.integers(0, 64, size=(3, 100, 4, 1)).astype(np.int16)
        tokens_joint = rng.integers(0, 64, size=(100, 4, 1)).astype(np.int16)
        result = mod.compare_strategies(tokens_ind, tokens_joint, vocab_size=64)
        assert result["num_leads"] == 3
        assert len(result["per_lead_entropy_bits"]) == 3
        assert result["mean_entropy_bits"] > 0
        assert result["mean_cross_lead_mi_bits"] >= 0
        assert "joint_strategy" in result
        assert result["joint_strategy"]["token_entropy_bits"] > 0
