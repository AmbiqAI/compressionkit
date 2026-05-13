"""Tests for two-stage runtime: EntropyPrior + TwoStageCodec (D4)."""

from __future__ import annotations

import struct
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestArithmeticCoding:
    """Test arithmetic encode/decode roundtrip with known distributions."""

    def test_uniform_roundtrip(self):
        from compressionkit.runtime.two_stage import _arithmetic_decode_with_prior, _arithmetic_encode

        rng = np.random.default_rng(42)
        vocab = 4
        num_tokens = 20
        tokens = rng.integers(0, vocab, size=num_tokens).astype(np.int32)

        # Uniform probabilities
        probs = np.full((num_tokens, vocab), 1.0 / vocab, dtype=np.float32)

        bitstream = _arithmetic_encode(tokens, probs, vocab)
        assert len(bitstream) > 4  # at least header

        # For decode, create a mock prior that always returns uniform
        class MockPrior:
            vocab_size = vocab
            context_length = num_tokens

            def predict_logits(self, tokens_in):
                seq_len = tokens_in.shape[1]
                return np.zeros((1, seq_len, vocab), dtype=np.float32)

        decoded = _arithmetic_decode_with_prior(
            bitstream=bitstream,
            num_tokens=num_tokens,
            indices_shape=(num_tokens,),
            prior=MockPrior(),
            vocab_size=vocab,
        )
        np.testing.assert_array_equal(tokens, decoded)

    def test_skewed_roundtrip(self):
        """Test with non-uniform (skewed) probabilities."""
        from compressionkit.runtime.two_stage import _arithmetic_decode_with_prior, _arithmetic_encode

        vocab = 4
        num_tokens = 30
        # Tokens heavily biased toward 0
        rng = np.random.default_rng(123)
        tokens = rng.choice(vocab, size=num_tokens, p=[0.7, 0.1, 0.1, 0.1]).astype(np.int32)

        # Skewed probabilities — but position 0 must be uniform to match decoder
        skewed = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32)
        probs = np.tile(skewed, (num_tokens, 1)).astype(np.float32)
        probs[0] = 1.0 / vocab  # position 0 = uniform (matches decoder)

        bitstream = _arithmetic_encode(tokens, probs, vocab)

        class MockPrior:
            vocab_size = vocab
            context_length = num_tokens

            def predict_logits(self, tokens_in):
                seq_len = tokens_in.shape[1]
                # Return logits that produce the same skewed distribution
                logits = np.log(np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32))
                return np.tile(logits, (1, seq_len, 1))

        decoded = _arithmetic_decode_with_prior(
            bitstream=bitstream,
            num_tokens=num_tokens,
            indices_shape=(num_tokens,),
            prior=MockPrior(),
            vocab_size=vocab,
        )
        np.testing.assert_array_equal(tokens, decoded)

        # Skewed coding should use fewer bits than uniform
        uniform_bits = num_tokens * 2  # log2(4) = 2 bits/token
        actual_bits = (len(bitstream) - 4) * 8  # subtract header
        assert actual_bits < uniform_bits

    def test_header_stores_count(self):
        from compressionkit.runtime.two_stage import _arithmetic_encode

        tokens = np.array([0, 1, 2], dtype=np.int32)
        probs = np.full((3, 4), 0.25, dtype=np.float32)
        bs = _arithmetic_encode(tokens, probs, 4)
        count = struct.unpack(">I", bs[:4])[0]
        assert count == 3


class TestEntropyPrior:
    """Test EntropyPrior logic without building TFLite models."""

    def test_flatten_indices_3d(self):
        from compressionkit.runtime.prior import EntropyPrior

        prior = object.__new__(EntropyPrior)
        indices = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.int32)
        flat = prior._flatten_indices(indices)
        assert flat.shape == (1, 6)
        np.testing.assert_array_equal(flat[0], [1, 2, 3, 4, 5, 6])

    def test_flatten_indices_4d(self):
        from compressionkit.runtime.prior import EntropyPrior

        prior = object.__new__(EntropyPrior)
        indices = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
        flat = prior._flatten_indices(indices)
        assert flat.shape == (1, 4)

    def test_missing_prior_raises(self, tmp_path):
        from compressionkit.runtime.prior import EntropyPrior

        with pytest.raises(FileNotFoundError):
            EntropyPrior(tmp_path / "nonexistent.tflite")

    def test_predict_log_probs_uniform_baseline(self):
        """Position 0 should use uniform prior (log(1/K))."""
        from compressionkit.runtime.prior import EntropyPrior

        vocab = 8
        prior = object.__new__(EntropyPrior)
        prior._vocab_size = vocab
        prior._context_length = 10

        # Mock predict_logits to return zeros (uniform after softmax)
        prior.predict_logits = lambda tokens: np.zeros((tokens.shape[0], tokens.shape[1], vocab), dtype=np.float32)

        indices = np.zeros((1, 5, 2), dtype=np.int32)
        log_probs = prior.predict_log_probs(indices)
        # Position 0 should be -log(vocab)
        expected_lp0 = -np.log(vocab)
        np.testing.assert_allclose(log_probs[0, 0], expected_lp0, rtol=1e-5)

    def test_bits_per_token_uniform(self):
        """Uniform prior should give log2(K) bits per token."""
        from compressionkit.runtime.prior import EntropyPrior

        vocab = 16
        prior = object.__new__(EntropyPrior)
        prior._vocab_size = vocab
        prior._context_length = 20

        prior.predict_logits = lambda tokens: np.zeros((tokens.shape[0], tokens.shape[1], vocab), dtype=np.float32)

        indices = np.zeros((1, 5, 2), dtype=np.int32)
        bpt = prior.bits_per_token(indices)
        np.testing.assert_allclose(bpt, np.log2(vocab), rtol=1e-5)


class TestTwoStageCodec:
    """Test TwoStageCodec with mock codec and prior."""

    @pytest.fixture()
    def two_stage(self):
        from compressionkit.runtime.two_stage import TwoStageCodec

        vocab = 16
        num_levels = 2

        # Mock codec
        codec = MagicMock()
        codec.num_embeddings = vocab
        codec.num_levels = num_levels
        codec.embedding_dim = 8
        codec.encode = MagicMock(
            side_effect=lambda sig: (
                np.random.default_rng(0).integers(0, vocab, size=(1, 1, 8, num_levels)).astype(np.int32)
            )
        )
        codec.decode = MagicMock(side_effect=lambda idx: np.zeros((1, 1, 32, 1), dtype=np.float32))

        # Mock prior using a uniform-ish model
        prior = MagicMock()
        prior.vocab_size = vocab
        prior.context_length = 32
        prior._flatten_indices = lambda indices: (
            indices[:, 0].reshape(indices.shape[0], -1).astype(np.int32)
            if indices.ndim == 4
            else indices.reshape(indices.shape[0], -1).astype(np.int32)
        )
        prior.predict_logits = lambda tokens: np.zeros((tokens.shape[0], tokens.shape[1], vocab), dtype=np.float32)
        prior.bits_per_token = lambda indices: float(np.log2(vocab))

        return TwoStageCodec(codec, prior)

    def test_compress_indices_roundtrip(self, two_stage):
        rng = np.random.default_rng(42)
        indices = rng.integers(0, 16, size=(1, 1, 8, 2)).astype(np.int32)

        result = two_stage.compress_indices(indices)
        assert result.num_tokens == 16
        assert len(result.bitstream) > 0
        assert result.bits_per_token_actual > 0
        assert result.bits_per_token_uniform == 4.0

        decoded = two_stage.decompress_indices(result)
        np.testing.assert_array_equal(indices, decoded)

    def test_estimate_bitrate(self, two_stage):
        indices = np.random.randint(0, 16, size=(1, 1, 8, 2)).astype(np.int32)
        metrics = two_stage.estimate_bitrate(indices)
        assert "bits_per_token_prior" in metrics
        assert "cr_uplift" in metrics
        assert metrics["bits_per_token_uniform"] == 4.0

    def test_compression_result_properties(self):
        from compressionkit.runtime.two_stage import CompressionResult

        result = CompressionResult(
            bitstream=b"\x00" * 10,
            num_tokens=20,
            indices_shape=(1, 10, 2),
            bits_per_token_prior=3.5,
            bits_per_token_actual=3.8,
            bits_per_token_uniform=8.0,
        )
        assert result.cr_uniform == 8.0 / 3.8
        assert result.cr_uplift == 8.0 / 3.8
