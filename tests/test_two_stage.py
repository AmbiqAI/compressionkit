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

            def predict_next_probs(self, tokens_in):
                return np.full((tokens_in.shape[0], vocab), 1.0 / vocab, dtype=np.float32)

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

            def predict_next_probs(self, tokens_in):
                return np.tile(np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32), (tokens_in.shape[0], 1))

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

    def test_peaked_large_vocab_does_not_hang(self):
        """Regression test: a confident real-model-style distribution over a
        large vocabulary must not produce a zero-width coder interval.

        Previously, ``_probs_to_cdf`` quantized to ``_WHOLE`` (2**24), the
        same constant used for the coder's own range. Since the coder's
        range can shrink to ``_QUARTER`` (2**22) after renormalization, a
        minimum-count-1 symbol could map to a zero-width interval
        (``rng * 1 // _WHOLE == 0``), causing ``low == high`` forever -- an
        infinite loop with unbounded memory growth. This only manifested
        with sharply peaked distributions (e.g. from a trained prior), not
        the small/uniform toy distributions used elsewhere in this file.
        """
        from compressionkit.runtime.two_stage import _arithmetic_decode_with_prior, _arithmetic_encode

        rng = np.random.default_rng(7)
        vocab = 256
        num_tokens = 300
        raw = np.array([0.6**k for k in range(vocab)])
        peaked = (raw / raw.sum()).astype(np.float32)
        tokens = rng.choice(vocab, size=num_tokens, p=peaked).astype(np.int32)
        probs = np.tile(peaked, (num_tokens, 1)).astype(np.float32)
        probs[0] = 1.0 / vocab  # position 0 = uniform (matches decoder)

        bitstream = _arithmetic_encode(tokens, probs, vocab)
        assert len(bitstream) > 4

        class MockPrior:
            vocab_size = vocab
            context_length = num_tokens

            def predict_next_probs(self, tokens_in):
                return np.tile(peaked, (tokens_in.shape[0], 1)).astype(np.float32)

        decoded = _arithmetic_decode_with_prior(
            bitstream=bitstream,
            num_tokens=num_tokens,
            indices_shape=(num_tokens,),
            prior=MockPrior(),
            vocab_size=vocab,
        )
        np.testing.assert_array_equal(tokens, decoded)


class TestRansCoding:
    """Test rANS encode/decode roundtrip, mirroring TestArithmeticCoding."""

    def test_uniform_roundtrip(self):
        from compressionkit.runtime.two_stage import _rans_decode_with_prior, _rans_encode

        rng = np.random.default_rng(42)
        vocab = 4
        num_tokens = 20
        tokens = rng.integers(0, vocab, size=num_tokens).astype(np.int32)
        probs = np.full((num_tokens, vocab), 1.0 / vocab, dtype=np.float32)

        bitstream = _rans_encode(tokens, probs, vocab)
        assert len(bitstream) > 4  # at least header

        class MockPrior:
            vocab_size = vocab
            context_length = num_tokens

            def predict_next_probs(self, tokens_in):
                return np.full((tokens_in.shape[0], vocab), 1.0 / vocab, dtype=np.float32)

        decoded = _rans_decode_with_prior(
            bitstream=bitstream,
            num_tokens=num_tokens,
            indices_shape=(num_tokens,),
            prior=MockPrior(),
            vocab_size=vocab,
        )
        np.testing.assert_array_equal(tokens, decoded)

    def test_skewed_roundtrip(self):
        """Test with non-uniform (skewed) probabilities."""
        from compressionkit.runtime.two_stage import _rans_decode_with_prior, _rans_encode

        vocab = 4
        num_tokens = 30
        rng = np.random.default_rng(123)
        tokens = rng.choice(vocab, size=num_tokens, p=[0.7, 0.1, 0.1, 0.1]).astype(np.int32)

        skewed = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32)
        probs = np.tile(skewed, (num_tokens, 1)).astype(np.float32)
        probs[0] = 1.0 / vocab

        bitstream = _rans_encode(tokens, probs, vocab)

        class MockPrior:
            vocab_size = vocab
            context_length = num_tokens

            def predict_next_probs(self, tokens_in):
                return np.tile(np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32), (tokens_in.shape[0], 1))

        decoded = _rans_decode_with_prior(
            bitstream=bitstream,
            num_tokens=num_tokens,
            indices_shape=(num_tokens,),
            prior=MockPrior(),
            vocab_size=vocab,
        )
        np.testing.assert_array_equal(tokens, decoded)

    def test_larger_alphabet_roundtrip(self):
        """Roundtrip with a realistic RVQ-sized vocabulary (K=256)."""
        from compressionkit.runtime.two_stage import _rans_decode_with_prior, _rans_encode

        rng = np.random.default_rng(7)
        vocab = 256
        num_tokens = 500
        raw = np.array([0.6**k for k in range(vocab)])
        p = (raw / raw.sum()).astype(np.float32)
        tokens = rng.choice(vocab, size=num_tokens, p=p).astype(np.int32)
        probs = np.tile(p, (num_tokens, 1)).astype(np.float32)
        probs[0] = 1.0 / vocab

        bitstream = _rans_encode(tokens, probs, vocab)

        class MockPrior:
            vocab_size = vocab
            context_length = num_tokens

            def predict_next_probs(self, tokens_in):
                return np.tile(p, (tokens_in.shape[0], 1)).astype(np.float32)

        decoded = _rans_decode_with_prior(
            bitstream=bitstream,
            num_tokens=num_tokens,
            indices_shape=(num_tokens,),
            prior=MockPrior(),
            vocab_size=vocab,
        )
        np.testing.assert_array_equal(tokens, decoded)

        # Should be close to the arithmetic coder's bit length (same
        # quantization order of magnitude), not wildly worse.
        from compressionkit.runtime.two_stage import _arithmetic_encode

        bs_ac = _arithmetic_encode(tokens, probs, vocab)
        rans_bpt = len(bitstream) * 8 / num_tokens
        ac_bpt = len(bs_ac) * 8 / num_tokens
        assert rans_bpt < ac_bpt * 1.05

    def test_header_stores_count(self):
        from compressionkit.runtime.two_stage import _rans_encode

        tokens = np.array([0, 1, 2], dtype=np.int32)
        probs = np.full((3, 4), 0.25, dtype=np.float32)
        bs = _rans_encode(tokens, probs, 4)
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

    def test_predict_log_probs_sliding_context(self):
        """Long streams should use a sliding context instead of failing."""
        from compressionkit.runtime.prior import EntropyPrior

        vocab = 8
        prior = object.__new__(EntropyPrior)
        prior._vocab_size = vocab
        prior._context_length = 3

        def predict_logits(tokens):
            assert tokens.shape[1] <= prior._context_length
            return np.zeros((tokens.shape[0], tokens.shape[1], vocab), dtype=np.float32)

        prior.predict_logits = predict_logits

        indices = np.zeros((1, 8, 1), dtype=np.int32)
        log_probs = prior.predict_log_probs(indices)

        assert log_probs.shape == (1, 8)
        np.testing.assert_allclose(log_probs, -np.log(vocab), rtol=1e-5)

    def test_bits_per_token_uniform(self):
        """Uniform prior should give log2(K) bits per token."""
        from compressionkit.runtime.prior import EntropyPrior

        vocab = 16
        prior = object.__new__(EntropyPrior)
        prior._vocab_size = vocab
        prior._context_length = 20

        prior.predict_logits = lambda tokens: np.zeros((tokens.shape[0], tokens.shape[1], vocab), dtype=np.float32)
        prior.predict_next_probs = lambda tokens: np.full((tokens.shape[0], vocab), 1.0 / vocab, dtype=np.float32)

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
        prior.predict_next_probs = lambda tokens: np.full((tokens.shape[0], vocab), 1.0 / vocab, dtype=np.float32)
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
        assert result.backend == "arithmetic"

        decoded = two_stage.decompress_indices(result)
        np.testing.assert_array_equal(indices, decoded)

    def test_compress_indices_roundtrip_rans_backend(self, two_stage):
        rng = np.random.default_rng(42)
        indices = rng.integers(0, 16, size=(1, 1, 8, 2)).astype(np.int32)

        result = two_stage.compress_indices(indices, backend="rans")
        assert result.num_tokens == 16
        assert len(result.bitstream) > 0
        assert result.backend == "rans"

        decoded = two_stage.decompress_indices(result)
        np.testing.assert_array_equal(indices, decoded)

    def test_unknown_backend_raises(self, two_stage):
        rng = np.random.default_rng(42)
        indices = rng.integers(0, 16, size=(1, 1, 8, 2)).astype(np.int32)
        with pytest.raises(ValueError):
            two_stage.compress_indices(indices, backend="bogus")

    def test_compress_indices_roundtrip_with_short_prior_context(self, two_stage):
        rng = np.random.default_rng(42)
        indices = rng.integers(0, 16, size=(1, 1, 8, 2)).astype(np.int32)
        two_stage.prior.context_length = 4

        def predict_next_probs(context_tokens):
            assert context_tokens.shape[1] <= 4
            return np.full((context_tokens.shape[0], 16), 1.0 / 16, dtype=np.float32)

        two_stage.prior._flatten_indices = lambda arr: arr.reshape(arr.shape[0], -1).astype(np.int32)
        two_stage.prior.predict_next_probs = predict_next_probs
        two_stage.prior.bits_per_token = lambda arr: float(np.log2(16))

        result = two_stage.compress_indices(indices)
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
