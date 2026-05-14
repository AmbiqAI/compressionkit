"""Tests for cross-channel (multi-lead) entropy prior architectures (issue #9)."""

from __future__ import annotations

import numpy as np
import pytest

from compressionkit.generative.xlead_prior import (
    build_xlead_concat_prior,
    build_xlead_interleave_prior,
    deinterleave_leads,
    interleave_leads,
)


class TestXleadConcatPrior:
    def test_output_shape(self):
        model = build_xlead_concat_prior(
            vocab_size=64,
            context_length=16,
            num_leads=3,
            embed_dim=16,
            num_layers=2,
            kernel_size=3,
        )
        tokens = np.random.default_rng(0).integers(0, 64, size=(2, 16, 3)).astype(np.int32)
        out = model(tokens, training=False)
        assert out.shape == (2, 16, 3, 64)

    def test_causal_masking(self):
        """Verify predictions at t=0 don't depend on t=1 inputs."""
        model = build_xlead_concat_prior(
            vocab_size=32,
            context_length=8,
            num_leads=2,
            embed_dim=8,
            num_layers=2,
            kernel_size=3,
        )
        rng = np.random.default_rng(1)
        tokens_a = rng.integers(0, 32, size=(1, 8, 2)).astype(np.int32)
        tokens_b = tokens_a.copy()
        tokens_b[0, 1:, :] = rng.integers(0, 32, size=(7, 2))
        out_a = np.asarray(model(tokens_a, training=False))
        out_b = np.asarray(model(tokens_b, training=False))
        # Position 0 logits should be identical (only sees t=0)
        np.testing.assert_allclose(out_a[0, 0], out_b[0, 0], atol=1e-5)


class TestXleadInterleavePrior:
    def test_output_shape(self):
        model = build_xlead_interleave_prior(
            vocab_size=64,
            context_length=16,
            num_leads=3,
            embed_dim=16,
            num_layers=2,
            kernel_size=3,
        )
        seq_len = 16 * 3
        tokens = np.random.default_rng(2).integers(0, 64, size=(2, seq_len)).astype(np.int32)
        out = model(tokens, training=False)
        assert out.shape == (2, seq_len, 64)

    def test_causal_masking(self):
        model = build_xlead_interleave_prior(
            vocab_size=32,
            context_length=4,
            num_leads=2,
            embed_dim=8,
            num_layers=2,
            kernel_size=3,
        )
        seq_len = 4 * 2
        rng = np.random.default_rng(3)
        tokens_a = rng.integers(0, 32, size=(1, seq_len)).astype(np.int32)
        tokens_b = tokens_a.copy()
        tokens_b[0, 1:] = rng.integers(0, 32, size=(seq_len - 1,))
        out_a = np.asarray(model(tokens_a, training=False))
        out_b = np.asarray(model(tokens_b, training=False))
        np.testing.assert_allclose(out_a[0, 0], out_b[0, 0], atol=1e-5)


class TestInterleaveHelpers:
    def test_roundtrip(self):
        rng = np.random.default_rng(10)
        tokens = rng.integers(0, 256, size=(3, 20, 4, 1)).astype(np.int16)
        interleaved = interleave_leads(tokens)
        assert interleaved.shape == (3 * 20 * 4,)
        recovered = deinterleave_leads(interleaved, num_leads=3)
        expected = tokens[:, :, :, 0].reshape(3, 80)
        np.testing.assert_array_equal(recovered, expected)

    def test_2d_input(self):
        flat = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int16)  # (2 leads, 3 tokens)
        interleaved = interleave_leads(flat)
        # Expected interleave: t0_L0, t0_L1, t1_L0, t1_L1, t2_L0, t2_L1
        expected = np.array([10, 40, 20, 50, 30, 60], dtype=np.int16)
        np.testing.assert_array_equal(interleaved, expected)

    def test_deinterleave(self):
        interleaved = np.array([10, 40, 20, 50, 30, 60])
        result = deinterleave_leads(interleaved, num_leads=2)
        expected = np.array([[10, 20, 30], [40, 50, 60]])
        np.testing.assert_array_equal(result, expected)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="Expected 2-D or 4-D"):
            interleave_leads(np.zeros((3, 4, 5)))
