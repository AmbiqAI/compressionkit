"""Unit tests for the generative prior building blocks."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from compressionkit.generative import build_prior
from compressionkit.generative.sampling import (
    _softmax_np,
    _top_k_logits,
    sample_tokens,
)

VOCAB = 32
CTX = 16


def test_prior_forward_shape() -> None:
    model = build_prior(
        vocab_size=VOCAB,
        context_length=CTX,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
    )
    tokens = np.zeros((3, CTX), dtype=np.int32)
    logits = np.asarray(model(tokens, training=False))
    assert logits.shape == (3, CTX, VOCAB)


def test_prior_embed_dim_must_divide_num_heads() -> None:
    with pytest.raises(ValueError):
        build_prior(vocab_size=VOCAB, context_length=CTX, embed_dim=15, num_layers=1, num_heads=2, ffn_dim=32)


def test_softmax_and_top_k() -> None:
    logits = np.array([[1.0, 2.0, 3.0, 0.5], [-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    probs = _softmax_np(logits)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-6)

    masked = _top_k_logits(logits, k=2)
    # With distinct values each row keeps exactly two finite entries.
    finite = np.isfinite(masked)
    assert (finite.sum(axis=-1) == 2).all()


def test_sample_tokens_shape_and_dtype() -> None:
    keras.utils.set_random_seed(0)
    prior = build_prior(
        vocab_size=VOCAB,
        context_length=CTX,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
    )
    rng = np.random.default_rng(0)
    out = sample_tokens(
        prior,
        num_samples=2,
        context_length=CTX,
        vocab_size=VOCAB,
        temperature=1.0,
        top_k=5,
        rng=rng,
    )
    assert out.shape == (2, CTX)
    assert out.dtype == np.int32
    assert out.min() >= 0 and out.max() < VOCAB


def test_seed_tokens_are_preserved() -> None:
    prior = build_prior(
        vocab_size=VOCAB,
        context_length=CTX,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
    )
    seed = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    out = sample_tokens(
        prior,
        num_samples=2,
        context_length=CTX,
        vocab_size=VOCAB,
        seed_tokens=seed,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(out[:, :4], seed)
