"""Tests for pluggable entropy-coding algorithms (compressionkit.runtime.entropy_algorithms)."""

from __future__ import annotations

import numpy as np
import pytest

from compressionkit.pipeline.stages import EntropyCoder
from compressionkit.runtime.entropy_algorithms import (
    LearnedEntropyCoder,
    MarkovEntropyCoder,
    PriorLike,
    StaticHistogramEntropyCoder,
)


class _MockPrior:
    """Deterministic PriorLike double: probability depends only on context length."""

    def __init__(self, vocab_size: int, context_length: int = 32) -> None:
        self.vocab_size = vocab_size
        self.context_length = context_length

    def predict_next_probs(self, context_tokens: np.ndarray) -> np.ndarray:
        batch = context_tokens.shape[0]
        return np.full((batch, self.vocab_size), 1.0 / self.vocab_size, dtype=np.float32)


class TestPriorLikeProtocol:
    def test_mock_prior_satisfies_protocol(self):
        assert isinstance(_MockPrior(16), PriorLike)


class TestLearnedEntropyCoder:
    @pytest.mark.parametrize("backend", ["arithmetic", "rans"])
    def test_roundtrip(self, backend):
        vocab = 16
        rng = np.random.default_rng(0)
        tokens = rng.integers(0, vocab, size=200).astype(np.int32)
        coder = LearnedEntropyCoder(prior=_MockPrior(vocab), backend=backend)

        bitstream, nbits = coder.encode(tokens)
        assert nbits == len(bitstream) * 8
        decoded = coder.decode(bitstream, len(tokens))
        np.testing.assert_array_equal(tokens, decoded)

    def test_satisfies_entropy_coder_protocol(self):
        coder = LearnedEntropyCoder(prior=_MockPrior(16))
        assert isinstance(coder, EntropyCoder)

    def test_default_name(self):
        assert LearnedEntropyCoder(prior=_MockPrior(16), backend="rans").name == "learned_rans"
        assert LearnedEntropyCoder(prior=_MockPrior(16), backend="arithmetic").name == "learned_arithmetic"


class TestStaticHistogramEntropyCoder:
    def test_fit_and_roundtrip(self):
        vocab = 32
        rng = np.random.default_rng(1)
        raw = np.array([0.6**k for k in range(vocab)])
        p = raw / raw.sum()
        train_tokens = rng.choice(vocab, size=5000, p=p).astype(np.int32)
        val_tokens = rng.choice(vocab, size=300, p=p).astype(np.int32)

        coder = StaticHistogramEntropyCoder.fit(train_tokens, vocab_size=vocab)
        bitstream, nbits = coder.encode(val_tokens)
        decoded = coder.decode(bitstream, len(val_tokens))
        np.testing.assert_array_equal(val_tokens, decoded)

        # Should beat uniform coding on this skewed distribution.
        assert nbits / len(val_tokens) < np.log2(vocab)

    def test_satisfies_entropy_coder_protocol(self):
        coder = StaticHistogramEntropyCoder.fit(np.zeros(10, dtype=np.int32), vocab_size=4)
        assert isinstance(coder, EntropyCoder)


class TestMarkovEntropyCoder:
    def test_fit_and_roundtrip(self):
        vocab = 8
        rng = np.random.default_rng(2)
        # Strongly autocorrelated sequence: token repeats itself with high prob.
        tokens = [0]
        for _ in range(3000):
            tokens.append(tokens[-1] if rng.random() < 0.85 else rng.integers(0, vocab))
        tokens = np.array(tokens, dtype=np.int32)
        train, val = tokens[:2500], tokens[2500:]

        coder = MarkovEntropyCoder.fit(train, vocab_size=vocab)
        bitstream, nbits = coder.encode(val)
        decoded = coder.decode(bitstream, len(val))
        np.testing.assert_array_equal(val, decoded)

        # Order-1 model should beat uniform coding by a wide margin here.
        assert nbits / len(val) < np.log2(vocab) * 0.7

    def test_satisfies_entropy_coder_protocol(self):
        coder = MarkovEntropyCoder.fit(np.array([0, 1, 0, 1, 0, 1], dtype=np.int32), vocab_size=4)
        assert isinstance(coder, EntropyCoder)
