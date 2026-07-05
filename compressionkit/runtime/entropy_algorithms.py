"""Pluggable entropy-coding algorithms for RVQ (and other) token streams.

Everything here implements :class:`compressionkit.pipeline.stages.EntropyCoder`
(``encode(symbols) -> (bitstream, nbits)`` / ``decode(bitstream, n_symbols) ->
symbols``), the same narrow protocol already satisfied by
:class:`compressionkit.pipeline.dsp_stages.RawEntropy`, ``DeflateEntropy``,
and ``LzmaEntropy``. That means every algorithm below is a drop-in
replacement for the entropy stage of a :class:`compressionkit.pipeline.codec.PipelineCodec`,
*and* independently benchmarkable via
:mod:`compressionkit.evaluation.entropy_benchmark` — new algorithms only need
to satisfy this one interface to be tried and compared like-for-like.

Two families are provided:

* :class:`LearnedEntropyCoder` — wraps any causal probability model (an AI
  prior) satisfying :class:`PriorLike`, using either the classical arithmetic
  coder or rANS as the underlying bit-packer (see
  :mod:`compressionkit.runtime.two_stage` for the coder implementations).
* :class:`StaticHistogramEntropyCoder` / :class:`MarkovEntropyCoder` —
  non-AI classical baselines (order-0 / order-1) using the exact same
  arithmetic/rANS backends, so "does the AI prior actually help" is measured
  by swapping one object, not by comparing unrelated code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from compressionkit.runtime.two_stage import (
    _arithmetic_decode_with_prior,
    _arithmetic_encode,
    _rans_decode_with_prior,
    _rans_encode,
)

__all__ = [
    "LearnedEntropyCoder",
    "MarkovEntropyCoder",
    "PriorLike",
    "StaticHistogramEntropyCoder",
]

Backend = Literal["arithmetic", "rans"]


@runtime_checkable
class PriorLike(Protocol):
    """Minimal interface a causal probability model must expose.

    Satisfied by :class:`compressionkit.runtime.prior.EntropyPrior` (TFLite,
    production) as well as lightweight research wrappers around a Keras
    model (see ``scripts/benchmark_rans_vs_arithmetic.py``), so
    :class:`LearnedEntropyCoder` works with either without modification.
    """

    vocab_size: int
    context_length: int

    def predict_next_probs(self, context_tokens: np.ndarray) -> np.ndarray:
        """Return ``(batch, vocab_size)`` probs for the token after ``context_tokens``.

        ``context_tokens`` has shape ``(batch, seq_len)``; ``seq_len`` may be
        ``0`` (no context yet).
        """
        ...


def _probs_from_prior(prior: PriorLike, tokens: np.ndarray) -> np.ndarray:
    """Per-position probability table matching ``TwoStageCodec._get_probs``.

    Position 0 is always uniform (no context available yet); position ``t``
    uses the true tokens ``[0:t]`` seen so far, bounded to the prior's
    context window. This exact convention must match on the encode and
    decode side or the bitstream will not round-trip.
    """
    num_tokens = tokens.shape[0]
    vocab = prior.vocab_size
    probs = np.full((num_tokens, vocab), 1.0 / vocab, dtype=np.float32)
    tokens_2d = tokens.reshape(1, -1)
    for pos in range(1, num_tokens):
        start = max(0, pos - prior.context_length)
        probs[pos] = prior.predict_next_probs(tokens_2d[:, start:pos])[0]
    return probs


@dataclass
class LearnedEntropyCoder:
    """Entropy coder driven by an AI prior's predicted probabilities.

    Args:
        prior: Any :class:`PriorLike` causal probability model.
        backend: ``"rans"`` (default, production-preferred) or ``"arithmetic"``.
        name: Defaults to ``f"learned_{backend}"``.

    Example::

        coder = LearnedEntropyCoder(prior=my_wavenet_prior, backend="rans")
        bitstream, nbits = coder.encode(rvq_token_stream)
        recovered = coder.decode(bitstream, len(rvq_token_stream))
    """

    prior: PriorLike
    backend: Backend = "rans"
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"learned_{self.backend}"

    def encode(self, symbols: np.ndarray) -> tuple[bytes, int]:
        tokens = np.asarray(symbols, dtype=np.int32).reshape(-1)
        probs = _probs_from_prior(self.prior, tokens)
        encode_fn = _arithmetic_encode if self.backend == "arithmetic" else _rans_encode
        bitstream = encode_fn(tokens, probs, self.prior.vocab_size)
        return bitstream, len(bitstream) * 8

    def decode(self, bitstream: bytes, n_symbols: int) -> np.ndarray:
        decode_fn = _arithmetic_decode_with_prior if self.backend == "arithmetic" else _rans_decode_with_prior
        return decode_fn(
            bitstream=bitstream,
            num_tokens=n_symbols,
            indices_shape=(n_symbols,),
            prior=self.prior,
            vocab_size=self.prior.vocab_size,
        )


class _StaticPriorAdapter:
    """Adapts a fixed probability vector (order-0) to :class:`PriorLike`.

    Position 0 still reports uniform (to match the decode helpers'
    hardcoded convention for "no context yet") even though a static model
    has no real reason to treat position 0 specially; the cost is one
    token's worth of negligible overhead.
    """

    def __init__(self, probs: np.ndarray) -> None:
        self.vocab_size = len(probs)
        self.context_length = 1 << 30  # unbounded; static model ignores context anyway
        self._probs = probs.astype(np.float32)

    def predict_next_probs(self, context_tokens: np.ndarray) -> np.ndarray:
        batch = context_tokens.shape[0]
        return np.tile(self._probs, (batch, 1))


@dataclass
class StaticHistogramEntropyCoder:
    """Order-0 classical baseline: a single fixed histogram, no AI.

    Uses the exact same arithmetic/rANS backends as :class:`LearnedEntropyCoder`
    so any bpt difference is attributable purely to the probability source,
    not the bit-packing algorithm.
    """

    probs: np.ndarray
    backend: Backend = "rans"
    name: str = field(default="static_order0")

    @classmethod
    def fit(cls, tokens: np.ndarray, vocab_size: int, **kwargs) -> StaticHistogramEntropyCoder:
        """Fit a histogram from a (typically training-split) token stream."""
        counts = np.bincount(np.asarray(tokens).astype(np.int64), minlength=vocab_size).astype(np.float64)
        probs = (counts / max(counts.sum(), 1)).astype(np.float32)
        return cls(probs=probs, **kwargs)

    def _coder(self) -> LearnedEntropyCoder:
        return LearnedEntropyCoder(prior=_StaticPriorAdapter(self.probs), backend=self.backend, name=self.name)

    def encode(self, symbols: np.ndarray) -> tuple[bytes, int]:
        return self._coder().encode(symbols)

    def decode(self, bitstream: bytes, n_symbols: int) -> np.ndarray:
        return self._coder().decode(bitstream, n_symbols)


class _MarkovPriorAdapter:
    """Adapts a fitted order-1 transition matrix to :class:`PriorLike`."""

    def __init__(self, transition_probs: np.ndarray, fallback_probs: np.ndarray) -> None:
        self.vocab_size = transition_probs.shape[0]
        self.context_length = 1  # only the immediately preceding token matters
        self._transition = transition_probs.astype(np.float32)
        self._fallback = fallback_probs.astype(np.float32)

    def predict_next_probs(self, context_tokens: np.ndarray) -> np.ndarray:
        batch, seq_len = context_tokens.shape
        if seq_len == 0:
            return np.tile(self._fallback, (batch, 1))
        last = context_tokens[:, -1]
        return self._transition[last]


@dataclass
class MarkovEntropyCoder:
    """Order-1 classical baseline: next-token probs depend only on the previous token."""

    transition_probs: np.ndarray
    fallback_probs: np.ndarray
    backend: Backend = "rans"
    name: str = field(default="static_order1")

    @classmethod
    def fit(cls, tokens: np.ndarray, vocab_size: int, **kwargs) -> MarkovEntropyCoder:
        """Fit a row-normalized transition matrix from a token stream."""
        tokens = np.asarray(tokens).astype(np.int64)
        joint = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        np.add.at(joint, (tokens[:-1], tokens[1:]), 1)
        row_sums = joint.sum(axis=1, keepdims=True)
        transition = np.divide(joint, np.maximum(row_sums, 1), where=row_sums > 0)
        # Rows with no observed transitions fall back to the marginal distribution.
        marginal = np.bincount(tokens, minlength=vocab_size).astype(np.float64)
        marginal = marginal / max(marginal.sum(), 1)
        transition[row_sums.ravel() == 0] = marginal
        return cls(transition_probs=transition.astype(np.float32), fallback_probs=marginal.astype(np.float32), **kwargs)

    def _coder(self) -> LearnedEntropyCoder:
        prior = _MarkovPriorAdapter(self.transition_probs, self.fallback_probs)
        return LearnedEntropyCoder(prior=prior, backend=self.backend, name=self.name)

    def encode(self, symbols: np.ndarray) -> tuple[bytes, int]:
        return self._coder().encode(symbols)

    def decode(self, bitstream: bytes, n_symbols: int) -> np.ndarray:
        return self._coder().decode(bitstream, n_symbols)
