"""Generative prior models on top of frozen RVQ compression models.

This subpackage turns the frozen compression autoencoder into a
generator by learning a prior over its discrete codebook token
sequences. The decoder and RVQ codebooks are reused as-is — only a
small causal transformer needs to be trained.

Pipeline
--------

1. Extract per-frame token sequences from the trained encoder+RVQ
   using :func:`extract_rvq_tokens`.
2. Train a small causal transformer prior with :func:`build_prior`.
3. Sample new token sequences and decode them back into signals with
   :func:`sample_signals`.

All components use only operators that lower cleanly to LiteRT so the
prior itself is deployable. The decoder is already edge-ready.
"""

from compressionkit.generative.sampling import decode_tokens_to_signal, sample_signals
from compressionkit.generative.token_extraction import extract_rvq_tokens
from compressionkit.generative.transformer_prior import build_prior

__all__ = [
    "build_prior",
    "decode_tokens_to_signal",
    "extract_rvq_tokens",
    "sample_signals",
]
