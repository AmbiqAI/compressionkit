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

from compressionkit.generative.causal_priors import (
    SplitHalf,
    build_cnn_prior,
    build_cnngru_prior,
    build_dscnn_prior,
    build_gru_prior,
    build_hybrid_prior,
    build_wavenet_bit_prior,
    build_wavenet_prior,
)
from compressionkit.generative.sampling import decode_tokens_to_signal, sample_signals
from compressionkit.generative.token_extraction import extract_rvq_tokens
from compressionkit.generative.transformer_prior import build_prior
from compressionkit.generative.xlead_prior import (
    build_xlead_concat_prior,
    build_xlead_interleave_prior,
    deinterleave_leads,
    interleave_leads,
)

__all__ = [
    "SplitHalf",
    "build_cnn_prior",
    "build_cnngru_prior",
    "build_dscnn_prior",
    "build_gru_prior",
    "build_hybrid_prior",
    "build_prior",
    "build_wavenet_bit_prior",
    "build_wavenet_prior",
    "build_xlead_concat_prior",
    "build_xlead_interleave_prior",
    "decode_tokens_to_signal",
    "deinterleave_leads",
    "extract_rvq_tokens",
    "interleave_leads",
    "sample_signals",
]
