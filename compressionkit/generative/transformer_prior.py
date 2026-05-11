"""Small causal transformer prior over RVQ codebook tokens.

The prior learns :math:`P(c_t \\mid c_{<t})` on sequences of discrete
codebook indices produced by a frozen RVQ compression model. Once
trained, it can be sampled autoregressively to produce novel token
sequences which are decoded back into signal space by the frozen
decoder.

Design choices driven by edge deployment (AGENTS.md):

* Token and positional embeddings are learned lookup tables — fixed
  size, no trig functions to compute at inference.
* Attention uses a pre-built causal mask stored as a constant; no
  dynamic shape tricks.
* All layers are pure Keras 3 ops that lower cleanly to LiteRT INT8.
"""

from __future__ import annotations

import keras


class _PositionalEmbedding(keras.layers.Layer):
    """Adds a learned ``(1, L, D)`` positional embedding to its input."""

    def __init__(self, length: int, dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.length = int(length)
        self.dim = int(dim)

    def build(self, input_shape):
        self.pe = self.add_weight(
            name="pe",
            shape=(1, self.length, self.dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pe

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"length": self.length, "dim": self.dim})
        return cfg


def _transformer_block(
    x: keras.KerasTensor,
    *,
    embed_dim: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    block_idx: int,
) -> keras.KerasTensor:
    """Pre-norm transformer block with causal self-attention + FFN."""
    # Causal self-attention
    h = keras.layers.LayerNormalization(epsilon=1e-5, name=f"blk{block_idx}_ln1")(x)
    h = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=dropout,
        name=f"blk{block_idx}_mha",
    )(h, h, use_causal_mask=True)
    x = keras.layers.Add(name=f"blk{block_idx}_res1")([x, h])

    # Feed-forward
    h = keras.layers.LayerNormalization(epsilon=1e-5, name=f"blk{block_idx}_ln2")(x)
    h = keras.layers.Dense(ffn_dim, activation="gelu", name=f"blk{block_idx}_ff1")(h)
    if dropout > 0:
        h = keras.layers.Dropout(dropout, name=f"blk{block_idx}_drop")(h)
    h = keras.layers.Dense(embed_dim, name=f"blk{block_idx}_ff2")(h)
    x = keras.layers.Add(name=f"blk{block_idx}_res2")([x, h])
    return x


def build_prior(
    *,
    vocab_size: int,
    context_length: int,
    embed_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    ffn_dim: int = 128,
    dropout: float = 0.0,
    name: str = "rvq_prior",
) -> keras.Model:
    """Construct a causal transformer prior over codebook tokens.

    Args:
        vocab_size: Codebook size (``K``) — the model predicts over ``K`` classes.
        context_length: Fixed token sequence length used at training and inference.
        embed_dim: Token/positional embedding dimensionality.
        num_layers: Transformer blocks.
        num_heads: Attention heads per block.
        ffn_dim: Hidden size of the FFN.
        dropout: Dropout probability inside blocks.

    Returns:
        Keras model that maps ``(B, context_length)`` int token ids to
        ``(B, context_length, vocab_size)`` next-token logits.
    """
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")

    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    tok_emb = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="token_embedding",
    )(tokens_in)
    x = _PositionalEmbedding(context_length, embed_dim, name="position_embedding")(tok_emb)

    for block_idx in range(num_layers):
        x = _transformer_block(
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            block_idx=block_idx,
        )

    x = keras.layers.LayerNormalization(epsilon=1e-5, name="final_ln")(x)
    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)
    return keras.Model(tokens_in, logits, name=name)


__all__ = ["build_prior"]
