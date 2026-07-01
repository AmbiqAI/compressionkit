"""MLP and Transformer encoder/decoder for DWT-domain signal compression.

These operate on packed DWT coefficient vectors (flat, same length as input).
They are NOT translation-equivariant — each position can learn its own
importance weighting, which is exactly what you want for DWT coefficients
where energy distribution is highly non-uniform across bands.

Architecture options:
  - MLP: Dense → GeLU → Dense → ... → projection to embedding_dim
  - Transformer: positional embedding + N self-attention blocks → projection

Both output shape (B, 1, latent_tokens, embedding_dim) to fit the existing
VQ + decoder pipeline.
"""

from __future__ import annotations

import keras
import numpy as np

# ---------------------------------------------------------------------------
# MLP Encoder / Decoder
# ---------------------------------------------------------------------------


def build_mlp_encoder(
    input_len: int = 320,
    in_ch: int = 1,
    embedding_dim: int = 32,
    num_stages: int = 2,
    hidden_dim: int = 512,
    num_layers: int = 3,
    dropout: float = 0.0,
) -> keras.Model:
    """MLP encoder: flattens DWT coefficients and compresses via FC layers.

    Architecture:
        input (B, 1, input_len, 1) → flatten → Dense layers → reshape to latent

    The output has shape (B, 1, input_len // 2^num_stages, embedding_dim) to
    match the VQ bottleneck interface.

    Args:
        input_len: Number of input coefficients (= frame_size for DWT-packed).
        in_ch: Input channels (1 for single-channel signals).
        embedding_dim: Latent embedding dimension per token.
        num_stages: Defines latent sequence length = input_len // 2^num_stages.
        hidden_dim: Hidden layer width.
        num_layers: Number of hidden FC layers.
        dropout: Dropout rate (0 = disabled).
    """
    downsample_factor = 2**num_stages
    latent_len = input_len // downsample_factor
    latent_total = latent_len * embedding_dim

    inp = keras.layers.Input(shape=(1, input_len, in_ch), name="enc_in")

    # Flatten: (B, 1, T, C) → (B, T*C)
    x = keras.layers.Reshape((input_len * in_ch,), name="enc_flatten")(inp)

    # Hidden layers
    for i in range(num_layers):
        x = keras.layers.Dense(hidden_dim, name=f"enc_fc{i}")(x)
        x = keras.layers.Activation("gelu", name=f"enc_act{i}")(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"enc_drop{i}")(x)

    # Project to latent space
    x = keras.layers.Dense(latent_total, name="enc_proj")(x)
    x = keras.layers.LayerNormalization(axis=-1, name="enc_ln")(x)

    # Reshape to (B, 1, latent_len, embedding_dim)
    x = keras.layers.Reshape((1, latent_len, embedding_dim), name="enc_reshape")(x)

    return keras.Model(inp, x, name=f"MLPEncoder_ds{downsample_factor}")


def build_mlp_decoder(
    output_len: int = 320,
    out_ch: int = 1,
    embedding_dim: int = 32,
    num_stages: int = 2,
    hidden_dim: int = 512,
    num_layers: int = 3,
    dropout: float = 0.0,
) -> keras.Model:
    """MLP decoder: expands latent back to DWT coefficient vector.

    Architecture:
        input (B, 1, latent_len, embedding_dim) → flatten → Dense layers → reshape

    Args:
        output_len: Number of output coefficients (= frame_size).
        out_ch: Output channels.
        embedding_dim: Latent embedding dimension per token.
        num_stages: Defines latent sequence length = output_len // 2^num_stages.
        hidden_dim: Hidden layer width.
        num_layers: Number of hidden FC layers.
        dropout: Dropout rate (0 = disabled).
    """
    downsample_factor = 2**num_stages
    latent_len = output_len // downsample_factor

    inp = keras.layers.Input(shape=(1, latent_len, embedding_dim), name="latent_in")

    # Flatten latent: (B, 1, L, D) → (B, L*D)
    x = keras.layers.Reshape((latent_len * embedding_dim,), name="dec_flatten")(inp)

    # Hidden layers
    for i in range(num_layers):
        x = keras.layers.Dense(hidden_dim, name=f"dec_fc{i}")(x)
        x = keras.layers.Activation("gelu", name=f"dec_act{i}")(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"dec_drop{i}")(x)

    # Project to output
    x = keras.layers.Dense(output_len * out_ch, name="dec_proj")(x)

    # Reshape to (B, 1, output_len, out_ch)
    x = keras.layers.Reshape((1, output_len, out_ch), name="dec_reshape")(x)

    return keras.Model(inp, x, name=f"MLPDecoder_ds{downsample_factor}")


# ---------------------------------------------------------------------------
# Transformer Encoder / Decoder
# ---------------------------------------------------------------------------


def _positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding."""
    pos = np.arange(seq_len)[:, np.newaxis]
    dim = np.arange(d_model)[np.newaxis, :]
    angles = pos / (10000 ** (2 * (dim // 2) / d_model))
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def build_transformer_encoder(
    input_len: int = 320,
    in_ch: int = 1,
    embedding_dim: int = 32,
    num_stages: int = 2,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    ff_dim: int = 256,
    dropout: float = 0.0,
    patch_size: int = 4,
) -> keras.Model:
    """Transformer encoder for DWT coefficients.

    Tokenizes the input into patches, applies self-attention, then projects
    to the latent dimension.

    Architecture:
        input (B, 1, input_len, 1) → patchify → pos_embed → N × transformer blocks → project

    Args:
        input_len: Number of input coefficients.
        in_ch: Input channels.
        embedding_dim: Output embedding dimension per latent token.
        num_stages: Defines latent token count = input_len // 2^num_stages.
        d_model: Transformer internal dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
        patch_size: Number of coefficients per input patch/token.
    """
    downsample_factor = 2**num_stages
    latent_len = input_len // downsample_factor
    n_patches = input_len // patch_size

    inp = keras.layers.Input(shape=(1, input_len, in_ch), name="enc_in")

    # Reshape to (B, input_len)
    x = keras.layers.Reshape((input_len * in_ch,), name="enc_flatten")(inp)

    # Patchify: (B, input_len) → (B, n_patches, patch_size)
    x = keras.layers.Reshape((n_patches, patch_size * in_ch), name="enc_patchify")(x)

    # Linear projection to d_model
    x = keras.layers.Dense(d_model, name="enc_patch_proj")(x)

    # Add positional encoding
    pe = _positional_encoding(n_patches, d_model)
    x = x + pe[np.newaxis, :, :]  # broadcast over batch

    # Transformer blocks
    for i in range(num_layers):
        # Multi-head self-attention
        attn_out = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            name=f"enc_mha_{i}",
        )(x, x)
        if dropout > 0:
            attn_out = keras.layers.Dropout(dropout, name=f"enc_attn_drop_{i}")(attn_out)
        x = keras.layers.Add(name=f"enc_attn_add_{i}")([x, attn_out])
        x = keras.layers.LayerNormalization(name=f"enc_attn_ln_{i}")(x)

        # Feed-forward
        ff = keras.layers.Dense(ff_dim, activation="gelu", name=f"enc_ff1_{i}")(x)
        ff = keras.layers.Dense(d_model, name=f"enc_ff2_{i}")(ff)
        if dropout > 0:
            ff = keras.layers.Dropout(dropout, name=f"enc_ff_drop_{i}")(ff)
        x = keras.layers.Add(name=f"enc_ff_add_{i}")([x, ff])
        x = keras.layers.LayerNormalization(name=f"enc_ff_ln_{i}")(x)

    # Project tokens to latent dimension.
    # If n_patches == latent_len, just project each token individually.
    # Otherwise, flatten and re-project (creates larger params).
    if n_patches == latent_len:
        # Per-token projection: (B, n_patches, d_model) → (B, n_patches, embedding_dim)
        x = keras.layers.Dense(embedding_dim, name="enc_token_proj")(x)
        x = keras.layers.LayerNormalization(name="enc_out_ln")(x)
        x = keras.layers.Reshape((1, latent_len, embedding_dim), name="enc_out_reshape")(x)
    else:
        # Flatten and project: (B, n_patches, d_model) → (B, latent_len * embedding_dim)
        x = keras.layers.Reshape((n_patches * d_model,), name="enc_pool_flat")(x)
        x = keras.layers.Dense(latent_len * embedding_dim, name="enc_pool_proj")(x)
        x = keras.layers.LayerNormalization(name="enc_out_ln")(x)
        x = keras.layers.Reshape((1, latent_len, embedding_dim), name="enc_out_reshape")(x)

    return keras.Model(inp, x, name=f"TransformerEncoder_ds{downsample_factor}")


def build_transformer_decoder(
    output_len: int = 320,
    out_ch: int = 1,
    embedding_dim: int = 32,
    num_stages: int = 2,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    ff_dim: int = 256,
    dropout: float = 0.0,
    patch_size: int = 4,
) -> keras.Model:
    """Transformer decoder that reconstructs DWT coefficients from latent tokens.

    Architecture:
        input (B, 1, latent_len, embedding_dim) → expand → pos_embed → N × transformer → unpatchify

    Args:
        output_len: Number of output coefficients.
        out_ch: Output channels.
        embedding_dim: Latent embedding dim per token.
        num_stages: Latent token count = output_len // 2^num_stages.
        d_model: Transformer internal dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
        patch_size: Output patch size (coefficients per output token).
    """
    downsample_factor = 2**num_stages
    latent_len = output_len // downsample_factor
    n_patches = output_len // patch_size

    inp = keras.layers.Input(shape=(1, latent_len, embedding_dim), name="latent_in")

    # Expand latent to sequence of patches.
    if n_patches == latent_len:
        # Per-token expansion: (B, 1, latent_len, embedding_dim) → (B, n_patches, d_model)
        x = keras.layers.Reshape((latent_len, embedding_dim), name="dec_strip_1d")(inp)
        x = keras.layers.Dense(d_model, name="dec_token_expand")(x)
    else:
        # Flatten and expand: full dense layer
        x = keras.layers.Reshape((latent_len * embedding_dim,), name="dec_flatten")(inp)
        x = keras.layers.Dense(n_patches * d_model, name="dec_expand")(x)
        x = keras.layers.Reshape((n_patches, d_model), name="dec_to_seq")(x)

    # Add positional encoding
    pe = _positional_encoding(n_patches, d_model)
    x = x + pe[np.newaxis, :, :]

    # Transformer blocks
    for i in range(num_layers):
        attn_out = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            name=f"dec_mha_{i}",
        )(x, x)
        if dropout > 0:
            attn_out = keras.layers.Dropout(dropout, name=f"dec_attn_drop_{i}")(attn_out)
        x = keras.layers.Add(name=f"dec_attn_add_{i}")([x, attn_out])
        x = keras.layers.LayerNormalization(name=f"dec_attn_ln_{i}")(x)

        ff = keras.layers.Dense(ff_dim, activation="gelu", name=f"dec_ff1_{i}")(x)
        ff = keras.layers.Dense(d_model, name=f"dec_ff2_{i}")(ff)
        if dropout > 0:
            ff = keras.layers.Dropout(dropout, name=f"dec_ff_drop_{i}")(ff)
        x = keras.layers.Add(name=f"dec_ff_add_{i}")([x, ff])
        x = keras.layers.LayerNormalization(name=f"dec_ff_ln_{i}")(x)

    # Unpatchify: (B, n_patches, d_model) → (B, output_len, out_ch)
    x = keras.layers.Dense(patch_size * out_ch, name="dec_unpatch")(x)
    x = keras.layers.Reshape((output_len * out_ch,), name="dec_flat_out")(x)
    x = keras.layers.Reshape((1, output_len, out_ch), name="dec_reshape")(x)

    return keras.Model(inp, x, name=f"TransformerDecoder_ds{downsample_factor}")
