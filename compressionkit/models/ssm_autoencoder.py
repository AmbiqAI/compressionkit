"""SSM-based autoencoder for ECG/PPG compression.

Mirrors the structure of the RVQ autoencoder (strided-conv encoder, mirrored
decoder) but replaces the bottleneck latent's *temporal* model with a stack of
:class:`S4DBlock` layers from :mod:`compressionkit.layers.ssm`.  This lets the
network model long-range cardiac structure with a fixed-memory recurrence that
maps directly to embedded C / LiteRT.

The bottleneck is a continuous latent (``latent_dim`` channels at the
downsampled rate); pair with a quantizer (FSQ / RVQ) when bit-exact
compression is needed.
"""

from __future__ import annotations

import keras
from keras import layers

from ..layers.ssm import S4DBlock


def _down_block(x, filters: int, name: str):
    x = layers.Conv1D(filters, kernel_size=5, strides=2, padding="same", name=f"{name}_conv")(x)
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln")(x)
    x = layers.Activation("gelu", name=f"{name}_act")(x)
    return x


def _up_block(x, filters: int, name: str):
    x = layers.Conv1DTranspose(filters, kernel_size=5, strides=2, padding="same", name=f"{name}_tconv")(x)
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln")(x)
    x = layers.Activation("gelu", name=f"{name}_act")(x)
    return x


def build_ssm_encoder(
    *,
    frame_size: int,
    latent_dim: int = 4,
    base_filters: int = 16,
    multiplier: float = 1.5,
    num_stages: int = 4,
    state_size: int = 32,
    num_ssm_blocks: int = 2,
    name: str = "ssm_encoder",
) -> keras.Model:
    """Strided-conv encoder followed by SSM blocks at the bottleneck."""
    x_in = keras.Input(shape=(frame_size, 1), name="signal")
    x = x_in
    f = base_filters
    for stage in range(num_stages):
        x = _down_block(x, int(f), name=f"enc_s{stage}")
        f *= multiplier
    # SSM stack on the downsampled time axis.
    for i in range(num_ssm_blocks):
        x = S4DBlock(state_size=state_size, name=f"enc_ssm_{i}")(x)
    z = layers.Dense(latent_dim, name="enc_to_latent")(x)
    return keras.Model(x_in, z, name=name)


def build_ssm_decoder(
    *,
    frame_size: int,
    latent_dim: int = 4,
    base_filters: int = 16,
    multiplier: float = 1.5,
    num_stages: int = 4,
    state_size: int = 32,
    num_ssm_blocks: int = 2,
    name: str = "ssm_decoder",
) -> keras.Model:
    """Mirror of :func:`build_ssm_encoder`."""
    latent_time = frame_size // (2**num_stages)
    width = int(base_filters * (multiplier**num_stages))
    z_in = keras.Input(shape=(latent_time, latent_dim), name="latent")
    x = layers.Dense(width, name="dec_from_latent")(z_in)
    for i in range(num_ssm_blocks):
        x = S4DBlock(state_size=state_size, name=f"dec_ssm_{i}")(x)
    f = width
    for stage in range(num_stages):
        f = max(int(f / multiplier), 1)
        x = _up_block(x, f, name=f"dec_s{stage}")
    x_out = layers.Conv1D(1, kernel_size=5, padding="same", name="dec_head")(x)
    return keras.Model(z_in, x_out, name=name)


def build_ssm_autoencoder(
    *,
    frame_size: int = 512,
    latent_dim: int = 4,
    base_filters: int = 16,
    multiplier: float = 1.5,
    num_stages: int = 4,
    state_size: int = 32,
    num_ssm_blocks: int = 2,
    quantizer: keras.layers.Layer | None = None,
) -> tuple[keras.Model, keras.Model, keras.Model]:
    """Build encoder, decoder, and end-to-end autoencoder.

    Args:
        quantizer: Optional bottleneck layer (e.g. ``FiniteScalarQuantizer`` or
            ``ResidualVectorQuantizer``). It is invoked between the encoder and
            decoder; any losses/metrics it adds are propagated through the AE.

    Returns:
        ``(encoder, decoder, autoencoder)``.
    """
    enc = build_ssm_encoder(
        frame_size=frame_size,
        latent_dim=latent_dim,
        base_filters=base_filters,
        multiplier=multiplier,
        num_stages=num_stages,
        state_size=state_size,
        num_ssm_blocks=num_ssm_blocks,
    )
    dec = build_ssm_decoder(
        frame_size=frame_size,
        latent_dim=latent_dim,
        base_filters=base_filters,
        multiplier=multiplier,
        num_stages=num_stages,
        state_size=state_size,
        num_ssm_blocks=num_ssm_blocks,
    )
    x_in = keras.Input(shape=(frame_size, 1), name="signal")
    z = enc(x_in)
    if quantizer is not None:
        z = quantizer(z)
    x_out = dec(z)
    ae = keras.Model(x_in, x_out, name="ssm_autoencoder")
    return enc, dec, ae


def compute_compression_ratio(
    *,
    frame_size: int,
    latent_dim: int,
    num_stages: int,
    input_bits: int = 16,
    latent_bits: int = 32,
) -> float:
    """Compression ratio assuming uncompressed latent storage."""
    in_bits = frame_size * input_bits
    latent_time = frame_size // (2**num_stages)
    out_bits = latent_time * latent_dim * latent_bits
    return in_bits / out_bits


__all__ = [
    "build_ssm_autoencoder",
    "build_ssm_decoder",
    "build_ssm_encoder",
    "compute_compression_ratio",
]
