"""Helper utilities for building ECG RVQ encoder/decoder stacks."""

from __future__ import annotations

import math
from typing import Dict

import keras

from compressionkit.layers.residual_vector_quantizer import ResidualVectorQuantizer


def make_divisible(value: float, divisor: int, min_value: int | None = None) -> int:
    """Round value to be divisible by divisor while staying close to original."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * value:
        new_v += divisor
    return int(new_v)


def conv2d_block(
    x,
    filters: int,
    k_w: int = 7,
    stride_w: int = 1,
    name: str | None = None,
    block_norm: str = "batch",
):
    x = keras.layers.Conv2D(filters, (1, k_w), strides=(1, stride_w), padding="same", name=None if name is None else f"{name}_conv")(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_act")(x)
    return x


def depthwise2d_block(
    x,
    filters: int,
    k_w: int = 7,
    stride_w: int = 1,
    name: str | None = None,
    block_norm: str = "batch",
):
    x = keras.layers.DepthwiseConv2D((1, k_w), strides=(1, stride_w), padding="same", name=None if name is None else f"{name}_dw")(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_dw_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_dw_act")(x)
    x = keras.layers.Conv2D(filters, (1, 1), padding="same", name=None if name is None else f"{name}_pw")(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_pw_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_pw_act")(x)
    return x


def _apply_norm_2d(x, mode: str, name: str) -> keras.KerasTensor:
    """Apply optional normalization on channels-last 2D tensors."""
    norm_mode = str(mode).strip().lower()
    if norm_mode == "batch":
        return keras.layers.BatchNormalization(axis=-1, epsilon=1e-5, name=name)(x)
    if norm_mode == "layer":
        return keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name=name)(x)
    if norm_mode in {"none", "off"}:
        return x
    raise ValueError(f"Unsupported normalization mode: {mode}")


def up2d_block(
    x,
    filters: int,
    k_w: int = 7,
    name: str | None = None,
    block_norm: str = "none",
):
    x = keras.layers.UpSampling2D(size=(1, 2), name=None if name is None else f"{name}_up")(x)
    x = keras.layers.Conv2D(filters, (1, k_w), padding="same", name=None if name is None else f"{name}_conv")(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_conv_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_act")(x)
    x = keras.layers.SeparableConv2D(filters, (1, 5), padding="same", name=None if name is None else f"{name}_aa")(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_aa_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_aa_act")(x)
    return x


def build_encoder_16x_2d(input_len: int = 2048, in_ch: int = 1, base: int = 32, embedding_dim: int = 16, multiplier: float = 1.25):
    """Construct a stride-2-per-stage encoder used by ECG RVQ models."""
    return build_encoder_2d(
        input_len=input_len,
        in_ch=in_ch,
        base=base,
        embedding_dim=embedding_dim,
        multiplier=multiplier,
        num_stages=4,
    )


def build_encoder_2d(
    input_len: int = 2048,
    in_ch: int = 1,
    base: int = 32,
    embedding_dim: int = 16,
    multiplier: float = 1.25,
    num_stages: int = 4,
    block_norm: str = "batch",
    head_norm: str = "none",
):
    """Construct a configurable encoder with ``2**num_stages`` downsampling."""
    downsample_factor = 2 ** num_stages
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if input_len % downsample_factor != 0:
        raise ValueError(
            f"input_len ({input_len}) must be divisible by 2**num_stages ({downsample_factor})"
        )

    inp = keras.layers.Input(shape=(1, input_len, in_ch), name="ecg_in")
    x = inp
    filters = base
    conv_stages = min(2, num_stages)
    for stage in range(conv_stages):
        x = conv2d_block(x, filters, stride_w=2, name=f"enc_s{stage + 1}", block_norm=block_norm)
        filters = make_divisible(filters * multiplier, 8)
    for stage in range(conv_stages, num_stages):
        x = depthwise2d_block(x, filters, stride_w=2, name=f"enc_s{stage + 1}", block_norm=block_norm)
        filters = make_divisible(filters * multiplier, 8)
    x = keras.layers.Conv2D(embedding_dim, (1, 1), padding="same", name="to_vq")(x)
    x = _apply_norm_2d(x, head_norm, name="enc_head_norm")
    return keras.Model(inp, x, name=f"Encoder2D_ds{downsample_factor}")


def build_decoder_16x_2d(output_len: int = 2048, out_ch: int = 1, base: int = 32, embedding_dim: int = 16, multiplier: float = 1.25):
    """Construct a decoder that mirrors a 16× encoder."""
    return build_decoder_2d(
        output_len=output_len,
        out_ch=out_ch,
        base=base,
        embedding_dim=embedding_dim,
        multiplier=multiplier,
        num_stages=4,
    )


def build_decoder_2d(
    output_len: int = 2048,
    out_ch: int = 1,
    base: int = 32,
    embedding_dim: int = 16,
    multiplier: float = 1.25,
    num_stages: int = 4,
    decoder_block_norm: str = "none",
    head_norm: str = "layer",
):
    """Construct a configurable decoder that mirrors the encoder stages."""
    downsample_factor = 2 ** num_stages
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if output_len % downsample_factor != 0:
        raise ValueError(
            f"output_len ({output_len}) must be divisible by 2**num_stages ({downsample_factor})"
        )

    inp = keras.layers.Input(shape=(1, output_len // downsample_factor, embedding_dim), name="latent_in")
    x = inp
    filters = make_divisible(base * (multiplier ** max(num_stages - 1, 0)), 8)
    for stage in range(num_stages):
        x = up2d_block(
            x,
            filters,
            name=f"dec_s{stage + 1}",
            block_norm=decoder_block_norm,
        )
        filters = max(8, make_divisible(filters / multiplier, 8))
    norm_mode = str(head_norm).strip().lower()
    if norm_mode == "layer":
        x = keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="head_ln")(x)
    elif norm_mode == "batch":
        x = keras.layers.BatchNormalization(axis=-1, epsilon=1e-5, name="head_bn")(x)
    elif norm_mode in {"none", "off"}:
        pass
    else:
        raise ValueError(f"Unsupported decoder head_norm: {head_norm}")
    out = keras.layers.Conv2D(out_ch, (1, 1), padding="same", name="out")(x)
    return keras.Model(inp, out, name=f"Decoder2D_ds{downsample_factor}")


def build_rvq_autoencoder(
    frame_size: int,
    *,
    embedding_dim: int = 16,
    latent_width: int = 256,
    in_ch: int = 1,
    out_ch: int = 1,
    base_filters: int = 32,
    multiplier: float = 1.25,
    num_levels: int = 2,
    beta: float = 0.25,
    num_stages: int = 4,
    encoder_block_norm: str = "batch",
    encoder_head_norm: str = "none",
    decoder_block_norm: str = "none",
    decoder_head_norm: str = "layer",
):
    """Build encoder, RVQ bottleneck, decoder, and composite model."""
    downsample_factor = 2 ** num_stages
    if frame_size % downsample_factor != 0:
        raise ValueError(
            f"frame_size ({frame_size}) must be divisible by 2**num_stages ({downsample_factor})"
        )

    encoder = build_encoder_2d(
        input_len=frame_size,
        in_ch=in_ch,
        base=base_filters,
        embedding_dim=embedding_dim,
        multiplier=multiplier,
        num_stages=num_stages,
        block_norm=encoder_block_norm,
        head_norm=encoder_head_norm,
    )
    decoder = build_decoder_2d(
        output_len=frame_size,
        out_ch=out_ch,
        base=base_filters,
        embedding_dim=embedding_dim,
        multiplier=multiplier,
        num_stages=num_stages,
        decoder_block_norm=decoder_block_norm,
        head_norm=decoder_head_norm,
    )
    rvq = ResidualVectorQuantizer(
        num_levels=num_levels,
        num_embeddings=latent_width,
        embedding_dim=embedding_dim,
        beta=beta,
    )

    inp = keras.layers.Input(shape=(1, frame_size, in_ch), name="ecg_in")
    z = encoder(inp)
    zq = rvq(z)
    out = decoder(zq)
    model = keras.Model(inp, out, name=f"RVQAE_2D_ds{downsample_factor}")
    return encoder, rvq, decoder, model


def compute_compression_stats(
    frame_size: int,
    *,
    bit_depth: int,
    latent_width: int,
    num_levels: int,
    downsample_factor: int = 16,
) -> Dict[str, float]:
    """Return helper metrics that describe compression achieved by RVQ settings."""
    latent_positions = frame_size // downsample_factor
    bits_per_index = math.log2(latent_width)
    compressed_bits = latent_positions * num_levels * bits_per_index
    raw_bits = frame_size * bit_depth
    ratio = raw_bits / compressed_bits if compressed_bits else float("inf")
    return {
        "frame_size": frame_size,
        "latent_positions": latent_positions,
        "bits_per_index": bits_per_index,
        "compressed_bits_per_window": compressed_bits,
        "raw_bits_per_window": raw_bits,
        "compression_ratio": ratio,
    }


__all__ = [
    "build_encoder_2d",
    "build_encoder_16x_2d",
    "build_decoder_2d",
    "build_decoder_16x_2d",
    "build_rvq_autoencoder",
    "compute_compression_stats",
]
