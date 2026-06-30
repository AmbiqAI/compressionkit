"""SoundStream/Encodec-style encoder and decoder for 1-D signal compression.

Architecture inspired by SoundStream (Zeghidour et al., 2021) and Encodec
(Defossez et al., 2022), adapted for physiological signals on edge devices.

Key design choices:
- Residual blocks with dilated convolutions at each scale for large receptive field
- Strided convolutions for downsampling (encoder) / transposed for upsampling (decoder)
- All ops are standard Conv2D — fully LiteRT-compatible and INT8 quantizable
- Uses (B, 1, T, C) tensor convention to match existing compressionkit models

Compared to the original SoundStream which uses 1-D convolutions, we use
(1, k) kernels in Conv2D to stay compatible with the existing 2-D tensor
layout used throughout compressionkit.
"""

from __future__ import annotations

import keras

from compressionkit.models.blocks import _apply_norm_2d, make_divisible


# ---------------------------------------------------------------------------
# Core building block: Residual unit with dilated convolution
# ---------------------------------------------------------------------------


def _residual_unit(
    x: keras.KerasTensor,
    filters: int,
    kernel_size: int = 7,
    dilation: int = 1,
    name: str = "res",
    norm: str = "none",
) -> keras.KerasTensor:
    """Single residual unit: dilated_conv → norm → ELU → 1x1_conv → norm → ELU + skip.

    This mirrors the SoundStream residual unit design:
        h = Conv(dilation=d)(ELU(x))
        h = Conv(1x1)(ELU(h))
        return x + h

    Args:
        x: Input tensor (B, 1, T, C).
        filters: Output channel count.
        kernel_size: Temporal kernel width for dilated conv.
        dilation: Dilation rate for the first convolution.
        name: Block name prefix.
        norm: Normalization mode ('none', 'batch', 'layer').
    """
    shortcut = x
    in_ch = x.shape[-1]

    # If channel mismatch, project the shortcut
    if in_ch != filters:
        shortcut = keras.layers.Conv2D(
            filters, (1, 1), padding="same", name=f"{name}_skip_proj"
        )(shortcut)

    # Dilated conv path
    h = keras.layers.ELU(name=f"{name}_act1")(x)
    h = keras.layers.Conv2D(
        filters,
        (1, kernel_size),
        padding="same",
        dilation_rate=(1, dilation),
        name=f"{name}_dconv",
    )(h)
    h = _apply_norm_2d(h, norm, name=f"{name}_norm1")

    # Pointwise conv
    h = keras.layers.ELU(name=f"{name}_act2")(h)
    h = keras.layers.Conv2D(
        filters, (1, 1), padding="same", name=f"{name}_pw"
    )(h)
    h = _apply_norm_2d(h, norm, name=f"{name}_norm2")

    return keras.layers.Add(name=f"{name}_add")([shortcut, h])


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def _encoder_block(
    x: keras.KerasTensor,
    filters: int,
    stride: int,
    n_residual: int = 3,
    kernel_size: int = 7,
    dilations: tuple[int, ...] = (1, 3, 9),
    name: str = "enc_blk",
    norm: str = "none",
) -> keras.KerasTensor:
    """Encoder block: N residual units + strided downsampling conv.

    Args:
        x: Input tensor.
        filters: Channel count for this block.
        stride: Temporal downsampling factor.
        n_residual: Number of residual units.
        kernel_size: Kernel width for dilated convolutions.
        dilations: Dilation rates for residual units (cycled if n_residual > len).
        name: Block name prefix.
        norm: Normalization mode.
    """
    for i in range(n_residual):
        d = dilations[i % len(dilations)]
        x = _residual_unit(
            x, filters, kernel_size=kernel_size, dilation=d,
            name=f"{name}_res{i}", norm=norm,
        )

    # Strided downsampling conv (kernel = 2*stride for good coverage)
    ds_kernel = 2 * stride
    x = keras.layers.Conv2D(
        filters,
        (1, ds_kernel),
        strides=(1, stride),
        padding="same",
        name=f"{name}_down",
    )(x)
    return x


def build_soundstream_encoder(
    input_len: int = 320,
    in_ch: int = 1,
    embedding_dim: int = 32,
    base_filters: int = 32,
    multiplier: float = 2.0,
    num_stages: int = 2,
    n_residual: int = 3,
    kernel_size: int = 7,
    dilations: tuple[int, ...] = (1, 3, 9),
    norm: str = "none",
    initial_kernel: int = 7,
    head_norm: str = "layer",
) -> keras.Model:
    """Build SoundStream-style encoder with residual blocks + strided downsampling.

    Architecture:
        1. Initial conv (kernel=initial_kernel) to bootstrap channels
        2. For each stage: N residual blocks (dilated) + strided conv (2× downsample)
        3. Final 1×1 conv to embedding_dim + normalization

    Args:
        input_len: Number of input samples per frame.
        in_ch: Input channels (1 for PPG/ECG).
        embedding_dim: Latent embedding dimension (output channels).
        base_filters: Channel count for first stage.
        multiplier: Channel multiplier per stage (doubles each stage in SoundStream).
        num_stages: Number of downsampling stages (total downsample = 2^num_stages).
        n_residual: Residual units per encoder block.
        kernel_size: Temporal kernel width for dilated convolutions.
        dilations: Dilation rate sequence for residual units.
        norm: Normalization mode for conv blocks.
        initial_kernel: Kernel width for the initial conv layer.
        head_norm: Normalization on encoder output before VQ ('layer', 'batch', 'none').

    Returns:
        Keras Model: input (B, 1, input_len, in_ch) → output (B, 1, input_len/2^S, embedding_dim).
    """
    downsample_factor = 2**num_stages
    if input_len % downsample_factor != 0:
        raise ValueError(
            f"input_len ({input_len}) must be divisible by 2^num_stages ({downsample_factor})"
        )

    inp = keras.layers.Input(shape=(1, input_len, in_ch), name="enc_in")

    # Initial conv to expand channels
    x = keras.layers.Conv2D(
        base_filters,
        (1, initial_kernel),
        padding="same",
        name="enc_initial",
    )(inp)

    # Encoder blocks with increasing channel count
    filters = base_filters
    for stage in range(num_stages):
        filters = make_divisible(base_filters * (multiplier**stage), 8) if stage > 0 else base_filters
        x = _encoder_block(
            x,
            filters=make_divisible(base_filters * (multiplier ** (stage + 1)), 8),
            stride=2,
            n_residual=n_residual,
            kernel_size=kernel_size,
            dilations=dilations,
            name=f"enc_s{stage}",
            norm=norm,
        )

    # Final projection to embedding dim + normalization for VQ stability
    x = keras.layers.ELU(name="enc_final_act")(x)
    x = keras.layers.Conv2D(
        embedding_dim, (1, 1), padding="same", name="enc_to_vq"
    )(x)
    x = _apply_norm_2d(x, head_norm, name="enc_head_norm")

    return keras.Model(inp, x, name=f"SoundStreamEncoder_ds{downsample_factor}")


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def _decoder_block(
    x: keras.KerasTensor,
    filters: int,
    stride: int,
    n_residual: int = 3,
    kernel_size: int = 7,
    dilations: tuple[int, ...] = (1, 3, 9),
    name: str = "dec_blk",
    norm: str = "none",
) -> keras.KerasTensor:
    """Decoder block: transposed conv upsample + N residual units.

    Args:
        x: Input tensor.
        filters: Output channel count.
        stride: Temporal upsampling factor.
        n_residual: Number of residual units.
        kernel_size: Kernel width for dilated convolutions.
        dilations: Dilation rates for residual units.
        name: Block name prefix.
        norm: Normalization mode.
    """
    # Transposed conv for upsampling (kernel = 2*stride)
    up_kernel = 2 * stride
    x = keras.layers.Conv2DTranspose(
        filters,
        (1, up_kernel),
        strides=(1, stride),
        padding="same",
        name=f"{name}_up",
    )(x)

    for i in range(n_residual):
        d = dilations[i % len(dilations)]
        x = _residual_unit(
            x, filters, kernel_size=kernel_size, dilation=d,
            name=f"{name}_res{i}", norm=norm,
        )
    return x


def build_soundstream_decoder(
    output_len: int = 320,
    out_ch: int = 1,
    embedding_dim: int = 32,
    base_filters: int = 32,
    multiplier: float = 2.0,
    num_stages: int = 2,
    n_residual: int = 3,
    kernel_size: int = 7,
    dilations: tuple[int, ...] = (1, 3, 9),
    norm: str = "none",
    final_kernel: int = 7,
) -> keras.Model:
    """Build SoundStream-style decoder with transposed conv upsampling + residual blocks.

    Architecture (mirrors encoder):
        1. Initial 1×1 conv from embedding_dim to widest channel count
        2. For each stage (reversed): transposed conv (2× upsample) + N residual blocks
        3. Final conv (kernel=final_kernel) to output channels

    Args:
        output_len: Number of output samples per frame.
        out_ch: Output channels (1 for PPG/ECG).
        embedding_dim: Latent embedding dimension (input channels).
        base_filters: Channel count for the last (narrowest) stage.
        multiplier: Channel multiplier per stage.
        num_stages: Number of upsampling stages.
        n_residual: Residual units per decoder block.
        kernel_size: Temporal kernel width for dilated convolutions.
        dilations: Dilation rate sequence for residual units.
        norm: Normalization mode for conv blocks.
        final_kernel: Kernel width for the output conv layer.

    Returns:
        Keras Model: input (B, 1, output_len/2^S, embedding_dim) → output (B, 1, output_len, out_ch).
    """
    downsample_factor = 2**num_stages
    if output_len % downsample_factor != 0:
        raise ValueError(
            f"output_len ({output_len}) must be divisible by 2^num_stages ({downsample_factor})"
        )

    latent_len = output_len // downsample_factor
    inp = keras.layers.Input(shape=(1, latent_len, embedding_dim), name="latent_in")

    # Initial projection from embedding_dim to widest channel count
    widest = make_divisible(base_filters * (multiplier**num_stages), 8)
    x = keras.layers.Conv2D(
        widest, (1, 1), padding="same", name="dec_from_vq"
    )(inp)

    # Decoder blocks with decreasing channel count (mirror of encoder)
    for stage in range(num_stages):
        # Channel count decreases as we go up in resolution
        out_filters = make_divisible(
            base_filters * (multiplier ** (num_stages - 1 - stage)), 8
        )
        x = _decoder_block(
            x,
            filters=out_filters,
            stride=2,
            n_residual=n_residual,
            kernel_size=kernel_size,
            dilations=dilations,
            name=f"dec_s{stage}",
            norm=norm,
        )

    # Final output conv
    x = keras.layers.ELU(name="dec_final_act")(x)
    x = keras.layers.Conv2D(
        out_ch, (1, final_kernel), padding="same", name="dec_out"
    )(x)

    return keras.Model(inp, x, name=f"SoundStreamDecoder_ds{downsample_factor}")
