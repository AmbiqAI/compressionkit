"""Encoder architectures for 1-D signal and 2-D spectrogram compression."""

from __future__ import annotations

import keras

from compressionkit.models.blocks import (
    _apply_norm_2d,
    conv2d_block,
    conv2d_spatial_block,
    depthwise2d_block,
    depthwise2d_spatial_block,
    inverted_residual_2d_block,
    make_divisible,
    res_conv2d_block,
    res_depthwise2d_block,
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
    use_residual: bool = False,
    blocks_per_stage: int = 1,
) -> keras.Model:
    """Build a configurable encoder with ``2**num_stages`` downsampling.

    Args:
        input_len: Number of input time samples.
        in_ch: Number of input channels.
        base: Base filter count for the first stage.
        embedding_dim: Latent channel dimension after projection.
        multiplier: Filter count multiplier per stage.
        num_stages: Number of stride-2 downsampling stages.
        block_norm: Normalization mode for conv blocks.
        head_norm: Normalization mode for the final projection.
        use_residual: If True, add shortcut connections to each stage.
        blocks_per_stage: Number of blocks per stage. First block does stride-2,
            additional blocks run at stride-1 for extra capacity.
    """
    downsample_factor = 2**num_stages
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if input_len % downsample_factor != 0:
        raise ValueError(f"input_len ({input_len}) must be divisible by 2**num_stages ({downsample_factor})")

    conv_fn = res_conv2d_block if use_residual else conv2d_block
    dw_fn = res_depthwise2d_block if use_residual else depthwise2d_block

    inp = keras.layers.Input(shape=(1, input_len, in_ch), name="enc_in")
    x = inp
    filters = base
    conv_stages = min(2, num_stages)
    for stage in range(conv_stages):
        x = conv_fn(x, filters, stride_w=2, name=f"enc_s{stage + 1}", block_norm=block_norm)
        for blk in range(1, blocks_per_stage):
            x = conv_fn(x, filters, stride_w=1, name=f"enc_s{stage + 1}_b{blk}", block_norm=block_norm)
        filters = make_divisible(filters * multiplier, 8)
    for stage in range(conv_stages, num_stages):
        x = dw_fn(x, filters, stride_w=2, name=f"enc_s{stage + 1}", block_norm=block_norm)
        for blk in range(1, blocks_per_stage):
            x = dw_fn(x, filters, stride_w=1, name=f"enc_s{stage + 1}_b{blk}", block_norm=block_norm)
        filters = make_divisible(filters * multiplier, 8)
    x = keras.layers.Conv2D(embedding_dim, (1, 1), padding="same", name="to_vq")(x)
    x = _apply_norm_2d(x, head_norm, name="enc_head_norm")
    return keras.Model(inp, x, name=f"Encoder2D_ds{downsample_factor}")


def build_encoder_2d_invres(
    input_len: int = 2048,
    in_ch: int = 1,
    base: int = 32,
    embedding_dim: int = 16,
    multiplier: float = 1.25,
    num_stages: int = 4,
    block_norm: str = "batch",
    head_norm: str = "none",
    expand_ratio: float = 4.0,
    causal: bool = False,
    discard_tail: int = 0,
) -> keras.Model:
    """Encoder using inverted-residual blocks with optional causal padding.

    The first ``min(2, num_stages)`` stages use a standard Conv2D stride-2
    (to bootstrap the channel count), and all remaining stages use
    inverted-residual blocks with stride-2 downsampling.

    When ``causal=True``, all convolutions use left-only padding so the
    encoder's receptive field is strictly backward in time.
    """
    downsample_factor = 2**num_stages
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if input_len % downsample_factor != 0:
        raise ValueError(f"input_len ({input_len}) must be divisible by 2**num_stages ({downsample_factor})")
    if discard_tail > 0 and not causal:
        raise ValueError("discard_tail > 0 requires causal=True")

    inp = keras.layers.Input(shape=(1, input_len, in_ch), name="enc_in")
    x = inp

    # Left-pad with zeros so the first `discard_tail` latent positions
    # are pure warm-up and can be safely cropped.
    overlap_samples = discard_tail * downsample_factor
    if overlap_samples > 0:
        x = keras.layers.ZeroPadding2D(
            padding=((0, 0), (overlap_samples, 0)),
            name="warmup_pad",
        )(x)

    filters = base
    conv_stages = min(2, num_stages)

    # First stages: standard conv with stride-2 (optionally causal)
    for stage in range(conv_stages):
        if causal:
            k_w = 7
            pad_left = k_w - 2  # causal pad for stride-2: k - s
            x = keras.layers.ZeroPadding2D(
                padding=((0, 0), (pad_left, 0)),
                name=f"enc_s{stage + 1}_pad",
            )(x)
            x = keras.layers.Conv2D(
                filters,
                (1, k_w),
                strides=(1, 2),
                padding="valid",
                name=f"enc_s{stage + 1}_conv",
            )(x)
        else:
            x = keras.layers.Conv2D(
                filters,
                (1, 7),
                strides=(1, 2),
                padding="same",
                name=f"enc_s{stage + 1}_conv",
            )(x)
        x = _apply_norm_2d(x, block_norm, name=f"enc_s{stage + 1}_norm")
        x = keras.layers.Activation("relu6", name=f"enc_s{stage + 1}_act")(x)
        filters = make_divisible(filters * multiplier, 8)

    # Remaining stages: inverted-residual with stride-2
    for stage in range(conv_stages, num_stages):
        x = inverted_residual_2d_block(
            x,
            filters,
            stride_w=2,
            expand_ratio=expand_ratio,
            name=f"enc_s{stage + 1}",
            block_norm=block_norm,
            causal=causal,
        )
        filters = make_divisible(filters * multiplier, 8)

    # Projection to embedding dim
    x = keras.layers.Conv2D(embedding_dim, (1, 1), padding="same", name="to_vq")(x)
    x = _apply_norm_2d(x, head_norm, name="enc_head_norm")

    # Discard warm-up latent positions
    if discard_tail > 0:
        x = keras.layers.Cropping2D(
            cropping=((0, 0), (discard_tail, 0)),
            name="discard_tail",
        )(x)

    name_parts = [f"EncoderInvRes_ds{downsample_factor}"]
    if causal:
        name_parts.append("causal")
    if discard_tail > 0:
        name_parts.append(f"dt{discard_tail}")
    return keras.Model(inp, x, name="_".join(name_parts))


def build_encoder_2d_spatial(
    input_height: int,
    input_width: int,
    in_ch: int = 2,
    base: int = 32,
    embedding_dim: int = 16,
    multiplier: float = 1.25,
    num_stages: int = 3,
    block_norm: str = "batch",
    head_norm: str = "none",
) -> keras.Model:
    """Build a 2-D spatial encoder for spectrogram inputs.

    Each stage applies stride-(2,2) downsampling in both spatial dimensions.
    """
    ds = 2**num_stages
    inp = keras.layers.Input(shape=(input_height, input_width, in_ch), name="enc_in")
    x = inp
    filters = base
    conv_stages = min(2, num_stages)
    for stage in range(conv_stages):
        x = conv2d_spatial_block(x, filters, stride=2, name=f"enc_s{stage + 1}", block_norm=block_norm)
        filters = make_divisible(filters * multiplier, 8)
    for stage in range(conv_stages, num_stages):
        x = depthwise2d_spatial_block(x, filters, stride=2, name=f"enc_s{stage + 1}", block_norm=block_norm)
        filters = make_divisible(filters * multiplier, 8)
    x = keras.layers.Conv2D(embedding_dim, (1, 1), padding="same", name="to_vq")(x)
    x = _apply_norm_2d(x, head_norm, name="enc_head_norm")
    return keras.Model(inp, x, name=f"Encoder2D_spatial_ds{ds}")
