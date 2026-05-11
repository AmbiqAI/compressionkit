"""Shared building blocks for encoder and decoder architectures.

All block functions take and return ``keras.KerasTensor`` and follow the
convention ``(B, 1, T, C)`` for temporal (1-D signal) models and
``(B, H, W, C)`` for spatial (STFT spectrogram) models.
"""

from __future__ import annotations

import keras


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_divisible(value: float, divisor: int, min_value: int | None = None) -> int:
    """Round *value* to the nearest multiple of *divisor*."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * value:
        new_v += divisor
    return int(new_v)


def _apply_norm_2d(x: keras.KerasTensor, mode: str, name: str) -> keras.KerasTensor:
    """Apply optional normalization on channels-last 2D tensors."""
    norm_mode = str(mode).strip().lower()
    if norm_mode == "batch":
        return keras.layers.BatchNormalization(axis=-1, epsilon=1e-5, name=name)(x)
    if norm_mode == "layer":
        return keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name=name)(x)
    if norm_mode in {"none", "off"}:
        return x
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _apply_activation(x: keras.KerasTensor, activation: str, name: str | None = None) -> keras.KerasTensor:
    """Apply activation — supports 'snake' as well as standard Keras names."""
    if activation == "snake":
        from compressionkit.layers import Snake
        return Snake(name=name)(x)
    return keras.layers.Activation(activation, name=name)(x)


def _shortcut_2d(
    x: keras.KerasTensor,
    filters: int,
    stride_w: int = 1,
    name: str | None = None,
) -> keras.KerasTensor:
    """1x1 projection shortcut with optional temporal stride."""
    return keras.layers.Conv2D(
        filters, (1, 1), strides=(1, stride_w), padding="same",
        name=None if name is None else f"{name}_proj",
    )(x)


# ---------------------------------------------------------------------------
# Temporal building blocks (B, 1, T, C)
# ---------------------------------------------------------------------------

def conv2d_block(
    x: keras.KerasTensor,
    filters: int,
    k_w: int = 7,
    stride_w: int = 1,
    name: str | None = None,
    block_norm: str = "batch",
) -> keras.KerasTensor:
    """Standard Conv2D + norm + ReLU block."""
    x = keras.layers.Conv2D(
        filters, (1, k_w), strides=(1, stride_w), padding="same",
        name=None if name is None else f"{name}_conv",
    )(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_act")(x)
    return x


def depthwise2d_block(
    x: keras.KerasTensor,
    filters: int,
    k_w: int = 7,
    stride_w: int = 1,
    name: str | None = None,
    block_norm: str = "batch",
) -> keras.KerasTensor:
    """Depthwise-separable Conv2D + norm + ReLU block."""
    x = keras.layers.DepthwiseConv2D(
        (1, k_w), strides=(1, stride_w), padding="same",
        name=None if name is None else f"{name}_dw",
    )(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_dw_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_dw_act")(x)
    x = keras.layers.Conv2D(
        filters, (1, 1), padding="same",
        name=None if name is None else f"{name}_pw",
    )(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_pw_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_pw_act")(x)
    return x


def inverted_residual_2d_block(
    x: keras.KerasTensor,
    filters: int,
    k_w: int = 7,
    stride_w: int = 1,
    expand_ratio: float = 4.0,
    name: str | None = None,
    block_norm: str = "batch",
    causal: bool = False,
) -> keras.KerasTensor:
    """Inverted residual block (MobileNetV2-style): PW-expand -> DW -> PW-project.

    Downsampling uses an average-pool after the depthwise conv (rather than
    stride on the DW itself) to avoid the TF GPU kernel restriction on
    unequal ``DepthwiseConv2D`` strides.

    When ``stride_w == 1`` and the input/output channel counts match, a
    residual shortcut is added automatically.
    """
    in_ch = x.shape[-1]
    hidden = make_divisible(in_ch * expand_ratio, 8)
    use_shortcut = stride_w == 1 and in_ch == filters

    # 1) pointwise expand
    h = keras.layers.Conv2D(
        hidden, (1, 1), padding="same",
        name=None if name is None else f"{name}_pw_exp",
    )(x)
    h = _apply_norm_2d(h, block_norm, name=None if name is None else f"{name}_pw_exp_norm")
    h = keras.layers.Activation("relu6", name=None if name is None else f"{name}_pw_exp_act")(h)

    # 2) depthwise (stride-1, optionally causal) + separate pooling
    if causal and k_w > 1:
        pad_left = k_w - 1
        h = keras.layers.ZeroPadding2D(
            padding=((0, 0), (pad_left, 0)),
            name=None if name is None else f"{name}_dw_pad",
        )(h)
        dw_padding = "valid"
    else:
        dw_padding = "same"

    h = keras.layers.DepthwiseConv2D(
        (1, k_w), strides=(1, 1), padding=dw_padding,
        name=None if name is None else f"{name}_dw",
    )(h)
    h = _apply_norm_2d(h, block_norm, name=None if name is None else f"{name}_dw_norm")
    h = keras.layers.Activation("relu6", name=None if name is None else f"{name}_dw_act")(h)

    # Downsample via average pooling (avoids unequal DW stride issue)
    if stride_w > 1:
        h = keras.layers.AveragePooling2D(
            pool_size=(1, stride_w),
            name=None if name is None else f"{name}_pool",
        )(h)

    # 3) pointwise project (linear — no activation)
    h = keras.layers.Conv2D(
        filters, (1, 1), padding="same",
        name=None if name is None else f"{name}_pw_proj",
    )(h)
    h = _apply_norm_2d(h, block_norm, name=None if name is None else f"{name}_pw_proj_norm")

    if use_shortcut:
        h = keras.layers.Add(name=None if name is None else f"{name}_add")([x, h])
    return h


def up2d_block(
    x: keras.KerasTensor,
    filters: int,
    k_w: int = 7,
    name: str | None = None,
    block_norm: str = "none",
    activation: str = "relu",
) -> keras.KerasTensor:
    """Upsample + conv + separable anti-alias block."""
    x = keras.layers.UpSampling2D(size=(1, 2), name=None if name is None else f"{name}_up")(x)
    x = keras.layers.Conv2D(
        filters, (1, k_w), padding="same",
        name=None if name is None else f"{name}_conv",
    )(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_conv_norm")
    x = _apply_activation(x, activation, name=None if name is None else f"{name}_act")
    x = keras.layers.SeparableConv2D(
        filters, (1, 5), padding="same",
        name=None if name is None else f"{name}_aa",
    )(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_aa_norm")
    x = _apply_activation(x, activation, name=None if name is None else f"{name}_aa_act")
    return x


def res_conv2d_block(
    x: keras.KerasTensor,
    filters: int,
    k_w: int = 7,
    stride_w: int = 1,
    name: str | None = None,
    block_norm: str = "batch",
) -> keras.KerasTensor:
    """Conv2D block with residual shortcut (pre-activation style)."""
    shortcut = _shortcut_2d(x, filters, stride_w, name=name)
    x = conv2d_block(x, filters, k_w=k_w, stride_w=stride_w, name=name, block_norm=block_norm)
    return keras.layers.Add(name=None if name is None else f"{name}_add")([shortcut, x])


def res_depthwise2d_block(
    x: keras.KerasTensor,
    filters: int,
    k_w: int = 7,
    stride_w: int = 1,
    name: str | None = None,
    block_norm: str = "batch",
) -> keras.KerasTensor:
    """Depthwise-separable block with residual shortcut."""
    shortcut = _shortcut_2d(x, filters, stride_w, name=name)
    x = depthwise2d_block(x, filters, k_w=k_w, stride_w=stride_w, name=name, block_norm=block_norm)
    return keras.layers.Add(name=None if name is None else f"{name}_add")([shortcut, x])


def res_up2d_block(
    x: keras.KerasTensor,
    filters: int,
    k_w: int = 7,
    name: str | None = None,
    block_norm: str = "none",
    activation: str = "relu",
) -> keras.KerasTensor:
    """Upsample block with residual shortcut around the conv pair."""
    shortcut = keras.layers.UpSampling2D(
        size=(1, 2), name=None if name is None else f"{name}_skip_up",
    )(x)
    shortcut = keras.layers.Conv2D(
        filters, (1, 1), padding="same",
        name=None if name is None else f"{name}_skip_proj",
    )(shortcut)
    x = up2d_block(x, filters, k_w=k_w, name=name, block_norm=block_norm, activation=activation)
    return keras.layers.Add(name=None if name is None else f"{name}_add")([shortcut, x])


# ---------------------------------------------------------------------------
# Spatial building blocks (B, H, W, C) — for STFT spectrograms
# ---------------------------------------------------------------------------

def conv2d_spatial_block(
    x: keras.KerasTensor,
    filters: int,
    kernel_size: int = 5,
    stride: int = 1,
    name: str | None = None,
    block_norm: str = "batch",
) -> keras.KerasTensor:
    """Conv2D with spatial (H, W) kernel + norm + ReLU."""
    x = keras.layers.Conv2D(
        filters, (kernel_size, kernel_size), strides=(stride, stride), padding="same",
        name=None if name is None else f"{name}_conv",
    )(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_act")(x)
    return x


def depthwise2d_spatial_block(
    x: keras.KerasTensor,
    filters: int,
    kernel_size: int = 5,
    stride: int = 1,
    name: str | None = None,
    block_norm: str = "batch",
) -> keras.KerasTensor:
    """Depthwise-separable Conv2D with spatial kernels + norm + ReLU."""
    x = keras.layers.DepthwiseConv2D(
        (kernel_size, kernel_size), strides=(stride, stride), padding="same",
        name=None if name is None else f"{name}_dw",
    )(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_dw_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_dw_act")(x)
    x = keras.layers.Conv2D(
        filters, (1, 1), padding="same",
        name=None if name is None else f"{name}_pw",
    )(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_pw_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_pw_act")(x)
    return x


def up2d_spatial_block(
    x: keras.KerasTensor,
    filters: int,
    kernel_size: int = 5,
    name: str | None = None,
    block_norm: str = "none",
) -> keras.KerasTensor:
    """Upsample (2,2) + conv + separable anti-alias for spatial data."""
    x = keras.layers.UpSampling2D(size=(2, 2), name=None if name is None else f"{name}_up")(x)
    x = keras.layers.Conv2D(
        filters, (kernel_size, kernel_size), padding="same",
        name=None if name is None else f"{name}_conv",
    )(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_conv_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_act")(x)
    x = keras.layers.SeparableConv2D(
        filters, (3, 3), padding="same",
        name=None if name is None else f"{name}_aa",
    )(x)
    x = _apply_norm_2d(x, block_norm, name=None if name is None else f"{name}_aa_norm")
    x = keras.layers.Activation("relu", name=None if name is None else f"{name}_aa_act")(x)
    return x
