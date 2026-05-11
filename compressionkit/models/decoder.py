"""Decoder architectures for 1-D signal and 2-D spectrogram compression."""

from __future__ import annotations

import keras

from compressionkit.models.blocks import (
    _apply_activation,
    _apply_norm_2d,
    make_divisible,
    res_up2d_block,
    up2d_block,
    up2d_spatial_block,
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
    use_residual: bool = False,
    activation: str = "relu",
) -> keras.Model:
    """Build a configurable decoder that mirrors the encoder stages.

    Args:
        output_len: Number of output time samples.
        out_ch: Number of output channels.
        base: Base filter count (mirroring the encoder).
        embedding_dim: Latent channel dimension.
        multiplier: Filter count multiplier per stage.
        num_stages: Number of upsample stages (must match encoder).
        decoder_block_norm: Normalization mode for decoder blocks.
        head_norm: Normalization mode for the output head.
        use_residual: If True, add shortcut connections to each stage.
        activation: Activation function for decoder blocks.
    """
    downsample_factor = 2 ** num_stages
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if output_len % downsample_factor != 0:
        raise ValueError(
            f"output_len ({output_len}) must be divisible by 2**num_stages ({downsample_factor})"
        )

    up_fn = res_up2d_block if use_residual else up2d_block

    inp = keras.layers.Input(
        shape=(1, output_len // downsample_factor, embedding_dim),
        name="latent_in",
    )
    x = inp
    filters = make_divisible(base * (multiplier ** max(num_stages - 1, 0)), 8)
    for stage in range(num_stages):
        x = up_fn(x, filters, name=f"dec_s{stage + 1}", block_norm=decoder_block_norm, activation=activation)
        filters = max(8, make_divisible(filters / multiplier, 8))
    x = _apply_norm_2d(x, head_norm, name="head_norm")
    out = keras.layers.Conv2D(out_ch, (1, 1), padding="same", name="out")(x)
    return keras.Model(inp, out, name=f"Decoder2D_ds{downsample_factor}")


def build_decoder_2d_ssm(
    output_len: int = 2048,
    out_ch: int = 1,
    base: int = 32,
    embedding_dim: int = 16,
    multiplier: float = 1.25,
    num_stages: int = 4,
    head_norm: str = "layer",
    state_size: int = 32,
    num_ssm_blocks: int = 2,
    max_phase: float = 3.141592653589793,
) -> keras.Model:
    """Decoder that mixes time with diagonal SSM blocks at every spatial scale.

    Drop-in shape contract for :func:`build_decoder_2d`:
        input  ``(B, 1, output_len // 2**num_stages, embedding_dim)``
        output ``(B, 1, output_len, out_ch)``
    """
    from compressionkit.layers.ssm import S4DBlock  # local import to avoid cycle

    downsample_factor = 2 ** num_stages
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if output_len % downsample_factor != 0:
        raise ValueError(
            f"output_len ({output_len}) must be divisible by "
            f"2**num_stages ({downsample_factor})"
        )

    inp = keras.layers.Input(
        shape=(1, output_len // downsample_factor, embedding_dim),
        name="latent_in",
    )
    x = inp
    filters = make_divisible(base * (multiplier ** max(num_stages - 1, 0)), 8)

    def _ssm_stack(x, name):
        x = keras.layers.Lambda(
            lambda t: keras.ops.squeeze(t, axis=1), name=f"{name}_sqz",
        )(x)
        for i in range(num_ssm_blocks):
            x = S4DBlock(
                state_size=state_size, max_phase=max_phase,
                name=f"{name}_ssm{i}",
            )(x)
        x = keras.layers.Lambda(
            lambda t: keras.ops.expand_dims(t, axis=1), name=f"{name}_unsqz",
        )(x)
        return x

    for stage in range(num_stages):
        x = keras.layers.UpSampling2D(
            size=(1, 2), name=f"dec_s{stage + 1}_up",
        )(x)
        x = keras.layers.Conv2D(
            filters, (1, 1), padding="same", name=f"dec_s{stage + 1}_proj",
        )(x)
        x = _ssm_stack(x, name=f"dec_s{stage + 1}")
        filters = max(8, make_divisible(filters / multiplier, 8))

    x = _apply_norm_2d(x, head_norm, name="head_norm")
    out = keras.layers.Conv2D(out_ch, (1, 1), padding="same", name="out")(x)
    return keras.Model(inp, out, name=f"Decoder2DSsm_ds{downsample_factor}")


def build_hierarchical_decoder_2d(
    output_len: int = 2048,
    out_ch: int = 1,
    base: int = 32,
    embedding_dim: int = 16,
    num_levels: int = 2,
    multiplier: float = 1.25,
    num_stages: int = 4,
    decoder_block_norm: str = "none",
    head_norm: str = "layer",
    use_residual: bool = False,
    activation: str = "relu",
    detail_scale: float = 0.25,
    include_sum_input: bool = False,
) -> keras.Model:
    """Build a coarse + residual/detail decoder for per-level RVQ tensors."""
    if num_levels < 2:
        raise ValueError("hierarchical decoder requires num_levels >= 2")
    downsample_factor = 2 ** num_stages
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if output_len % downsample_factor != 0:
        raise ValueError(
            f"output_len ({output_len}) must be divisible by 2**num_stages ({downsample_factor})"
        )

    up_fn = res_up2d_block if use_residual else up2d_block
    latent_len = output_len // downsample_factor
    num_input_parts = num_levels + (1 if include_sum_input else 0)
    inp = keras.layers.Input(
        shape=(1, latent_len, embedding_dim * num_input_parts),
        name="hier_latent_in",
    )
    coarse = keras.layers.Lambda(
        lambda t: t[..., :embedding_dim], name="coarse_level",
    )(inp)
    detail = keras.layers.Lambda(
        lambda t: t[..., embedding_dim:], name="detail_levels",
    )(inp)

    filters = make_divisible(base * (multiplier ** max(num_stages - 1, 0)), 8)
    coarse_filters = filters
    detail_filters = max(8, make_divisible(filters / 2.0, 8))

    x_coarse = coarse
    x_detail = detail
    for stage in range(num_stages):
        x_coarse = up_fn(
            x_coarse,
            coarse_filters,
            name=f"coarse_dec_s{stage + 1}",
            block_norm=decoder_block_norm,
            activation=activation,
        )
        x_detail = up_fn(
            x_detail,
            detail_filters,
            name=f"detail_dec_s{stage + 1}",
            block_norm=decoder_block_norm,
            activation=activation,
        )
        coarse_filters = max(8, make_divisible(coarse_filters / multiplier, 8))
        detail_filters = max(8, make_divisible(detail_filters / multiplier, 8))

    x_coarse = _apply_norm_2d(x_coarse, head_norm, name="coarse_head_norm")
    coarse_out = keras.layers.Conv2D(out_ch, (1, 1), padding="same", name="coarse_out")(x_coarse)
    detail_out = keras.layers.Conv2D(out_ch, (1, 1), padding="same", name="detail_out")(x_detail)
    if detail_scale != 1.0:
        detail_out = keras.layers.Lambda(
            lambda t: t * detail_scale, name="detail_scale",
        )(detail_out)
    out = keras.layers.Add(name="hier_out")([coarse_out, detail_out])
    name = "HybridHierDecoder2D" if include_sum_input else "HierDecoder2D"
    return keras.Model(inp, out, name=f"{name}_ds{downsample_factor}")


def build_hierarchical_adaptor_decoder_2d(
    output_len: int = 2048,
    out_ch: int = 1,
    base: int = 32,
    embedding_dim: int = 16,
    num_levels: int = 2,
    multiplier: float = 1.25,
    num_stages: int = 4,
    decoder_block_norm: str = "none",
    head_norm: str = "layer",
    use_residual: bool = False,
    activation: str = "relu",
    detail_scale: float = 0.25,
) -> keras.Model:
    """Build a summed-latent decoder with per-level residual adaptors."""
    if num_levels < 2:
        raise ValueError("hierarchical adaptor decoder requires num_levels >= 2")
    downsample_factor = 2 ** num_stages
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if output_len % downsample_factor != 0:
        raise ValueError(
            f"output_len ({output_len}) must be divisible by 2**num_stages ({downsample_factor})"
        )

    up_fn = res_up2d_block if use_residual else up2d_block
    latent_len = output_len // downsample_factor
    inp = keras.layers.Input(
        shape=(1, latent_len, embedding_dim * (num_levels + 1)),
        name="hier_adapt_latent_in",
    )
    trunk = keras.layers.Lambda(
        lambda t: t[..., :embedding_dim], name="summed_latent",
    )(inp)
    detail = keras.layers.Lambda(
        lambda t: t[..., embedding_dim:], name="detail_levels",
    )(inp)

    filters = make_divisible(base * (multiplier ** max(num_stages - 1, 0)), 8)
    x = trunk
    detail_state = detail
    for stage in range(num_stages):
        x = up_fn(
            x,
            filters,
            name=f"dec_s{stage + 1}",
            block_norm=decoder_block_norm,
            activation=activation,
        )
        detail_state = keras.layers.UpSampling2D(
            size=(1, 2), name=f"detail_s{stage + 1}_up",
        )(detail_state)
        adaptor = keras.layers.Conv2D(
            filters, (1, 1), padding="same", name=f"detail_s{stage + 1}_proj",
        )(detail_state)
        adaptor = _apply_norm_2d(
            adaptor,
            decoder_block_norm,
            name=f"detail_s{stage + 1}_norm",
        )
        adaptor = _apply_activation(
            adaptor,
            activation,
            name=f"detail_s{stage + 1}_act",
        )
        if detail_scale != 1.0:
            adaptor = keras.layers.Lambda(
                lambda t: t * detail_scale, name=f"detail_s{stage + 1}_scale",
            )(adaptor)
        x = keras.layers.Add(name=f"detail_s{stage + 1}_add")([x, adaptor])
        filters = max(8, make_divisible(filters / multiplier, 8))

    x = _apply_norm_2d(x, head_norm, name="head_norm")
    out = keras.layers.Conv2D(out_ch, (1, 1), padding="same", name="out")(x)
    return keras.Model(inp, out, name=f"HierAdaptorDecoder2D_ds{downsample_factor}")


def build_decoder_2d_spatial(
    output_height: int,
    output_width: int,
    out_ch: int = 2,
    base: int = 32,
    embedding_dim: int = 16,
    multiplier: float = 1.25,
    num_stages: int = 3,
    decoder_block_norm: str = "none",
    head_norm: str = "layer",
) -> keras.Model:
    """Build a 2-D spatial decoder that mirrors the spatial encoder."""
    ds = 2 ** num_stages
    latent_h = output_height // ds
    latent_w = output_width // ds

    inp = keras.layers.Input(
        shape=(latent_h, latent_w, embedding_dim), name="latent_in",
    )
    x = inp
    filters = make_divisible(base * (multiplier ** max(num_stages - 1, 0)), 8)
    for stage in range(num_stages):
        x = up2d_spatial_block(x, filters, name=f"dec_s{stage + 1}", block_norm=decoder_block_norm)
        filters = max(8, make_divisible(filters / multiplier, 8))
    x = _apply_norm_2d(x, head_norm, name="head_norm")
    out = keras.layers.Conv2D(out_ch, (1, 1), padding="same", name="out")(x)
    return keras.Model(inp, out, name=f"Decoder2D_spatial_ds{ds}")
