"""RVQ autoencoder codec assembly and compression statistics.

This module wires encoder + bottleneck + decoder into composite VQAutoencoder
models. Building blocks and individual encoder/decoder builders live in
``compressionkit.models.blocks``, ``compressionkit.models.encoder``, and
``compressionkit.models.decoder`` respectively.

All public symbols from those modules are re-exported here for backwards
compatibility.
"""

from __future__ import annotations

import math

import keras
from helia_edge.layers import ResidualVectorQuantizer
from helia_edge.trainers import VQAutoencoder

from compressionkit.layers import EmaResidualVectorQuantizer, FiniteScalarQuantizer
from compressionkit.layers.significance_quantizer import SignificanceQuantizer

# Re-export building blocks and builders for backwards compatibility
from compressionkit.models.blocks import (  # noqa: F401
    _apply_activation,
    _apply_norm_2d,
    _shortcut_2d,
    conv2d_block,
    conv2d_spatial_block,
    depthwise2d_block,
    depthwise2d_spatial_block,
    inverted_residual_2d_block,
    make_divisible,
    res_conv2d_block,
    res_depthwise2d_block,
    res_up2d_block,
    up2d_block,
    up2d_spatial_block,
)
from compressionkit.models.decoder import (
    build_decoder_2d,
    build_decoder_2d_spatial,
    build_decoder_2d_ssm,
    build_hierarchical_adaptor_decoder_2d,
    build_hierarchical_decoder_2d,
)
from compressionkit.models.encoder import (
    build_encoder_2d,
    build_encoder_2d_invres,
    build_encoder_2d_spatial,
)
from compressionkit.models.soundstream import (
    build_soundstream_decoder,
    build_soundstream_encoder,
)
from compressionkit.models.dwt_learned import (
    build_mlp_decoder,
    build_mlp_encoder,
    build_transformer_decoder,
    build_transformer_encoder,
)

# ---------------------------------------------------------------------------
# Custom VQAutoencoder subclasses
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="compressionkit")
class PrefixSupervisedVQAutoencoder(VQAutoencoder):
    """VQ autoencoder with auxiliary coarse-to-fine RVQ prefix losses.

    The deployed model path is unchanged: encoder -> full RVQ stack -> decoder.
    During training, prefix losses decode the first N RVQ levels and compare
    them to a coarse target.
    """

    def __init__(
        self,
        encoder: keras.Model,
        vq: ResidualVectorQuantizer | EmaResidualVectorQuantizer,
        decoder: keras.Model,
        *,
        prefix_loss_weights: list[float] | None = None,
        prefix_loss_target: str = "lowpass",
        prefix_loss_lowpass_kernel: int = 9,
        prefix_loss_initial_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(encoder=encoder, vq=vq, decoder=decoder, **kwargs)
        self.prefix_loss_weights = [float(w) for w in (prefix_loss_weights or [])]
        self.prefix_loss_target = str(prefix_loss_target).lower()
        self.prefix_loss_lowpass_kernel = int(prefix_loss_lowpass_kernel)
        self.prefix_loss_initial_scale = float(prefix_loss_initial_scale)
        self.prefix_loss_scale = keras.Variable(
            self.prefix_loss_initial_scale,
            trainable=False,
            dtype="float32",
            name="prefix_loss_scale",
        )

    def _coarse_target(self, y):
        if self.prefix_loss_target in {"full", "identity"}:
            return y
        if self.prefix_loss_target != "lowpass":
            raise ValueError(f"Unsupported prefix_loss_target: {self.prefix_loss_target!r}")
        kernel = max(1, int(self.prefix_loss_lowpass_kernel))
        if kernel % 2 == 0:
            kernel += 1
        return keras.ops.average_pool(
            y,
            pool_size=(1, kernel),
            strides=(1, 1),
            padding="same",
            data_format="channels_last",
        )

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False):
        total = super().compute_loss(
            x=x,
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            allow_empty=allow_empty,
        )
        if not self.prefix_loss_weights or x is None or y is None:
            return total
        if not hasattr(self.vq, "call_at_level"):
            return total

        z = self.encoder(x, training=False)
        target = self._coarse_target(y)
        max_prefix = min(len(self.prefix_loss_weights), getattr(self.vq, "M", 1) - 1)
        for level in range(1, max_prefix + 1):
            weight = self.prefix_loss_weights[level - 1]
            if weight <= 0.0:
                continue
            zq_prefix, _indices = self.vq.call_at_level(z, level)
            y_prefix = self.decoder(zq_prefix, training=False)
            total = total + self.prefix_loss_scale * weight * keras.ops.mean(keras.ops.square(target - y_prefix))
        return total

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "prefix_loss_weights": self.prefix_loss_weights,
                "prefix_loss_target": self.prefix_loss_target,
                "prefix_loss_lowpass_kernel": self.prefix_loss_lowpass_kernel,
                "prefix_loss_initial_scale": self.prefix_loss_initial_scale,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="compressionkit")
class HierarchicalRVQAutoencoder(VQAutoencoder):
    """RVQ autoencoder that routes per-level RVQ tensors to the decoder."""

    def __init__(self, *args, include_summed_latent: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_summed_latent = bool(include_summed_latent)

    def call(
        self, x: keras.KerasTensor, training: bool = False, return_indices: bool = False
    ) -> keras.KerasTensor | tuple[keras.KerasTensor, list[keras.KerasTensor]]:
        if not hasattr(self.vq, "call_with_level_outputs"):
            raise ValueError("HierarchicalRVQAutoencoder requires a VQ layer with call_with_level_outputs().")
        z = self.encoder(x, training=training)
        _zq, level_outputs, indices = self.vq.call_with_level_outputs(
            z,
            training=training,
            return_indices=True,
        )
        inputs = [_zq, *level_outputs] if self.include_summed_latent else level_outputs
        z_hier = keras.ops.concatenate(inputs, axis=-1)
        y = self.decoder(z_hier, training=training)
        return (y, indices) if return_indices else y

    def get_config(self):
        config = super().get_config()
        config.update({"include_summed_latent": self.include_summed_latent})
        return config


# ---------------------------------------------------------------------------
# Codec factory functions
# ---------------------------------------------------------------------------


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
    use_residual: bool = False,
    use_ema: bool = False,
    ema_decay: float = 0.99,
    encoder_type: str = "default",
    expand_ratio: float = 4.0,
    causal: bool = False,
    discard_tail: int = 0,
    bottleneck_type: str = "rvq",
    fsq_levels: list[int] | None = None,
    decoder_type: str = "default",
    decoder_state_size: int = 32,
    decoder_num_ssm_blocks: int = 2,
    hier_detail_scale: float = 0.25,
    revive_dead_codes: bool = False,
    revive_threshold: float = 0.03,
    kmeans_init: bool = False,
    structured_dropout: bool = False,
    dropout_levels: list[int] | None = None,
    decoder_activation: str = "relu",
    encoder_blocks_per_stage: int = 1,
    codebook_sizes: list[int] | None = None,
    prefix_loss_weights: list[float] | None = None,
    prefix_loss_target: str = "lowpass",
    prefix_loss_lowpass_kernel: int = 9,
    prefix_loss_initial_scale: float = 1.0,
) -> tuple[
    keras.Model,
    ResidualVectorQuantizer | EmaResidualVectorQuantizer | FiniteScalarQuantizer,
    keras.Model,
    VQAutoencoder,
]:
    """Build encoder, bottleneck, decoder, and composite VQAutoencoder.

    Returns:
        ``(encoder, bottleneck, decoder, model)`` where *model* is a
        ``helia_edge.trainers.VQAutoencoder`` wrapping the three components.
    """
    downsample_factor = 2**num_stages
    if frame_size % downsample_factor != 0:
        raise ValueError(f"frame_size ({frame_size}) must be divisible by 2**num_stages ({downsample_factor})")

    bottleneck_type = bottleneck_type.lower()
    if bottleneck_type == "fsq":
        if not fsq_levels:
            raise ValueError("bottleneck_type='fsq' requires non-empty fsq_levels list")
        if embedding_dim != len(fsq_levels):
            embedding_dim = len(fsq_levels)
    elif bottleneck_type not in ("rvq", "significance"):
        raise ValueError(f"Unknown bottleneck_type: {bottleneck_type!r}")

    # --- Encoder ---
    if encoder_type == "soundstream":
        encoder = build_soundstream_encoder(
            input_len=frame_size,
            in_ch=in_ch,
            embedding_dim=embedding_dim,
            base_filters=base_filters,
            multiplier=multiplier,
            num_stages=num_stages,
            norm=encoder_block_norm,
            head_norm=encoder_head_norm,
        )
    elif encoder_type == "mlp":
        encoder = build_mlp_encoder(
            input_len=frame_size,
            in_ch=in_ch,
            embedding_dim=embedding_dim,
            num_stages=num_stages,
            hidden_dim=int(base_filters * multiplier),
            num_layers=encoder_blocks_per_stage,
        )
    elif encoder_type == "transformer":
        encoder = build_transformer_encoder(
            input_len=frame_size,
            in_ch=in_ch,
            embedding_dim=embedding_dim,
            num_stages=num_stages,
            d_model=int(base_filters * multiplier),
            num_heads=4,
            num_layers=encoder_blocks_per_stage,
            ff_dim=int(base_filters * multiplier * 2),
        )
    elif encoder_type == "inverted_residual":
        encoder = build_encoder_2d_invres(
            input_len=frame_size,
            in_ch=in_ch,
            base=base_filters,
            embedding_dim=embedding_dim,
            multiplier=multiplier,
            num_stages=num_stages,
            block_norm=encoder_block_norm,
            head_norm=encoder_head_norm,
            expand_ratio=expand_ratio,
            causal=causal,
            discard_tail=discard_tail,
        )
    else:
        encoder = build_encoder_2d(
            input_len=frame_size,
            in_ch=in_ch,
            base=base_filters,
            embedding_dim=embedding_dim,
            multiplier=multiplier,
            num_stages=num_stages,
            block_norm=encoder_block_norm,
            head_norm=encoder_head_norm,
            use_residual=use_residual,
            blocks_per_stage=encoder_blocks_per_stage,
        )

    # --- Decoder ---
    if decoder_type == "soundstream":
        decoder = build_soundstream_decoder(
            output_len=frame_size,
            out_ch=out_ch,
            embedding_dim=embedding_dim,
            base_filters=base_filters,
            multiplier=multiplier,
            num_stages=num_stages,
            norm=decoder_block_norm,
        )
    elif decoder_type == "mlp":
        decoder = build_mlp_decoder(
            output_len=frame_size,
            out_ch=out_ch,
            embedding_dim=embedding_dim,
            num_stages=num_stages,
            hidden_dim=int(base_filters * multiplier),
            num_layers=encoder_blocks_per_stage,
        )
    elif decoder_type == "transformer":
        decoder = build_transformer_decoder(
            output_len=frame_size,
            out_ch=out_ch,
            embedding_dim=embedding_dim,
            num_stages=num_stages,
            d_model=int(base_filters * multiplier),
            num_heads=4,
            num_layers=encoder_blocks_per_stage,
            ff_dim=int(base_filters * multiplier * 2),
        )
    elif decoder_type == "ssm":
        decoder = build_decoder_2d_ssm(
            output_len=frame_size,
            out_ch=out_ch,
            base=base_filters,
            embedding_dim=embedding_dim,
            multiplier=multiplier,
            num_stages=num_stages,
            head_norm=decoder_head_norm,
            state_size=decoder_state_size,
            num_ssm_blocks=decoder_num_ssm_blocks,
        )
    elif decoder_type in {"hierarchical", "hierarchical_hybrid"}:
        decoder = build_hierarchical_decoder_2d(
            output_len=frame_size,
            out_ch=out_ch,
            base=base_filters,
            embedding_dim=embedding_dim,
            num_levels=num_levels,
            multiplier=multiplier,
            num_stages=num_stages,
            decoder_block_norm=decoder_block_norm,
            head_norm=decoder_head_norm,
            use_residual=use_residual,
            activation=decoder_activation,
            detail_scale=hier_detail_scale,
            include_sum_input=(decoder_type == "hierarchical_hybrid"),
        )
    elif decoder_type == "hierarchical_adaptor":
        decoder = build_hierarchical_adaptor_decoder_2d(
            output_len=frame_size,
            out_ch=out_ch,
            base=base_filters,
            embedding_dim=embedding_dim,
            num_levels=num_levels,
            multiplier=multiplier,
            num_stages=num_stages,
            decoder_block_norm=decoder_block_norm,
            head_norm=decoder_head_norm,
            use_residual=use_residual,
            activation=decoder_activation,
            detail_scale=hier_detail_scale,
        )
    else:
        decoder = build_decoder_2d(
            output_len=frame_size,
            out_ch=out_ch,
            base=base_filters,
            embedding_dim=embedding_dim,
            multiplier=multiplier,
            num_stages=num_stages,
            decoder_block_norm=decoder_block_norm,
            head_norm=decoder_head_norm,
            use_residual=use_residual,
            activation=decoder_activation,
        )

    # --- Bottleneck ---
    if bottleneck_type == "fsq":
        bottleneck = FiniteScalarQuantizer(levels=list(fsq_levels))
        model = VQAutoencoder(
            encoder=encoder,
            vq=bottleneck,
            decoder=decoder,
            name=f"FSQAE_2D_ds{downsample_factor}",
        )
        return encoder, bottleneck, decoder, model

    if bottleneck_type == "significance":
        latent_len = frame_size // downsample_factor
        num_latents = latent_len * embedding_dim
        # keep_ratio: what fraction of latent values to keep.
        # Compute from bit budget: at input_bits/CR target, each kept value costs ~quant_bits
        # Default: keep half (can be tuned via beta repurposed as keep_ratio if >1, else rate_lambda)
        # We use num_levels to encode quant_bits, and beta as rate_lambda
        quant_bits = max(4, min(16, num_levels * 8 if num_levels >= 1 else 8))
        # Approximate keep_ratio for target CR:
        # budget_bits = frame_size * 16 / CR, where CR = frame_size / (latent_len * embedding_dim * keep_ratio * quant_bits / 16)
        # For simplicity: keep_ratio = budget_bits / (num_latents * quant_bits)
        # At 4× CR: budget = 320*16/4 = 1280, num_latents=160, qbits=8 → keep_ratio = 1280/(160*8) = 1.0
        # At 4× CR: budget = 320*16/4 = 1280, num_latents=320, qbits=8 → keep_ratio = 1280/(320*8) = 0.5
        keep_ratio = min(1.0, (frame_size * 16.0 / 4.0) / (num_latents * quant_bits))
        bottleneck = SignificanceQuantizer(
            keep_ratio=keep_ratio,
            quant_bits=quant_bits,
            rate_lambda=beta,
            temperature=0.1,
        )
        model = VQAutoencoder(
            encoder=encoder,
            vq=bottleneck,
            decoder=decoder,
            name=f"SigAE_2D_ds{downsample_factor}",
        )
        return encoder, bottleneck, decoder, model

    num_embeddings: int | list[int] = codebook_sizes if codebook_sizes else latent_width

    if use_ema:
        rvq = EmaResidualVectorQuantizer(
            num_levels=num_levels,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            beta=beta,
            ema_decay=ema_decay,
            revive_dead_codes=revive_dead_codes,
            revive_threshold=revive_threshold,
            kmeans_init=kmeans_init,
            structured_dropout=structured_dropout,
            dropout_levels=dropout_levels,
        )
    else:
        rvq = ResidualVectorQuantizer(
            num_levels=num_levels,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            beta=beta,
        )

    # --- Model class selection ---
    if decoder_type in {"hierarchical", "hierarchical_hybrid", "hierarchical_adaptor"}:
        if not use_ema:
            raise ValueError("hierarchical decoder types currently require use_ema=True")
        model_cls = HierarchicalRVQAutoencoder
    else:
        model_cls = PrefixSupervisedVQAutoencoder if prefix_loss_weights else VQAutoencoder
    model_kwargs = {
        "encoder": encoder,
        "vq": rvq,
        "decoder": decoder,
        "name": f"RVQAE_2D_ds{downsample_factor}",
    }
    if prefix_loss_weights:
        model_kwargs.update(
            {
                "prefix_loss_weights": prefix_loss_weights,
                "prefix_loss_target": prefix_loss_target,
                "prefix_loss_lowpass_kernel": prefix_loss_lowpass_kernel,
                "prefix_loss_initial_scale": prefix_loss_initial_scale,
            }
        )
    if decoder_type in {"hierarchical_hybrid", "hierarchical_adaptor"}:
        model_kwargs["include_summed_latent"] = True
    model = model_cls(**model_kwargs)

    return encoder, rvq, decoder, model


def build_rvq_autoencoder_2d_spatial(
    input_height: int,
    input_width: int,
    *,
    in_ch: int = 2,
    out_ch: int = 2,
    embedding_dim: int = 16,
    latent_width: int = 256,
    base_filters: int = 32,
    multiplier: float = 1.25,
    num_levels: int = 2,
    beta: float = 0.25,
    num_stages: int = 3,
    encoder_block_norm: str = "batch",
    encoder_head_norm: str = "none",
    decoder_block_norm: str = "none",
    decoder_head_norm: str = "layer",
) -> tuple[keras.Model, ResidualVectorQuantizer, keras.Model, VQAutoencoder]:
    """Build 2-D spatial RVQ autoencoder for STFT spectrograms.

    Returns:
        ``(encoder, rvq, decoder, model)``.
    """
    encoder = build_encoder_2d_spatial(
        input_height=input_height,
        input_width=input_width,
        in_ch=in_ch,
        base=base_filters,
        embedding_dim=embedding_dim,
        multiplier=multiplier,
        num_stages=num_stages,
        block_norm=encoder_block_norm,
        head_norm=encoder_head_norm,
    )
    decoder = build_decoder_2d_spatial(
        output_height=input_height,
        output_width=input_width,
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
    ds = 2**num_stages
    model = VQAutoencoder(
        encoder=encoder,
        vq=rvq,
        decoder=decoder,
        name=f"RVQAE_2D_spatial_ds{ds}",
    )
    return encoder, rvq, decoder, model


# ---------------------------------------------------------------------------
# Compression statistics
# ---------------------------------------------------------------------------


def compute_compression_stats(
    frame_size: int,
    *,
    bit_depth: int,
    num_channels: int = 1,
    latent_width: int,
    num_levels: int,
    downsample_factor: int = 16,
    bottleneck_type: str = "rvq",
    fsq_levels: list[int] | None = None,
    codebook_sizes: list[int] | None = None,
) -> dict[str, float]:
    """Compute compression ratio and related statistics.

    Supports both RVQ and FSQ bottlenecks. For FSQ, ``fsq_levels`` is
    required and ``latent_width`` / ``num_levels`` are ignored.
    """
    latent_positions = frame_size // downsample_factor
    bottleneck_type = bottleneck_type.lower()
    if bottleneck_type == "fsq":
        if not fsq_levels:
            raise ValueError("bottleneck_type='fsq' requires fsq_levels")
        codebook_size = 1
        for L in fsq_levels:
            codebook_size *= int(L)
        bits_per_index = math.log2(codebook_size)
        compressed_bits = latent_positions * bits_per_index
    elif codebook_sizes:
        bits_per_level = [math.log2(k) for k in codebook_sizes]
        bits_per_index = sum(bits_per_level) / len(bits_per_level)
        compressed_bits = latent_positions * sum(bits_per_level)
    else:
        bits_per_index = math.log2(latent_width)
        compressed_bits = latent_positions * num_levels * bits_per_index
    raw_bits = frame_size * int(num_channels) * bit_depth
    ratio = raw_bits / compressed_bits if compressed_bits else float("inf")
    result = {
        "frame_size": frame_size,
        "num_channels": int(num_channels),
        "latent_positions": latent_positions,
        "bits_per_index": bits_per_index,
        "compressed_bits_per_window": compressed_bits,
        "raw_bits_per_window": raw_bits,
        "compression_ratio": ratio,
        "bottleneck_type": bottleneck_type,
    }
    if codebook_sizes:
        result["codebook_sizes"] = codebook_sizes
        result["bits_per_level"] = [math.log2(k) for k in codebook_sizes]
    return result


__all__ = [
    "HierarchicalRVQAutoencoder",
    "PrefixSupervisedVQAutoencoder",
    "build_decoder_2d",
    "build_decoder_2d_spatial",
    "build_decoder_2d_ssm",
    "build_encoder_2d",
    "build_encoder_2d_invres",
    "build_encoder_2d_spatial",
    "build_hierarchical_adaptor_decoder_2d",
    "build_hierarchical_decoder_2d",
    "build_rvq_autoencoder",
    "build_rvq_autoencoder_2d_spatial",
    "compute_compression_stats",
]
