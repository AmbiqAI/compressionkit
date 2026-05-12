"""Two-stream PPG autoencoder: baseline codec + pulsatile codec.

Builds two independent RVQ autoencoders:
- A tiny model for the smooth baseline (high CR, few codes).
- A standard model for the pulsatile residual (preserves pulse morphology).

Both share the same VQAutoencoder architecture from helia_edge but with
independently configurable sizes.
"""

from __future__ import annotations

import keras

from compressionkit.configs.ppg_two_stream import (
    PpgTwoStreamConfig,
)
from compressionkit.models.rvq_autoencoder import (
    build_rvq_autoencoder,
    compute_compression_stats,
)


def build_baseline_model(
    cfg: PpgTwoStreamConfig,
) -> tuple[keras.Model, dict[str, float]]:
    """Build the baseline (trend) RVQ autoencoder.

    The baseline stream is downsampled by ``baseline_model.downsample_factor``
    before encoding, so the effective frame size for the model is smaller.

    Returns:
        ``(model, compression_stats)``
    """
    data = cfg.data
    bm = cfg.baseline_model

    # Baseline frame length after extra downsampling
    baseline_frame = data.frame_size // bm.downsample_factor
    if baseline_frame < 4:
        raise ValueError(f"baseline frame too short: {data.frame_size} // {bm.downsample_factor} = {baseline_frame}")

    encoder, rvq, decoder, model = build_rvq_autoencoder(
        frame_size=baseline_frame,
        embedding_dim=bm.embedding_dim,
        latent_width=bm.latent_width,
        num_levels=bm.num_levels,
        num_stages=bm.num_stages,
        base_filters=bm.base_filters,
        multiplier=bm.multiplier,
        beta=bm.beta,
        use_ema=bm.use_ema,
        ema_decay=bm.ema_decay,
        encoder_block_norm=bm.encoder_block_norm,
        decoder_block_norm=bm.decoder_block_norm,
        decoder_head_norm=bm.decoder_head_norm,
    )

    stats = compute_compression_stats(
        frame_size=data.frame_size,  # original frame size for overall CR
        bit_depth=cfg.evaluation.input_bit_depth,
        latent_width=bm.latent_width,
        num_levels=bm.num_levels,
        downsample_factor=bm.downsample_factor * (2**bm.num_stages),
    )
    return model, stats


def build_pulsatile_model(
    cfg: PpgTwoStreamConfig,
) -> tuple[keras.Model, dict[str, float]]:
    """Build the pulsatile (residual) RVQ autoencoder.

    Returns:
        ``(model, compression_stats)``
    """
    data = cfg.data
    pm = cfg.pulsatile_model

    encoder, rvq, decoder, model = build_rvq_autoencoder(
        frame_size=data.frame_size,
        embedding_dim=pm.embedding_dim,
        latent_width=pm.latent_width,
        num_levels=pm.num_levels,
        num_stages=pm.num_stages,
        base_filters=pm.base_filters,
        multiplier=pm.multiplier,
        beta=pm.beta,
        use_ema=pm.use_ema,
        ema_decay=pm.ema_decay,
        encoder_block_norm=pm.encoder_block_norm,
        encoder_head_norm=pm.encoder_head_norm,
        decoder_block_norm=pm.decoder_block_norm,
        decoder_head_norm=pm.decoder_head_norm,
    )

    stats = compute_compression_stats(
        frame_size=data.frame_size,
        bit_depth=cfg.evaluation.input_bit_depth,
        latent_width=pm.latent_width,
        num_levels=pm.num_levels,
        downsample_factor=2**pm.num_stages,
    )
    return model, stats


def compute_combined_compression_stats(
    cfg: PpgTwoStreamConfig,
) -> dict[str, float]:
    """Compute the overall two-stream compression ratio.

    The total compressed bits is the sum of both streams' compressed bits
    for one frame of the original signal.
    """
    data = cfg.data
    bm = cfg.baseline_model
    pm = cfg.pulsatile_model
    import math

    bit_depth = cfg.evaluation.input_bit_depth
    raw_bits = data.frame_size * bit_depth

    # Baseline stream
    baseline_frame = data.frame_size // bm.downsample_factor
    baseline_latent_positions = baseline_frame // (2**bm.num_stages)
    baseline_bits_per_idx = math.log2(bm.latent_width)
    baseline_bits = baseline_latent_positions * bm.num_levels * baseline_bits_per_idx

    # Pulsatile stream
    pulsatile_latent_positions = data.frame_size // (2**pm.num_stages)
    pulsatile_bits_per_idx = math.log2(pm.latent_width)
    pulsatile_bits = pulsatile_latent_positions * pm.num_levels * pulsatile_bits_per_idx

    total_compressed = baseline_bits + pulsatile_bits
    ratio = raw_bits / total_compressed if total_compressed > 0 else float("inf")

    return {
        "raw_bits_per_frame": raw_bits,
        "baseline_compressed_bits": baseline_bits,
        "pulsatile_compressed_bits": pulsatile_bits,
        "total_compressed_bits": total_compressed,
        "compression_ratio": ratio,
        "baseline_fraction": baseline_bits / total_compressed if total_compressed > 0 else 0.0,
    }
