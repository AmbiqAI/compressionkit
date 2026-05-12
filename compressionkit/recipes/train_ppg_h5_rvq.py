"""Canonical-h5 PPG RVQ training recipe.

A minimal, top-to-bottom recipe that pairs the new h5-backed loader (with
patient-disjoint splits and window-level sanitization) with the same
``build_rvq_autoencoder`` model used by the legacy MESA-based PPG recipe.

Designed to train across an arbitrary mixture of canonical PPG datasets
(``bidmc``, ``ppg_dalia``, ``wesad``, ``butppg``, …). For sweeps, copy this
file into the top-level ``recipes/`` folder and edit freely.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import keras
import tensorflow as tf

from compressionkit.configs.ppg_h5_rvq import PpgH5RvqConfig
from compressionkit.datasets.ppg_h5 import (
    PpgH5Source,
    SplitConfig,
    WindowSpec,
    make_h5_ppg_dataset,
    summarize_sources,
)
from compressionkit.losses import (
    build_derivative_loss,
    build_multi_scale_spectral_loss,
)
from compressionkit.models.rvq_autoencoder import (
    build_rvq_autoencoder,
    compute_compression_stats,
)
from compressionkit.preprocessing.sanitize import SanitizeConfig
from compressionkit.recipes._registry import recipe
from compressionkit.trainers.common import (
    BEST_CKPT_NAME,
    reload_best_weights,
    save_config_snapshot,
    save_model_artifacts,
    setup_run_dir,
)
from compressionkit.trainers.utils import (
    build_callbacks,
    build_learning_rate,
    setup_logger,
)

logger = logging.getLogger("ppg-h5-rvq-trainer")


# ---------------------------------------------------------------------------
# Data + model builders
# ---------------------------------------------------------------------------


def _build_window_spec(cfg: PpgH5RvqConfig) -> WindowSpec:
    san_cfg = (
        SanitizeConfig(
            min_std=cfg.data.sanitize.min_std,
            max_saturation_frac=cfg.data.sanitize.max_saturation_frac,
            max_abs_z=cfg.data.sanitize.max_abs_z,
            max_outlier_frac=cfg.data.sanitize.max_outlier_frac,
        )
        if cfg.data.sanitize.enabled
        else None
    )
    return WindowSpec(
        target_fs=cfg.data.target_fs,
        window_seconds=cfg.data.window_seconds,
        hop_seconds=cfg.data.hop_seconds,
        sanitize=san_cfg,
        normalize=cfg.data.normalize,
    )


def _build_sources(cfg: PpgH5RvqConfig) -> list[PpgH5Source]:
    from pathlib import Path as _Path

    return [
        PpgH5Source(
            slug=src.slug,
            root=_Path(cfg.data.root),
            glob=src.glob,
            butppg_quality_only=src.butppg_quality_only,
        )
        for src in cfg.data.sources
    ]


def _to_xy(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Loader yields ``(B, 1, T)``; model expects ``(B, 1, T, 1)`` for both x and y."""
    x = tf.expand_dims(x, axis=-1)
    return x, x


def build_datasets(
    cfg: PpgH5RvqConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[str, Any]]:
    """Build train and val datasets driven by the canonical-h5 loader."""
    spec = _build_window_spec(cfg)
    sources = _build_sources(cfg)

    info: dict[str, Any] = {
        "mode": "h5",
        "target_fs": cfg.data.target_fs,
        "window_samples": spec.window_samples,
        "sources": [s.slug for s in sources],
    }

    # Quick reject-rate audit on the first few files of each source.
    audit = summarize_sources(sources, spec, max_files_per_source=3)
    info["audit"] = audit
    logger.info("Source audit: %s", json.dumps(audit, indent=2, default=str))

    split_train = SplitConfig(
        train_frac=cfg.data.split.train_frac,
        val_frac=cfg.data.split.val_frac,
        seed=cfg.data.split.seed,
        split="train",
    )
    split_val = SplitConfig(
        train_frac=cfg.data.split.train_frac,
        val_frac=cfg.data.split.val_frac,
        seed=cfg.data.split.seed,
        split="val",
    )

    train_ds = (
        make_h5_ppg_dataset(
            sources,
            spec,
            batch_size=cfg.data.batch_size,
            shuffle_buffer=cfg.data.shuffle_buffer,
            seed=cfg.data.split.seed,
            split_cfg=split_train,
        )
        .map(_to_xy)
        .cache()
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        make_h5_ppg_dataset(
            sources,
            spec,
            batch_size=cfg.data.batch_size,
            shuffle_buffer=0,  # deterministic validation
            split_cfg=split_val,
        )
        .map(_to_xy)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, info


def build_model(cfg: PpgH5RvqConfig) -> keras.Model:
    """Build the RVQ autoencoder using the same builder as the legacy recipe."""
    mcfg = cfg.model
    win = cfg.data.window_samples
    _enc, _rvq, _dec, model = build_rvq_autoencoder(
        frame_size=win,
        embedding_dim=mcfg.embedding_dim,
        latent_width=mcfg.latent_width,
        base_filters=mcfg.base_filters,
        multiplier=mcfg.multiplier,
        num_levels=mcfg.num_levels,
        beta=mcfg.beta,
        num_stages=mcfg.num_stages,
        encoder_block_norm=mcfg.encoder_block_norm,
        encoder_head_norm=mcfg.encoder_head_norm,
        decoder_block_norm=mcfg.decoder_block_norm,
        decoder_head_norm=mcfg.decoder_head_norm,
        use_ema=mcfg.use_ema,
        ema_decay=mcfg.ema_decay,
        revive_dead_codes=mcfg.revive_dead_codes,
        revive_threshold=mcfg.revive_threshold,
        kmeans_init=mcfg.kmeans_init,
        codebook_sizes=mcfg.codebook_sizes,
    )
    return model


def compile_model(
    model: keras.Model,
    cfg: PpgH5RvqConfig,
    *,
    learning_rate: float | keras.optimizers.schedules.LearningRateSchedule,
) -> None:
    extra: list[callable] = []
    dloss = cfg.training.derivative_loss
    if dloss.enabled:
        extra.append(build_derivative_loss(dloss.weight))
    sloss = cfg.training.spectral_loss
    if sloss.enabled:
        extra.append(build_multi_scale_spectral_loss(sloss.weight, sloss.fft_sizes))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            keras.metrics.MeanSquaredError(name="mse"),
            keras.metrics.CosineSimilarity(name="cos", axis=-2),
        ],
        extra_losses=extra or None,
    )


def build_compression_stats(cfg: PpgH5RvqConfig) -> dict[str, Any]:
    mcfg = cfg.model
    downsample_factor = 2**mcfg.num_stages
    stats = compute_compression_stats(
        cfg.data.window_samples,
        bit_depth=cfg.evaluation.input_bit_depth,
        latent_width=mcfg.latent_width,
        num_levels=mcfg.num_levels,
        downsample_factor=downsample_factor,
        codebook_sizes=mcfg.codebook_sizes,
    )
    stats["effective_downsample_factor"] = downsample_factor
    return stats


# ---------------------------------------------------------------------------
# Recipe entry point
# ---------------------------------------------------------------------------


@recipe("train-ppg-h5-rvq", config_cls=PpgH5RvqConfig)
def train(cfg: PpgH5RvqConfig) -> dict[str, Any]:
    """Train an RVQ autoencoder on a mix of canonical h5 PPG datasets.

    Args:
        cfg: Validated pipeline configuration.

    Returns:
        Summary dict with metrics, compression stats, and artifact basenames.
    """
    run_dir = setup_run_dir(cfg.output.results_root, cfg.run_name)
    setup_logger(logger, run_dir, cfg.output.log_file)
    logger.info("Results directory: %s", run_dir)

    cfg_dump = cfg.model_dump()
    save_config_snapshot(cfg_dump, run_dir)

    train_ds, val_ds, ds_info = build_datasets(cfg)
    logger.info(
        "Dataset info: %s",
        json.dumps(
            {k: v for k, v in ds_info.items() if k != "audit"},
            indent=2,
        ),
    )

    model = build_model(cfg)
    lr_value = build_learning_rate(cfg, steps_per_epoch=cfg.data.steps_per_epoch)
    compile_model(model, cfg, learning_rate=lr_value)

    comp_stats = build_compression_stats(cfg)
    logger.info("Compression stats: %s", json.dumps(comp_stats, indent=2, default=str))

    callbacks = build_callbacks(cfg, run_dir=run_dir, lr_value=lr_value, wandb_run=None)
    history = model.fit(
        train_ds,
        steps_per_epoch=cfg.data.steps_per_epoch,
        epochs=cfg.data.epochs,
        validation_data=val_ds,
        validation_steps=cfg.training.validation_steps,
        verbose=2,
        callbacks=callbacks,
    )

    reload_best_weights(model, run_dir)
    artifacts = save_model_artifacts(model, run_dir)

    final_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    summary = {
        "run_name": cfg.run_name,
        "ds_info": ds_info,
        "compression_stats": comp_stats,
        "artifacts": artifacts,
        "best_checkpoint": BEST_CKPT_NAME,
        "history": final_history,
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote summary to %s", run_dir / "summary.json")
    return summary
