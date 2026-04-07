"""PPG RVQ training orchestration.

Wires together datasets, preprocessing, model build, training, evaluation,
and export into one coherent flow driven by a ``PpgRvqConfig``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf
from helia_edge.layers import ResidualVectorQuantizer

from compressionkit.configs.ppg_rvq import PpgRvqConfig
from compressionkit.datasets.ppg import (
    bandpass_filter_batch,
    build_ppg_tfrecord_cache,
    collect_random_samples,
    load_ppg_file_splits,
    load_ppg_splits,
    make_ppg_inmemory_dataset,
    make_ppg_stream_dataset,
    make_ppg_tfrecord_dataset,
)
from compressionkit.evaluation.artifacts import save_sample_artifacts
from compressionkit.evaluation.metrics import (
    TruePRD,
    compute_signal_metrics,
    summarize_physiokit_alignment,
)
from compressionkit.export.tflite import export_encoder_tflite
from compressionkit.logging.wandb_utils import (
    build_wandb_callbacks,
    finalize_wandb_run,
    init_wandb_run,
)
from compressionkit.models.rvq_autoencoder import (
    build_rvq_autoencoder,
    compute_compression_stats,
)
from compressionkit.preprocessing.ppg import (
    build_augmenter,
    build_preprocessor,
    generate_synthetic_ppg_batch,
)

logger = logging.getLogger("ppg-rvq-trainer")


# ---------------------------------------------------------------------------
# Learning rate builder
# ---------------------------------------------------------------------------

def build_learning_rate(
    cfg: PpgRvqConfig,
    *,
    steps_per_epoch: int,
) -> float | keras.optimizers.schedules.LearningRateSchedule:
    """Build optimizer learning rate or schedule from config."""
    lr_cfg = cfg.training.lr_schedule
    if not lr_cfg.enabled:
        return float(cfg.training.learning_rate)

    stype = lr_cfg.type.strip().lower()
    if stype != "cosine_restarts":
        raise ValueError(f"Unsupported lr_schedule.type: {stype}")

    first_decay_steps = lr_cfg.first_decay_steps
    if first_decay_steps is None:
        first_decay_steps = max(1, steps_per_epoch)
    else:
        first_decay_steps = max(1, first_decay_steps)

    return keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=float(cfg.training.learning_rate),
        first_decay_steps=first_decay_steps,
        t_mul=lr_cfg.t_mul,
        m_mul=lr_cfg.m_mul,
        alpha=lr_cfg.alpha,
    )


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_datasets(
    cfg: PpgRvqConfig,
    preprocessor: keras.layers.Layer,
    augmenter: keras.layers.Layer,
) -> tuple[tf.data.Dataset, tf.data.Dataset, int | None, dict[str, Any]]:
    """Build train and val tf.data datasets from config.

    Returns:
        ``(train_ds, val_ds, validation_steps, info_dict)``
    """
    data = cfg.data
    validation_steps = cfg.training.validation_steps
    info: dict[str, Any] = {
        "mode": "in-memory",
        "synthetic_added": 0,
        "synthetic_fraction_effective": 0.0,
        "cache_dir": None,
    }

    cache_cfg = data.cache
    streaming_cfg = data.streaming

    if cache_cfg.enabled and streaming_cfg.enabled:
        raise ValueError("Enable only one of data.cache.enabled or data.streaming.enabled.")

    input_filter_dict = data.input_filter.model_dump() if data.input_filter.enabled else {}
    target_filter_dict = data.target_filter.model_dump() if data.target_filter.enabled else {}

    if cache_cfg.enabled:
        info["mode"] = "cache"
        min_segment_scale = cache_cfg.min_segment_scale
        effective_segment_samples = max(
            data.segment_samples,
            int(np.ceil(float(data.frame_size) * max(1.0, min_segment_scale))),
        )
        cache_root = Path(cache_cfg.cache_root)
        if not cache_root.is_absolute():
            cache_root = cache_root.resolve()

        cache_dir, cache_meta = build_ppg_tfrecord_cache(
            datasets_dir=Path(data.datasets_dir),
            glob_pattern=data.dataset_glob,
            cache_root=cache_root,
            target_rate=data.sampling_rate,
            target_label=data.target_label,
            offset_samples=data.offset_samples,
            segment_samples=effective_segment_samples,
            frame_size=data.frame_size,
            shuffle_seed=data.shuffle_seed,
            train_ratio=cache_cfg.train_ratio,
            val_ratio=cache_cfg.val_ratio,
            windows_per_subject_train=cache_cfg.windows_per_subject_train,
            windows_per_subject_val=cache_cfg.windows_per_subject_val,
            force_rebuild=cache_cfg.force_rebuild,
        )
        info["cache_dir"] = str(cache_dir)
        train_path = cache_dir / str(cache_meta["train_tfrecord"])
        val_path = cache_dir / str(cache_meta["val_tfrecord"])
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"Missing TFRecord files in cache dir: {cache_dir}")

        train_ds = make_ppg_tfrecord_dataset(
            [train_path],
            frame_size=data.frame_size,
            segment_samples=effective_segment_samples,
            batch_size=data.batch_size,
            shuffle_buffer_size=data.buffer_size,
            preprocessor=preprocessor,
            augmenter=augmenter,
            input_filter_cfg=input_filter_dict if data.input_filter.enabled else None,
            target_filter_cfg=target_filter_dict if data.target_filter.enabled else None,
            sample_rate=data.sampling_rate,
            shuffle=True,
            seed=data.shuffle_seed,
        )
        val_ds = make_ppg_tfrecord_dataset(
            [val_path],
            frame_size=data.frame_size,
            segment_samples=effective_segment_samples,
            batch_size=data.batch_size,
            shuffle_buffer_size=data.buffer_size,
            preprocessor=preprocessor,
            augmenter=None,
            input_filter_cfg=input_filter_dict if data.input_filter.enabled else None,
            target_filter_cfg=target_filter_dict if data.target_filter.enabled else None,
            sample_rate=data.sampling_rate,
            shuffle=False,
            seed=data.shuffle_seed,
        )
        if validation_steps is None:
            val_examples = int(cache_meta.get("val_examples", 0))
            validation_steps = max(1, val_examples // data.batch_size)

    elif streaming_cfg.enabled:
        info["mode"] = "streaming"
        train_files, val_files, _ = load_ppg_file_splits(
            Path(data.datasets_dir),
            data.dataset_glob,
            seed=data.shuffle_seed,
        )
        logger.info("Streaming subject split: train=%d, val=%d", len(train_files), len(val_files))

        stream_kwargs: dict[str, Any] = dict(
            frame_size=data.frame_size,
            window_samples=data.segment_samples,
            batch_size=data.batch_size,
            interleave_cycle_length=streaming_cfg.interleave_cycle_length,
            target_rate=data.sampling_rate,
            target_label=data.target_label,
            offset_samples=data.offset_samples,
            preprocessor=preprocessor,
            input_filter_cfg=input_filter_dict if data.input_filter.enabled else None,
            target_filter_cfg=target_filter_dict if data.target_filter.enabled else None,
            seed=data.shuffle_seed,
        )
        train_ds = make_ppg_stream_dataset(
            train_files,
            subject_buffer_size=streaming_cfg.subject_buffer_size,
            window_buffer_size=streaming_cfg.window_buffer_size,
            windows_per_subject=streaming_cfg.windows_per_subject_train,
            augmenter=augmenter,
            shuffle=True,
            **stream_kwargs,
        )
        val_ds = make_ppg_stream_dataset(
            val_files,
            subject_buffer_size=max(32, len(val_files)),
            window_buffer_size=streaming_cfg.window_buffer_size,
            windows_per_subject=streaming_cfg.windows_per_subject_val,
            augmenter=None,
            shuffle=False,
            **stream_kwargs,
        )
        if validation_steps is None:
            val_windows = max(1, len(val_files) * max(1, streaming_cfg.windows_per_subject_val))
            validation_steps = max(1, val_windows // data.batch_size)

    else:
        # In-memory mode
        train_data, val_data, _ = load_ppg_splits(
            Path(data.datasets_dir),
            data.dataset_glob,
            seed=data.shuffle_seed,
            target_rate=data.sampling_rate,
            offset_samples=data.offset_samples,
            num_samples=data.segment_samples,
            target_label=data.target_label,
        )
        # Optional synthetic PPG mixing
        synth_cfg = data.synthetic_mix
        if synth_cfg.enabled:
            if not 0.0 < synth_cfg.fraction < 1.0:
                raise ValueError(f"synthetic_mix.fraction must be in (0, 1), got {synth_cfg.fraction}")
            n_real = train_data.shape[0]
            synthetic_added = int(np.ceil((synth_cfg.fraction * n_real) / (1.0 - synth_cfg.fraction)))
            synth_data = generate_synthetic_ppg_batch(
                num_segments=synthetic_added,
                signal_length=data.segment_samples,
                sample_rate=data.sampling_rate,
                heart_rate_bpm=synth_cfg.heart_rate_bpm,
                frequency_modulation=synth_cfg.frequency_modulation,
                ibi_randomness=synth_cfg.ibi_randomness,
                seed=synth_cfg.seed,
            )
            train_data = np.concatenate([train_data, synth_data], axis=0)
            info["synthetic_added"] = synthetic_added
            info["synthetic_fraction_effective"] = float(synthetic_added / max(train_data.shape[0], 1))

        train_data = train_data[:, :, np.newaxis]
        val_data = val_data[:, :, np.newaxis]

        # Optional filtering
        train_target_data = None
        val_target_data = None
        if data.input_filter.enabled:
            train_data = bandpass_filter_batch(
                train_data, sample_rate=data.sampling_rate,
                low_hz=data.input_filter.low_hz, high_hz=data.input_filter.high_hz,
                order=data.input_filter.order,
            )
            val_data = bandpass_filter_batch(
                val_data, sample_rate=data.sampling_rate,
                low_hz=data.input_filter.low_hz, high_hz=data.input_filter.high_hz,
                order=data.input_filter.order,
            )
        if data.target_filter.enabled:
            same_as_input = (
                data.input_filter.enabled
                and abs(data.target_filter.low_hz - data.input_filter.low_hz) < 1e-12
                and abs(data.target_filter.high_hz - data.input_filter.high_hz) < 1e-12
                and data.target_filter.order == data.input_filter.order
            )
            if same_as_input:
                train_target_data = train_data.copy()
                val_target_data = val_data.copy()
            else:
                raw_train = train_data.copy()
                raw_val = val_data.copy()
                train_target_data = bandpass_filter_batch(
                    raw_train, sample_rate=data.sampling_rate,
                    low_hz=data.target_filter.low_hz, high_hz=data.target_filter.high_hz,
                    order=data.target_filter.order,
                )
                val_target_data = bandpass_filter_batch(
                    raw_val, sample_rate=data.sampling_rate,
                    low_hz=data.target_filter.low_hz, high_hz=data.target_filter.high_hz,
                    order=data.target_filter.order,
                )

        train_ds = make_ppg_inmemory_dataset(
            train_data, frame_size=data.frame_size, batch_size=data.batch_size,
            buffer_size=data.buffer_size, preprocessor=preprocessor,
            augmenter=augmenter, target_data=train_target_data, shuffle=True,
        )
        val_ds = make_ppg_inmemory_dataset(
            val_data, frame_size=data.frame_size, batch_size=data.batch_size,
            buffer_size=data.buffer_size, preprocessor=preprocessor,
            augmenter=augmenter, target_data=val_target_data, shuffle=False,
        )

    return train_ds, val_ds, validation_steps, info


# ---------------------------------------------------------------------------
# Callback builder
# ---------------------------------------------------------------------------

def build_callbacks(
    cfg: PpgRvqConfig,
    *,
    run_dir: Path,
    lr_value: float | keras.optimizers.schedules.LearningRateSchedule,
    wandb_run: Any,
) -> list[keras.callbacks.Callback]:
    """Build the list of Keras training callbacks from config."""
    tcfg = cfg.training
    selection_monitor = tcfg.selection_metric if tcfg.selection_metric.startswith("val_") else f"val_{tcfg.selection_metric}"
    best_ckpt_path = run_dir / "best_model.weights.h5"

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_ckpt_path,
            monitor=selection_monitor,
            mode=tcfg.val_mode,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=f"val_{tcfg.val_metric}",
            patience=tcfg.early_stop_patience,
            mode=tcfg.val_mode,
            restore_best_weights=True,
        ),
        keras.callbacks.CSVLogger(run_dir / f"training_history_{cfg.run_name}.csv"),
    ]

    using_schedule = isinstance(lr_value, keras.optimizers.schedules.LearningRateSchedule)
    if tcfg.reduce_lr_on_plateau and not using_schedule:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=f"val_{tcfg.val_metric}",
                factor=tcfg.reduce_lr_factor,
                patience=tcfg.reduce_lr_patience,
                mode=tcfg.val_mode,
                min_lr=tcfg.reduce_lr_min_lr,
                verbose=1,
            )
        )

    if cfg.output.tensorboard:
        tb_dir = run_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=tb_dir, write_graph=False, write_images=False, update_freq="epoch",
            )
        )

    callbacks.extend(
        build_wandb_callbacks(run=wandb_run, log_model=cfg.output.wandb.log_model)
    )
    return callbacks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    cfg: PpgRvqConfig,
    *,
    model: keras.Model,
    val_ds: tf.data.Dataset,
    run_dir: Path,
    validation_steps: int | None,
) -> dict[str, Any]:
    """Run post-training evaluation: sample reconstruction, band metrics, physiokit."""
    data = cfg.data
    eval_cfg = cfg.evaluation
    rng = np.random.default_rng(data.shuffle_seed)
    sample_inputs, sample_targets = collect_random_samples(val_ds, eval_cfg.num_samples, rng)
    reconstructions = model.predict(sample_inputs, verbose=0)

    band_cfg = eval_cfg.band_metrics
    physio_cfg = eval_cfg.physiokit_metrics

    band_sample_targets = None
    band_reconstructions = None
    best_band_metrics = None
    best_physiokit_metrics = None
    best_physio_per_sample: list[dict[str, Any] | None] = []

    if band_cfg.enabled:
        targets_seq = sample_targets.reshape(sample_targets.shape[0], -1).astype(np.float32)
        recon_seq = reconstructions.reshape(reconstructions.shape[0], -1).astype(np.float32)
        band_sample_targets = bandpass_filter_batch(
            targets_seq, sample_rate=data.sampling_rate,
            low_hz=band_cfg.low_hz, high_hz=band_cfg.high_hz, order=band_cfg.order,
        )
        band_reconstructions = bandpass_filter_batch(
            recon_seq, sample_rate=data.sampling_rate,
            low_hz=band_cfg.low_hz, high_hz=band_cfg.high_hz, order=band_cfg.order,
        )
        best_band_metrics = compute_signal_metrics(band_sample_targets, band_reconstructions)

    if physio_cfg.enabled:
        targets_seq = sample_targets.reshape(sample_targets.shape[0], -1).astype(np.float32)
        recon_seq = reconstructions.reshape(reconstructions.shape[0], -1).astype(np.float32)
        best_physiokit_metrics, best_physio_per_sample = summarize_physiokit_alignment(
            targets_seq, recon_seq,
            sample_rate=data.sampling_rate,
            low_hz=physio_cfg.low_hz, high_hz=physio_cfg.high_hz,
            order=physio_cfg.order, min_peaks=physio_cfg.min_peaks,
        )

    sample_results = {
        str(idx): save_sample_artifacts(
            idx, target.squeeze(), recon.squeeze(), data.sampling_rate, run_dir,
            band_original=None if band_sample_targets is None else band_sample_targets[idx],
            band_reconstructed=None if band_reconstructions is None else band_reconstructions[idx],
            physiokit_metrics=None if not best_physio_per_sample else best_physio_per_sample[idx],
        )
        for idx, (target, recon) in enumerate(zip(sample_targets, reconstructions))
    }

    return {
        "samples": sample_results,
        "sample_inputs": sample_inputs,
        "band_metrics": best_band_metrics,
        "physiokit_metrics": best_physiokit_metrics,
    }


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------

def train(cfg: PpgRvqConfig) -> dict[str, Any]:
    """Run the full PPG RVQ training pipeline.

    Args:
        cfg: Validated pipeline configuration.

    Returns:
        Summary dictionary with metrics, compression stats, and artifact paths.
    """
    # Setup output directory
    results_root = Path(cfg.output.results_root).resolve()
    run_dir = results_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _setup_logger(run_dir, cfg.output.log_file)
    logger.info("Results directory: %s", run_dir)

    # Save config
    cfg_dump = cfg.model_dump()
    with (run_dir / "config.json").open("w") as f:
        json.dump(cfg_dump, f, indent=2)

    wandb_run = init_wandb_run(cfg=cfg_dump, run_name=cfg.run_name, run_dir=run_dir)

    # Build preprocessing
    data = cfg.data
    preprocessor = build_preprocessor(frame_size=data.frame_size, epsilon=data.epsilon)
    augmenter = build_augmenter(tuple(data.gaussian_noise))

    # Build datasets
    train_ds, val_ds, validation_steps, ds_info = build_datasets(cfg, preprocessor, augmenter)
    logger.info("Dataset mode: %s", ds_info["mode"])

    # Build model
    mcfg = cfg.model
    enc, rvq, dec, model = build_rvq_autoencoder(
        frame_size=data.frame_size,
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
    )

    # Compile
    lr_value = build_learning_rate(cfg, steps_per_epoch=data.steps_per_epoch)
    optimizer = keras.optimizers.Adam(lr_value)
    metrics = [
        keras.metrics.MeanSquaredError(name="mse"),
        keras.metrics.CosineSimilarity(name="cos", axis=-2),
        TruePRD(name="prd"),
    ]
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError(), metrics=metrics)

    # Callbacks
    callbacks = build_callbacks(cfg, run_dir=run_dir, lr_value=lr_value, wandb_run=wandb_run)

    # Train
    history = model.fit(
        train_ds,
        steps_per_epoch=data.steps_per_epoch,
        epochs=data.epochs,
        validation_data=val_ds,
        validation_steps=validation_steps,
        verbose=2,
        callbacks=callbacks,
    )

    # Reload best weights onto the model.
    # EarlyStopping(restore_best_weights=True) already restores in-memory,
    # but we also explicitly load the checkpoint for safety.
    best_ckpt_path = run_dir / "best_model.weights.h5"
    if best_ckpt_path.exists():
        model.load_weights(best_ckpt_path)
    else:
        logger.warning("Best checkpoint not found; using current model state.")

    best_model = model

    # Save artifacts — encoder/decoder are standard Functional models (serializable).
    best_enc = best_model.encoder
    best_dec = best_model.decoder
    best_rvq = best_model.vq
    best_enc.save(run_dir / "encoder.keras")
    best_dec.save(run_dir / "decoder.keras")
    model.save_weights(run_dir / "model.weights.h5")
    rvq_weights = best_rvq.get_weights()
    np.savez(run_dir / "rvq_weights.npz", *rvq_weights)

    # Evaluation
    eval_results = run_evaluation(
        cfg, model=best_model, val_ds=val_ds, run_dir=run_dir,
        validation_steps=validation_steps,
    )

    # Metrics summary
    history_dict = history.history
    selection_metric = cfg.training.selection_metric
    if selection_metric not in history_dict:
        logger.warning("Selection metric '%s' not in history, falling back to 'val_loss'.", selection_metric)
        selection_metric = "val_loss"
    metric_series = np.asarray(history_dict[selection_metric], dtype=np.float64)
    best_idx = int(np.argmin(metric_series))
    best_epoch = best_idx + 1

    best_metrics = {
        key: float(values[best_idx])
        for key, values in history_dict.items()
        if isinstance(values, list) and len(values) > best_idx
    }
    final_metrics = {
        "final_loss": float(history_dict["loss"][-1]),
        "final_val_loss": float(history_dict["val_loss"][-1]),
        "final_val_mse": float(history_dict.get("val_mse", [0])[-1]),
    }

    downsample_factor = 2 ** mcfg.num_stages
    compression = compute_compression_stats(
        data.frame_size, bit_depth=cfg.evaluation.input_bit_depth,
        latent_width=mcfg.latent_width, num_levels=mcfg.num_levels,
        downsample_factor=downsample_factor,
    )
    compression["configured_num_stages"] = mcfg.num_stages
    compression["effective_downsample_factor"] = downsample_factor

    # TFLite export
    rep_batches_limit = max(1, cfg.evaluation.tflite_rep_batches)
    rep_batches: list[np.ndarray] = []
    for batch, _ in val_ds.take(rep_batches_limit):
        rep_batches.append(batch.numpy())
    rep_dataset = np.concatenate(rep_batches, axis=0) if rep_batches else eval_results["sample_inputs"].astype(np.float32)
    export_encoder_tflite(best_enc, rep_dataset=rep_dataset, output_dir=run_dir)

    # Build summary
    summary: dict[str, Any] = {
        "config": cfg_dump,
        "metrics": {
            "history": history_dict,
            "selection_metric": selection_metric,
            "best_epoch": best_epoch,
            "best": best_metrics,
            "final": final_metrics,
            "dataset_mode": ds_info["mode"],
            "synthetic_mix_num_added": ds_info["synthetic_added"],
            "synthetic_mix_fraction_effective": ds_info["synthetic_fraction_effective"],
            "cache_dir": ds_info["cache_dir"],
            "validation_steps": validation_steps,
        },
        "compression": compression,
        "samples": eval_results["samples"],
        "artifacts": {
            "model": "model.keras",
            "encoder": "encoder.keras",
            "decoder": "decoder.keras",
            "best_model": best_ckpt_path.name,
            "rvq_weights": "rvq_weights.npz",
            "encoder_tflite": "encoder.tflite",
            "encoder_header": "encoder.h",
        },
    }

    # Add band / physiokit metrics to summary
    band_cfg = cfg.evaluation.band_metrics
    if band_cfg.enabled and eval_results["band_metrics"] is not None:
        bm = eval_results["band_metrics"]
        summary["metrics"]["band_metrics_config"] = band_cfg.model_dump()
        summary["metrics"]["best_band"] = {
            "band_mse": bm["mse"],
            "band_mae": bm["mae"],
            "band_prd_percent": bm["prd_percent"],
            "band_cosine_similarity": bm["cosine_similarity"],
        }
    physio_cfg = cfg.evaluation.physiokit_metrics
    if physio_cfg.enabled and eval_results["physiokit_metrics"] is not None:
        summary["metrics"]["physiokit_metrics_config"] = physio_cfg.model_dump()
        summary["metrics"]["best_physiokit"] = eval_results["physiokit_metrics"]

    summary_path = run_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Best epoch by %s: %d (value=%.6f)",
        selection_metric, best_epoch, best_metrics[selection_metric],
    )

    # Finalize W&B
    finalize_wandb_run(
        run=wandb_run, summary=summary, run_dir=run_dir,
        artifact_summary_only=cfg.output.wandb.artifact_summary_only,
    )

    return summary


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def _setup_logger(run_dir: Path, log_file: str | None) -> None:
    """Configure file and stream handlers for the module logger."""
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = run_dir / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)


__all__ = ["build_datasets", "build_learning_rate", "train"]
