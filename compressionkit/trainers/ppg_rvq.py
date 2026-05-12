"""PPG RVQ training orchestration.

Wires together datasets, preprocessing, model build, training, evaluation,
and export into one coherent flow driven by a ``PpgRvqConfig``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from compressionkit.configs.ppg_rvq import PpgRvqConfig, TransformConfig
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
from compressionkit.evaluation.overlap_add import evaluate_long_recordings
from compressionkit.losses import (
    build_derivative_loss as _build_derivative_loss,
)
from compressionkit.losses import (
    build_multi_scale_spectral_loss as _build_multi_scale_spectral_loss,
)
from compressionkit.models.rvq_autoencoder import (
    build_rvq_autoencoder,
    compute_compression_stats,
)
from compressionkit.preprocessing.augmentations import (
    PPGAugmenter,
    build_noise_bank_from_h5,
)
from compressionkit.preprocessing.ppg import (
    generate_synthetic_ppg_batch,
)
from compressionkit.trainers.utils import (
    build_learning_rate,
)

logger = logging.getLogger("ppg-rvq-trainer")


# ---------------------------------------------------------------------------
# DWT transform wrapping
# ---------------------------------------------------------------------------


def _wrap_dataset_with_dwt(
    ds: tf.data.Dataset,
    transform_cfg: TransformConfig,
    frame_size: int,
) -> tf.data.Dataset:
    """Apply DWT forward transform to both input and target tensors.

    Input/output shape: ``(B, 1, frame_size, 1)`` — same shape, packed
    wavelet coefficients replace raw time-domain samples.
    """
    domain = transform_cfg.domain.strip().lower()
    if domain == "raw":
        return ds

    if domain != "dwt":
        raise ValueError(f"PPG RVQ only supports 'raw' or 'dwt' domain, got: {domain!r}")

    from compressionkit.dsp.transforms import DwtConfig, dwt_pack

    dwt_cfg = DwtConfig(levels=transform_cfg.dwt_levels, wavelet=transform_cfg.dwt_wavelet)

    def _dwt_forward_batch(x: np.ndarray) -> np.ndarray:
        B = x.shape[0]
        out = np.empty_like(x)
        for i in range(B):
            out[i, 0, :, 0] = dwt_pack(x[i, 0, :, 0], dwt_cfg)
        return out

    def _apply_dwt(inp, target):
        inp_t = tf.numpy_function(_dwt_forward_batch, [inp], tf.float32)
        tgt_t = tf.numpy_function(_dwt_forward_batch, [target], tf.float32)
        inp_t.set_shape(inp.shape)
        tgt_t.set_shape(target.shape)
        return inp_t, tgt_t

    return ds.map(_apply_dwt, num_parallel_calls=tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Tier-1 augmentation wrapping (input-only for denoising)
# ---------------------------------------------------------------------------


def _build_ppg_augmenter(cfg: PpgRvqConfig) -> PPGAugmenter | None:
    """Build Tier-1 augmenter from config, including noise bank."""
    aug_cfg = cfg.data.augmentation
    if not aug_cfg.enabled:
        return None

    # Build noise bank from configured sources
    noise_bank = None
    if aug_cfg.noise_bank_sources:
        import glob as glob_mod

        h5_paths: list[str] = []
        for slug in aug_cfg.noise_bank_sources:
            pattern = f"{aug_cfg.noise_bank_root}/{slug}/*.h5"
            h5_paths.extend(sorted(glob_mod.glob(pattern)))

        if h5_paths:
            logger.info("Building noise bank from %d h5 files...", len(h5_paths))
            noise_bank = build_noise_bank_from_h5(
                h5_paths,
                target_fs=cfg.data.sampling_rate,
                window_size=cfg.data.frame_size,
                max_segments=aug_cfg.noise_bank_max_segments,
            )
            logger.info("Noise bank: %d segments", len(noise_bank))

    return PPGAugmenter(
        sample_rate=cfg.data.sampling_rate,
        baseline_wander_prob=aug_cfg.baseline_wander_prob,
        motion_artifact_prob=aug_cfg.motion_artifact_prob,
        motion_snr_range=aug_cfg.motion_snr_range,
        beat_scale_prob=aug_cfg.beat_scale_prob,
        time_warp_prob=aug_cfg.time_warp_prob,
        empirical_noise_prob=aug_cfg.empirical_noise_prob,
        empirical_snr_range=aug_cfg.empirical_snr_range,
        noise_bank=noise_bank,
        seed=cfg.data.shuffle_seed,
    )


def _wrap_dataset_with_augmentation(
    ds: tf.data.Dataset,
    augmenter: PPGAugmenter,
    frame_size: int,
) -> tf.data.Dataset:
    """Apply Tier-1 augmentation to input only (target stays clean).

    This creates a denoising objective: model learns to reconstruct clean
    signal from corrupted input.
    """

    def _augment_input_batch(inp: np.ndarray) -> np.ndarray:
        B = inp.shape[0]
        out = np.empty_like(inp)
        for i in range(B):
            signal = inp[i, 0, :, 0]
            out[i, 0, :, 0] = augmenter.augment(signal)
        return out

    def _apply_augmentation(inp, target):
        aug_inp = tf.numpy_function(_augment_input_batch, [inp], tf.float32)
        aug_inp.set_shape(inp.shape)
        return aug_inp, target

    return ds.map(_apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)


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
    unified_cfg = data.unified_cache

    enabled_modes = sum([cache_cfg.enabled, streaming_cfg.enabled, unified_cfg.enabled])
    if enabled_modes > 1:
        raise ValueError("Enable at most one of data.cache, data.streaming, or data.unified_cache.")

    input_filter_dict = data.input_filter.model_dump() if data.input_filter.enabled else {}
    target_filter_dict = data.target_filter.model_dump() if data.target_filter.enabled else {}

    if unified_cfg.enabled:
        # ---- Unified multi-source TFRecord cache mode ----
        from compressionkit.datasets.ppg_cache import (
            SourceWeight,
            load_cache_metadata,
            make_cached_ppg_dataset,
        )

        info["mode"] = "unified_cache"
        source_weights = [SourceWeight(slug=s.slug, weight=s.weight) for s in unified_cfg.sources]
        cache_root = Path(unified_cfg.cache_root)
        if not cache_root.is_absolute():
            cache_root = cache_root.resolve()

        train_ds, train_info = make_cached_ppg_dataset(
            source_weights,
            cache_root=cache_root,
            frame_size=data.frame_size,
            batch_size=data.batch_size,
            shuffle_buffer=unified_cfg.shuffle_buffer,
            epsilon=data.epsilon,
            split="train",
            seed=data.shuffle_seed,
        )
        val_ds, val_info = make_cached_ppg_dataset(
            source_weights,
            cache_root=cache_root,
            frame_size=data.frame_size,
            batch_size=data.batch_size,
            shuffle_buffer=0,
            epsilon=data.epsilon,
            split="val",
            seed=data.shuffle_seed,
        )
        info["unified_sources"] = train_info["sources"]

        if validation_steps is None:
            total_val = sum(load_cache_metadata(cache_root, s.slug)["val_examples"] for s in unified_cfg.sources)
            validation_steps = max(1, total_val // data.batch_size)

    elif cache_cfg.enabled:
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

        stream_kwargs: dict[str, Any] = {
            "frame_size": data.frame_size,
            "window_samples": data.segment_samples,
            "batch_size": data.batch_size,
            "interleave_cycle_length": streaming_cfg.interleave_cycle_length,
            "target_rate": data.sampling_rate,
            "target_label": data.target_label,
            "offset_samples": data.offset_samples,
            "preprocessor": preprocessor,
            "input_filter_cfg": input_filter_dict if data.input_filter.enabled else None,
            "target_filter_cfg": target_filter_dict if data.target_filter.enabled else None,
            "seed": data.shuffle_seed,
        }
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
                train_data,
                sample_rate=data.sampling_rate,
                low_hz=data.input_filter.low_hz,
                high_hz=data.input_filter.high_hz,
                order=data.input_filter.order,
            )
            val_data = bandpass_filter_batch(
                val_data,
                sample_rate=data.sampling_rate,
                low_hz=data.input_filter.low_hz,
                high_hz=data.input_filter.high_hz,
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
                    raw_train,
                    sample_rate=data.sampling_rate,
                    low_hz=data.target_filter.low_hz,
                    high_hz=data.target_filter.high_hz,
                    order=data.target_filter.order,
                )
                val_target_data = bandpass_filter_batch(
                    raw_val,
                    sample_rate=data.sampling_rate,
                    low_hz=data.target_filter.low_hz,
                    high_hz=data.target_filter.high_hz,
                    order=data.target_filter.order,
                )

        train_ds = make_ppg_inmemory_dataset(
            train_data,
            frame_size=data.frame_size,
            batch_size=data.batch_size,
            buffer_size=data.buffer_size,
            preprocessor=preprocessor,
            augmenter=augmenter,
            target_data=train_target_data,
            shuffle=True,
        )
        val_ds = make_ppg_inmemory_dataset(
            val_data,
            frame_size=data.frame_size,
            batch_size=data.batch_size,
            buffer_size=data.buffer_size,
            preprocessor=preprocessor,
            augmenter=augmenter,
            target_data=val_target_data,
            shuffle=False,
        )

    # Apply Tier-1 augmentation (input-only corruption for denoising)
    tier1_augmenter = _build_ppg_augmenter(cfg)
    if tier1_augmenter is not None:
        logger.info("Applying Tier-1 augmentations to training inputs (denoising mode)")
        train_ds = _wrap_dataset_with_augmentation(train_ds, tier1_augmenter, data.frame_size)

    # Apply DWT transform if configured
    transform_cfg = data.transform
    if transform_cfg.domain.strip().lower() != "raw":
        logger.info(
            "Applying %s transform (levels=%d, wavelet=%s)",
            transform_cfg.domain,
            transform_cfg.dwt_levels,
            transform_cfg.dwt_wavelet,
        )
        train_ds = _wrap_dataset_with_dwt(train_ds, transform_cfg, data.frame_size)
        val_ds = _wrap_dataset_with_dwt(val_ds, transform_cfg, data.frame_size)

    return train_ds, val_ds, validation_steps, info


# NOTE: build_callbacks is imported from compressionkit.trainers.utils


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

    # If DWT domain, inverse-transform for time-domain metrics
    transform_cfg = data.transform
    if transform_cfg.domain.strip().lower() == "dwt":
        from compressionkit.dsp.transforms import DwtConfig, dwt_unpack

        dwt_cfg = DwtConfig(levels=transform_cfg.dwt_levels, wavelet=transform_cfg.dwt_wavelet)
        N = sample_targets.shape[0]
        for i in range(N):
            sample_targets[i, 0, :, 0] = dwt_unpack(sample_targets[i, 0, :, 0], dwt_cfg, data.frame_size)
            reconstructions[i, 0, :, 0] = dwt_unpack(reconstructions[i, 0, :, 0], dwt_cfg, data.frame_size)

    band_cfg = eval_cfg.band_metrics
    physio_cfg = eval_cfg.physiokit_metrics
    long_cfg = eval_cfg.long_recording

    band_sample_targets = None
    band_reconstructions = None
    best_band_metrics = None
    best_physiokit_metrics = None
    best_physio_per_sample: list[dict[str, Any] | None] = []
    best_long_recording_metrics = None
    best_long_per_recording: list[dict[str, Any] | None] = []

    if band_cfg.enabled:
        targets_seq = sample_targets.reshape(sample_targets.shape[0], -1).astype(np.float32)
        recon_seq = reconstructions.reshape(reconstructions.shape[0], -1).astype(np.float32)
        band_sample_targets = bandpass_filter_batch(
            targets_seq,
            sample_rate=data.sampling_rate,
            low_hz=band_cfg.low_hz,
            high_hz=band_cfg.high_hz,
            order=band_cfg.order,
        )
        band_reconstructions = bandpass_filter_batch(
            recon_seq,
            sample_rate=data.sampling_rate,
            low_hz=band_cfg.low_hz,
            high_hz=band_cfg.high_hz,
            order=band_cfg.order,
        )
        best_band_metrics = compute_signal_metrics(band_sample_targets, band_reconstructions)

    if physio_cfg.enabled:
        targets_seq = sample_targets.reshape(sample_targets.shape[0], -1).astype(np.float32)
        recon_seq = reconstructions.reshape(reconstructions.shape[0], -1).astype(np.float32)
        best_physiokit_metrics, best_physio_per_sample = summarize_physiokit_alignment(
            targets_seq,
            recon_seq,
            sample_rate=data.sampling_rate,
            low_hz=physio_cfg.low_hz,
            high_hz=physio_cfg.high_hz,
            order=physio_cfg.order,
            min_peaks=physio_cfg.min_peaks,
        )

    if long_cfg.enabled and physio_cfg.enabled:
        logger.info(
            "Running long-recording overlap-add evaluation (%.0fs, %d recordings, hop=%.0f%%)...",
            long_cfg.duration_sec,
            long_cfg.num_recordings,
            long_cfg.hop_ratio * 100,
        )
        best_long_recording_metrics, best_long_per_recording = evaluate_long_recordings(
            model,
            datasets_dir=Path(data.datasets_dir),
            dataset_glob=data.dataset_glob,
            frame_size=data.frame_size,
            sample_rate=data.sampling_rate,
            target_label=data.target_label,
            offset_samples=data.offset_samples,
            duration_sec=long_cfg.duration_sec,
            epsilon=data.epsilon,
            hop_ratio=long_cfg.hop_ratio,
            num_recordings=long_cfg.num_recordings,
            physiokit_low_hz=physio_cfg.low_hz,
            physiokit_high_hz=physio_cfg.high_hz,
            physiokit_order=physio_cfg.order,
            physiokit_min_peaks=physio_cfg.min_peaks,
            batch_size=long_cfg.batch_size,
            seed=data.shuffle_seed,
        )

    plot_cap = max(0, min(eval_cfg.num_plot_samples, len(sample_targets)))
    sample_results = {
        str(idx): save_sample_artifacts(
            idx,
            target.squeeze(),
            recon.squeeze(),
            data.sampling_rate,
            run_dir,
            band_original=None if band_sample_targets is None else band_sample_targets[idx],
            band_reconstructed=None if band_reconstructions is None else band_reconstructions[idx],
            physiokit_metrics=None if not best_physio_per_sample else best_physio_per_sample[idx],
            save_plot=(idx < plot_cap),
        )
        for idx, (target, recon) in enumerate(zip(sample_targets, reconstructions))
    }

    return {
        "samples": sample_results,
        "sample_inputs": sample_inputs,
        "sample_targets": sample_targets,
        "sample_reconstructions": reconstructions,
        "band_metrics": best_band_metrics,
        "physiokit_metrics": best_physiokit_metrics,
        "long_recording_metrics": best_long_recording_metrics,
        "long_recording_per_recording": best_long_per_recording,
    }


# ---------------------------------------------------------------------------
# Model build & compile
# ---------------------------------------------------------------------------


def build_model(cfg: PpgRvqConfig) -> keras.Model:
    """Build the RVQ autoencoder from the model section of *cfg*.

    Returns the trainable model; encoder / decoder / RVQ are accessible as
    ``model.encoder`` / ``model.decoder`` / ``model.vq``.
    """
    data = cfg.data
    mcfg = cfg.model
    _enc, _rvq, _dec, model = build_rvq_autoencoder(
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
        use_ema=mcfg.use_ema,
        ema_decay=mcfg.ema_decay,
        revive_dead_codes=mcfg.revive_dead_codes,
        revive_threshold=mcfg.revive_threshold,
        codebook_sizes=mcfg.codebook_sizes,
    )
    return model


def build_extra_losses(cfg: PpgRvqConfig) -> list[callable]:
    """Assemble the optional auxiliary losses enabled in *cfg*."""
    extra: list[callable] = []

    dloss = cfg.training.derivative_loss
    if dloss.enabled:
        extra.append(_build_derivative_loss(dloss.weight))
        logger.info("Derivative loss enabled, weight=%.4f", dloss.weight)

    sloss = cfg.training.spectral_loss
    if sloss.enabled:
        extra.append(_build_multi_scale_spectral_loss(weight=sloss.weight, fft_sizes=sloss.fft_sizes))
        logger.info(
            "Multi-scale spectral loss enabled, weight=%.2f, fft_sizes=%s",
            sloss.weight,
            sloss.fft_sizes,
        )
    return extra


def compile_model(
    model: keras.Model,
    cfg: PpgRvqConfig,
    *,
    learning_rate: float | keras.optimizers.schedules.LearningRateSchedule,
) -> None:
    """Compile *model* with the optimizer, metrics and losses implied by *cfg*."""
    extra = build_extra_losses(cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            keras.metrics.MeanSquaredError(name="mse"),
            keras.metrics.CosineSimilarity(name="cos", axis=-2),
            TruePRD(name="prd"),
        ],
        extra_losses=extra or None,
    )


# ---------------------------------------------------------------------------
# Summary assembly
# ---------------------------------------------------------------------------


def build_compression_stats(cfg: PpgRvqConfig) -> dict[str, Any]:
    """Compute the compression ratio / bit budget for *cfg*."""
    data = cfg.data
    mcfg = cfg.model
    downsample_factor = 2**mcfg.num_stages
    stats = compute_compression_stats(
        data.frame_size,
        bit_depth=cfg.evaluation.input_bit_depth,
        latent_width=mcfg.latent_width,
        num_levels=mcfg.num_levels,
        downsample_factor=downsample_factor,
        codebook_sizes=mcfg.codebook_sizes,
    )
    stats["configured_num_stages"] = mcfg.num_stages
    stats["effective_downsample_factor"] = downsample_factor
    return stats


def assemble_summary(
    cfg: PpgRvqConfig,
    *,
    cfg_dump: dict[str, Any],
    history: dict[str, list[float]],
    eval_results: dict[str, Any],
    ds_info: dict[str, Any],
    compression: dict[str, Any],
    deploy_artifacts: dict[str, Any],
    model_artifacts: dict[str, str],
    best_epoch: int,
    best_metrics: dict[str, float],
    final_metrics: dict[str, float],
    selection_metric: str,
    validation_steps: int | None,
) -> dict[str, Any]:
    """Build the run-summary dictionary that gets serialized to ``summary.json``."""
    summary: dict[str, Any] = {
        "config": cfg_dump,
        "metrics": {
            "history": history,
            "selection_metric": selection_metric,
            "best_epoch": best_epoch,
            "best": best_metrics,
            "final": final_metrics,
            "dataset_mode": ds_info["mode"],
            "synthetic_mix_num_added": ds_info.get("synthetic_added", 0),
            "synthetic_mix_fraction_effective": ds_info.get("synthetic_fraction_effective", 0.0),
            "cache_dir": ds_info.get("cache_dir"),
            "validation_steps": validation_steps,
        },
        "compression": compression,
        "samples": eval_results["samples"],
        "artifacts": {**model_artifacts, "deploy": deploy_artifacts},
    }

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

    long_cfg = cfg.evaluation.long_recording
    if long_cfg.enabled and eval_results["long_recording_metrics"] is not None:
        summary["metrics"]["long_recording_config"] = long_cfg.model_dump()
        summary["metrics"]["long_recording_physiokit"] = eval_results["long_recording_metrics"]

    return summary


__all__ = [
    "assemble_summary",
    "build_compression_stats",
    "build_datasets",
    "build_extra_losses",
    "build_learning_rate",
    "build_model",
    "compile_model",
    "run_evaluation",
]
