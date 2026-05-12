"""Two-stream PPG codec trainer.

Orchestrates:
1. Data loading and baseline/pulsatile decomposition.
2. Independent training of both stream codecs.
3. Comprehensive evaluation: primary metrics, HR/HRV, spectral, stitching.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from compressionkit.configs.ppg_two_stream import PpgTwoStreamConfig
from compressionkit.datasets.ppg import (
    build_ppg_tfrecord_cache,
    load_ppg_file_splits,
    load_ppg_signal,
    load_ppg_splits,
)
from compressionkit.evaluation.metrics import (
    compute_signal_metrics,
    summarize_physiokit_alignment,
)
from compressionkit.evaluation.spectral_metrics import psd_band_error
from compressionkit.evaluation.stitching import seam_discontinuity_ratio, stitch
from compressionkit.losses import (
    build_derivative_loss,
)
from compressionkit.losses import (
    build_filtered_mse_loss as _build_filtered_mse_loss,
)
from compressionkit.models.ppg_two_stream import (
    build_baseline_model,
    build_pulsatile_model,
    compute_combined_compression_stats,
)
from compressionkit.preprocessing.two_stream import (
    decompose_and_normalize,
    downsample_baseline,
    reconstruct_from_streams,
    upsample_baseline,
)
from compressionkit.trainers.utils import build_learning_rate

logger = logging.getLogger("ppg-two-stream-trainer")


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def _build_extra_losses(cfg: PpgTwoStreamConfig) -> list[callable]:
    """Assemble auxiliary losses enabled in *cfg*."""
    extra: list[callable] = []
    training = cfg.training

    dloss = training.derivative_loss
    if dloss.enabled:
        extra.append(build_derivative_loss(dloss.weight))
        logger.info("Derivative loss enabled, weight=%.4f", dloss.weight)

    floss = training.filtered_loss
    if floss.enabled:
        extra.append(
            _build_filtered_mse_loss(
                weight=floss.weight,
                sample_rate=cfg.data.sampling_rate,
                cutoff_hz=floss.cutoff_hz,
                num_taps=floss.num_taps,
            )
        )
        logger.info(
            "Filtered MSE loss enabled, weight=%.2f, cutoff=%.1f Hz",
            floss.weight, floss.cutoff_hz,
        )

    return extra


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _load_raw_segments(cfg: PpgTwoStreamConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load raw PPG segments (train, val) cropped to frame_size.

    When ``unified_cache_enabled`` is True, loads from per-source caches.
    When ``cache_enabled`` is True, uses the TFRecord cache (same as golden)
    to provide hundreds of diverse windows per subject.  Otherwise falls back
    to single-segment-per-subject loading.

    Returns arrays of shape ``(N, frame_size)``.
    """
    data = cfg.data

    if data.unified_cache_enabled:
        return _load_raw_segments_from_unified_cache(cfg)

    if data.cache_enabled:
        return _load_raw_segments_from_cache(cfg)

    train_data, val_data, _ = load_ppg_splits(
        Path(data.datasets_dir),
        data.dataset_glob,
        seed=data.shuffle_seed,
        target_rate=data.sampling_rate,
        offset_samples=data.offset_samples,
        num_samples=data.segment_samples,
        target_label=data.target_label,
    )
    # Crop to frame_size (center crop from longer segments)
    frame_size = data.frame_size
    if train_data.shape[1] > frame_size:
        offset = (train_data.shape[1] - frame_size) // 2
        train_data = train_data[:, offset:offset + frame_size]
        val_data = val_data[:, offset:offset + frame_size]
    return train_data, val_data


def _load_raw_segments_from_unified_cache(
    cfg: PpgTwoStreamConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw segments from the unified per-source TFRecord caches.

    Windows are already frame_size-aligned in the cache. No cropping needed.
    """
    from compressionkit.datasets.ppg_cache import SourceWeight, load_cached_raw_windows

    data = cfg.data
    sources = [
        SourceWeight(slug=s.slug, weight=s.weight)
        for s in data.unified_sources
    ]
    cache_root = Path(data.unified_cache_root)

    train_data = load_cached_raw_windows(
        sources,
        cache_root=cache_root,
        frame_size=data.frame_size,
        split="train",
        max_windows=data.max_train_windows,
        seed=data.shuffle_seed,
    )
    val_data = load_cached_raw_windows(
        sources,
        cache_root=cache_root,
        frame_size=data.frame_size,
        split="val",
        max_windows=data.max_val_windows,
        seed=data.shuffle_seed + 1,
    )

    logger.info(
        "Unified cache: train=%d windows, val=%d windows from %d sources",
        len(train_data), len(val_data), len(sources),
    )
    return train_data, val_data


def _load_raw_segments_from_cache(cfg: PpgTwoStreamConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load segments from TFRecord cache with random cropping for diversity.

    Uses the same cache infrastructure as the golden pipeline, providing
    ``windows_per_subject_train`` (default 512) diverse windows per subject.
    """
    data = cfg.data
    frame_size = data.frame_size
    segment_samples = data.segment_samples

    # Build cache (no-op if already built)
    cache_dir, metadata = build_ppg_tfrecord_cache(
        datasets_dir=Path(data.datasets_dir),
        glob_pattern=data.dataset_glob,
        cache_root=Path(data.cache_root),
        target_rate=data.sampling_rate,
        target_label=data.target_label,
        offset_samples=data.offset_samples,
        segment_samples=segment_samples,
        frame_size=frame_size,
        shuffle_seed=data.shuffle_seed,
        train_ratio=data.train_ratio,
        val_ratio=data.val_ratio,
        windows_per_subject_train=data.windows_per_subject_train,
        windows_per_subject_val=data.windows_per_subject_val,
    )

    logger.info(
        "Cache ready: %d train, %d val examples in %s",
        metadata["train_examples"], metadata["val_examples"], cache_dir,
    )

    train_path = cache_dir / metadata["train_tfrecord"]
    val_path = cache_dir / metadata["val_tfrecord"]

    train_data = _read_tfrecord_segments(
        train_path, segment_samples, frame_size, data.shuffle_seed,
    )
    val_data = _read_tfrecord_segments(
        val_path, segment_samples, frame_size, data.shuffle_seed + 1,
    )

    logger.info(
        "Loaded from cache: train=%s, val=%s",
        train_data.shape, val_data.shape,
    )
    return train_data, val_data


def _read_tfrecord_segments(
    tfrecord_path: Path,
    segment_samples: int,
    frame_size: int,
    seed: int,
) -> np.ndarray:
    """Read a TFRecord file and random-crop segments to frame_size."""
    spec = {"signal": tf.io.FixedLenFeature([segment_samples], tf.float32)}
    ds = tf.data.TFRecordDataset(str(tfrecord_path))
    ds = ds.batch(1024)  # batch parsing for speed

    segments: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    max_start = segment_samples - frame_size

    for batch in ds:
        parsed = tf.io.parse_example(batch, spec)
        sigs = parsed["signal"].numpy()  # (batch_size, segment_samples)
        # Random crop each segment
        starts = rng.integers(0, max_start + 1, size=sigs.shape[0])
        for i, start in enumerate(starts):
            segments.append(sigs[i, start:start + frame_size])

    return np.array(segments, dtype=np.float32)


def _decompose_batch(
    segments: np.ndarray,
    *,
    sample_rate: int,
    baseline_cutoff_hz: float,
    filter_order: int,
    epsilon: float,
    baseline_ds_factor: int,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose segments into (baseline_ds, pulsatile_norm) arrays.

    Uses vectorized filtering over the batch axis for efficiency with
    large datasets (e.g. 778K segments from TFRecord cache).
    Processes in chunks to limit peak memory.

    Returns:
        ``(baselines, pulsatiles)`` where:
        - ``baselines``: shape ``(N, frame_size // baseline_ds_factor)``
        - ``pulsatiles``: shape ``(N, frame_size)``
    """
    from scipy.signal import butter, sosfiltfilt

    n = segments.shape[0]
    frame_size = segments.shape[1]
    baseline_len = frame_size // baseline_ds_factor

    nyq = sample_rate / 2.0
    wn = min(baseline_cutoff_hz / nyq, 0.99)
    sos = butter(filter_order, wn, btype="low", output="sos")

    baselines = np.empty((n, baseline_len), dtype=np.float32)
    pulsatiles = np.empty((n, frame_size), dtype=np.float32)

    def _robust_norm_batch(arr: np.ndarray) -> np.ndarray:
        center = np.median(arr, axis=1, keepdims=True)
        mad = np.median(np.abs(arr - center), axis=1, keepdims=True)
        scale = np.maximum(1.4826 * mad, epsilon)
        return ((arr - center) / scale).astype(np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = segments[start:end]

        baseline_raw = sosfiltfilt(sos, chunk, axis=1).astype(np.float32)
        pulsatile_raw = chunk - baseline_raw

        baselines_norm = _robust_norm_batch(baseline_raw)
        pulsatiles[start:end] = _robust_norm_batch(pulsatile_raw)

        # Downsample baselines via striding
        bl_ds = baselines_norm[:, ::baseline_ds_factor]
        baselines[start:end] = bl_ds[:, :baseline_len]

    return baselines, pulsatiles


def _make_tf_dataset(
    data: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    buffer_size: int,
    seed: int,
    noise_range: tuple[float, float] | None = None,
    repeat: bool = False,
) -> tf.data.Dataset:
    """Create a self-supervised tf.data.Dataset (input=target).

    VQAutoencoder expects ``(input, target)`` tuples with shape
    ``(B, 1, T, 1)``.

    When *noise_range* is provided, additive Gaussian noise is applied
    to the input only (target stays clean) as a training regularizer.
    """
    # Shape: (N, 1, T, 1) for the 2D conv architecture
    shaped = data[:, np.newaxis, :, np.newaxis].astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((shaped, shaped))
    if shuffle:
        ds = ds.shuffle(buffer_size, seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)

    if noise_range is not None:
        lo, hi = noise_range

        def _add_noise(x, y):
            std = tf.random.uniform((), lo, hi)
            noise = tf.random.normal(tf.shape(x), stddev=std)
            return x + noise, y

        ds = ds.map(_add_noise, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _compile_model(
    model: keras.Model,
    cfg: PpgTwoStreamConfig,
) -> None:
    """Compile a stream model with optimizer, loss, and extra losses."""
    lr = build_learning_rate(cfg, steps_per_epoch=cfg.data.steps_per_epoch)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    extra = _build_extra_losses(cfg) or None
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        extra_losses=extra,
    )


def _train_stream(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    cfg: PpgTwoStreamConfig,
    stream_name: str,
    run_dir: Path,
) -> keras.callbacks.History:
    """Train one stream model."""
    data = cfg.data
    training = cfg.training

    callbacks_list: list[keras.callbacks.Callback] = []

    # Checkpointing
    weights_path = run_dir / f"best_{stream_name}.weights.h5"
    callbacks_list.append(
        keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            monitor=training.selection_metric,
            mode=training.val_mode,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    )

    # Early stopping
    callbacks_list.append(
        keras.callbacks.EarlyStopping(
            monitor=training.selection_metric,
            mode=training.val_mode,
            patience=training.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        )
    )

    # Reduce LR on plateau (only if not using a LR schedule)
    if training.reduce_lr_on_plateau and not training.lr_schedule.enabled:
        callbacks_list.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=training.selection_metric,
                mode=training.val_mode,
                factor=training.reduce_lr_factor,
                patience=training.reduce_lr_patience,
                min_lr=training.reduce_lr_min_lr,
                verbose=1,
            )
        )

    # TensorBoard
    tb_dir = run_dir / "tensorboard" / stream_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    callbacks_list.append(keras.callbacks.TensorBoard(log_dir=str(tb_dir)))

    logger.info("Training %s stream...", stream_name)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=data.epochs,
        steps_per_epoch=data.steps_per_epoch,
        callbacks=callbacks_list,
        verbose=2,
    )
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_reconstruction(
    baseline_model: keras.Model,
    pulsatile_model: keras.Model,
    val_segments: np.ndarray,
    *,
    cfg: PpgTwoStreamConfig,
) -> dict[str, Any]:
    """Evaluate on raw validation segments: decompose, encode/decode, reconstruct."""
    data = cfg.data
    decompose_cfg = data.decompose
    bm = cfg.baseline_model
    eval_cfg = cfg.evaluation
    n_eval = min(eval_cfg.num_samples, val_segments.shape[0])

    rng = np.random.default_rng(data.shuffle_seed)
    indices = rng.choice(val_segments.shape[0], size=n_eval, replace=False)
    samples = val_segments[indices]

    originals = []
    reconstructions = []

    for i in range(n_eval):
        sig = samples[i]
        # Decompose
        result = decompose_and_normalize(
            sig,
            sample_rate=data.sampling_rate,
            baseline_cutoff_hz=decompose_cfg.baseline_cutoff_hz,
            order=decompose_cfg.filter_order,
            epsilon=decompose_cfg.epsilon,
        )
        # Baseline: downsample → encode/decode → upsample
        baseline_ds = downsample_baseline(
            result["baseline_norm"], factor=bm.downsample_factor,
        )
        baseline_input = baseline_ds[np.newaxis, np.newaxis, :, np.newaxis]
        baseline_recon = baseline_model.predict(baseline_input, verbose=0)
        baseline_recon = baseline_recon.squeeze()
        baseline_up = upsample_baseline(
            baseline_recon, factor=bm.downsample_factor, target_len=data.frame_size,
        )

        # Pulsatile: encode/decode
        pulsatile_input = result["pulsatile_norm"][np.newaxis, np.newaxis, :, np.newaxis]
        pulsatile_recon = pulsatile_model.predict(pulsatile_input, verbose=0)
        pulsatile_recon = pulsatile_recon.squeeze()

        # Reconstruct full signal
        recon = reconstruct_from_streams(
            baseline_up, pulsatile_recon,
            baseline_center=result["baseline_center"],
            baseline_scale=result["baseline_scale"],
            pulsatile_center=result["pulsatile_center"],
            pulsatile_scale=result["pulsatile_scale"],
        )
        originals.append(sig)
        reconstructions.append(recon)

    originals_arr = np.array(originals, dtype=np.float32)
    recons_arr = np.array(reconstructions, dtype=np.float32)

    # Primary metrics (full signal)
    primary = compute_signal_metrics(originals_arr, recons_arr)
    return {
        "primary_metrics": primary,
        "originals": originals_arr,
        "reconstructions": recons_arr,
    }


def _evaluate_physiokit(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    *,
    cfg: PpgTwoStreamConfig,
) -> dict[str, Any] | None:
    """Compute HR/HRV metrics on original vs reconstructed."""
    physio_cfg = cfg.evaluation.physiokit_metrics
    if not physio_cfg.enabled:
        return None

    summary, per_sample = summarize_physiokit_alignment(
        originals, reconstructions,
        sample_rate=cfg.data.sampling_rate,
        low_hz=physio_cfg.low_hz,
        high_hz=physio_cfg.high_hz,
        order=physio_cfg.order,
        min_peaks=physio_cfg.min_peaks,
    )
    return {"summary": summary, "per_sample_count": len(per_sample)}


def _evaluate_spectral(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    *,
    cfg: PpgTwoStreamConfig,
) -> dict[str, Any] | None:
    """Compute per-band PSD error across evaluation samples."""
    spectral_cfg = cfg.evaluation.spectral_metrics
    if not spectral_cfg.enabled:
        return None

    fs = cfg.data.sampling_rate
    bands = [tuple(b) for b in spectral_cfg.bands]
    all_band_errors: list[dict[str, float]] = []

    for i in range(originals.shape[0]):
        errs = psd_band_error(
            originals[i], reconstructions[i], fs=fs, bands=bands,
        )
        all_band_errors.append(errs)

    # Aggregate: mean across samples for each metric
    agg: dict[str, float] = {}
    if all_band_errors:
        keys = all_band_errors[0].keys()
        for k in keys:
            vals = [e[k] for e in all_band_errors if k in e]
            agg[k] = float(np.mean(vals)) if vals else 0.0
    return agg


def _evaluate_stitching(
    baseline_model: keras.Model,
    pulsatile_model: keras.Model,
    *,
    cfg: PpgTwoStreamConfig,
) -> dict[str, Any] | None:
    """Evaluate stitching quality on long PPG recordings."""
    stitch_cfg = cfg.evaluation.stitching
    if not stitch_cfg.enabled:
        return None

    data = cfg.data
    decompose_cfg = data.decompose
    bm = cfg.baseline_model

    _, val_files, _ = load_ppg_file_splits(
        Path(data.datasets_dir), data.dataset_glob, seed=data.shuffle_seed,
    )
    if not val_files:
        logger.warning("No validation files for stitching evaluation.")
        return None

    rng = np.random.default_rng(data.shuffle_seed)
    n_rec = min(stitch_cfg.num_recordings, len(val_files))
    selected_indices = rng.choice(len(val_files), size=n_rec, replace=False)
    selected = [val_files[i] for i in sorted(selected_indices)]

    target_samples = int(stitch_cfg.duration_sec * data.sampling_rate)

    def _two_stream_predict(frames_4d: np.ndarray) -> np.ndarray:
        """Predict for frames that contain raw PPG (apply decompose internally)."""
        batch_size = frames_4d.shape[0]
        frame_size = frames_4d.shape[2]
        results = np.empty_like(frames_4d)

        for i in range(batch_size):
            sig = frames_4d[i, 0, :, 0]
            result = decompose_and_normalize(
                sig,
                sample_rate=data.sampling_rate,
                baseline_cutoff_hz=decompose_cfg.baseline_cutoff_hz,
                order=decompose_cfg.filter_order,
                epsilon=decompose_cfg.epsilon,
            )
            # Baseline
            baseline_ds = downsample_baseline(
                result["baseline_norm"], factor=bm.downsample_factor,
            )
            bl_in = baseline_ds[np.newaxis, np.newaxis, :, np.newaxis]
            bl_out = baseline_model.predict(bl_in, verbose=0).squeeze()
            bl_up = upsample_baseline(bl_out, factor=bm.downsample_factor, target_len=frame_size)

            # Pulsatile
            pl_in = result["pulsatile_norm"][np.newaxis, np.newaxis, :, np.newaxis]
            pl_out = pulsatile_model.predict(pl_in, verbose=0).squeeze()

            # Reconstruct
            recon = reconstruct_from_streams(
                bl_up, pl_out,
                baseline_center=result["baseline_center"],
                baseline_scale=result["baseline_scale"],
                pulsatile_center=result["pulsatile_center"],
                pulsatile_scale=result["pulsatile_scale"],
            )
            results[i, 0, :, 0] = recon

        return results

    # Vectorized prediction for batched stitching
    def _batch_predict(frames_4d: np.ndarray) -> np.ndarray:
        """Batch predict with per-frame decomposition."""
        return _two_stream_predict(frames_4d)

    per_method: dict[str, dict[str, list[float]]] = {
        m: {"prd_percent": [], "cosine_similarity": [], "mse": [],
            "seam_ratio": [], "seam_rms": [], "non_seam_rms": []}
        for m in stitch_cfg.methods
    }

    for fpath in selected:
        try:
            signal = load_ppg_signal(
                Path(fpath),
                target_rate=data.sampling_rate,
                target_label=data.target_label,
                offset_samples=data.offset_samples,
            )
        except Exception as exc:
            logger.debug("Skipping %s: %s", Path(fpath).name, exc)
            continue

        signal = np.asarray(signal, dtype=np.float32).ravel()
        if signal.size < data.frame_size * 2:
            continue
        if signal.size > target_samples:
            signal = signal[:target_samples]

        for method in stitch_cfg.methods:
            kwargs: dict[str, Any] = {"epsilon": decompose_cfg.epsilon}
            if method != "hard_concat":
                kwargs["hop_ratio"] = stitch_cfg.hop_ratio

            recon = stitch(method, _batch_predict, signal, data.frame_size, **kwargs)
            metrics = compute_signal_metrics(signal, recon)

            effective_hop = 1.0 if method == "hard_concat" else stitch_cfg.hop_ratio
            seam = seam_discontinuity_ratio(
                recon, frame_size=data.frame_size, hop_ratio=effective_hop, radius=4,
            )

            acc = per_method[method]
            acc["prd_percent"].append(metrics["prd_percent"])
            acc["cosine_similarity"].append(metrics["cosine_similarity"])
            acc["mse"].append(metrics["mse"])
            if np.isfinite(seam["ratio"]):
                acc["seam_ratio"].append(seam["ratio"])
                acc["seam_rms"].append(seam["seam_rms"])
                acc["non_seam_rms"].append(seam["non_seam_rms"])

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    summary: dict[str, dict[str, float | int]] = {}
    for m, acc in per_method.items():
        summary[m] = {
            "num_recordings": len(acc["prd_percent"]),
            "mean_prd_percent": _mean(acc["prd_percent"]),
            "mean_cosine_similarity": _mean(acc["cosine_similarity"]),
            "mean_mse": _mean(acc["mse"]),
            "mean_seam_ratio": _mean(acc["seam_ratio"]),
            "mean_seam_rms": _mean(acc["seam_rms"]),
            "mean_non_seam_rms": _mean(acc["non_seam_rms"]),
        }
    return {
        "duration_sec": stitch_cfg.duration_sec,
        "hop_ratio": stitch_cfg.hop_ratio,
        "frame_size": data.frame_size,
        "methods": summary,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_two_stream(cfg: PpgTwoStreamConfig) -> dict[str, Any]:
    """Full training + evaluation pipeline for the two-stream PPG codec.

    Args:
        cfg: Two-stream config (typically loaded from YAML).

    Returns:
        Results dict with metrics, compression stats, and report paths.
    """
    data = cfg.data
    run_dir = Path(cfg.output.results_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = run_dir / "config.json"
    config_path.write_text(cfg.model_dump_json(indent=2))
    logger.info("Config saved to %s", config_path)

    # Compression stats
    cr_stats = compute_combined_compression_stats(cfg)
    logger.info(
        "Two-stream compression ratio: %.2f "
        "(baseline: %.1f bits, pulsatile: %.1f bits, total: %.1f bits)",
        cr_stats["compression_ratio"],
        cr_stats["baseline_compressed_bits"],
        cr_stats["pulsatile_compressed_bits"],
        cr_stats["total_compressed_bits"],
    )

    # Load data
    logger.info("Loading raw PPG segments...")
    train_segments, val_segments = _load_raw_segments(cfg)
    logger.info("Train: %d segments, Val: %d segments", len(train_segments), len(val_segments))

    # Decompose into streams
    decompose_cfg = data.decompose
    bm = cfg.baseline_model

    logger.info("Decomposing into baseline + pulsatile streams...")
    train_baselines, train_pulsatiles = _decompose_batch(
        train_segments,
        sample_rate=data.sampling_rate,
        baseline_cutoff_hz=decompose_cfg.baseline_cutoff_hz,
        filter_order=decompose_cfg.filter_order,
        epsilon=decompose_cfg.epsilon,
        baseline_ds_factor=bm.downsample_factor,
    )
    val_baselines, val_pulsatiles = _decompose_batch(
        val_segments,
        sample_rate=data.sampling_rate,
        baseline_cutoff_hz=decompose_cfg.baseline_cutoff_hz,
        filter_order=decompose_cfg.filter_order,
        epsilon=decompose_cfg.epsilon,
        baseline_ds_factor=bm.downsample_factor,
    )
    logger.info(
        "Baseline shape: %s, Pulsatile shape: %s",
        train_baselines.shape, train_pulsatiles.shape,
    )

    # Build tf.data
    noise_range = tuple(data.gaussian_noise) if len(data.gaussian_noise) == 2 else None
    train_bl_ds = _make_tf_dataset(
        train_baselines, batch_size=data.batch_size, shuffle=True,
        buffer_size=data.buffer_size, seed=data.shuffle_seed,
        noise_range=noise_range, repeat=True,
    )
    val_bl_ds = _make_tf_dataset(
        val_baselines, batch_size=data.batch_size, shuffle=False,
        buffer_size=data.buffer_size, seed=data.shuffle_seed,
    )
    train_pl_ds = _make_tf_dataset(
        train_pulsatiles, batch_size=data.batch_size, shuffle=True,
        buffer_size=data.buffer_size, seed=data.shuffle_seed,
        noise_range=noise_range, repeat=True,
    )
    val_pl_ds = _make_tf_dataset(
        val_pulsatiles, batch_size=data.batch_size, shuffle=False,
        buffer_size=data.buffer_size, seed=data.shuffle_seed,
    )

    # Build models
    logger.info("Building baseline model...")
    baseline_model, bl_stats = build_baseline_model(cfg)
    _compile_model(baseline_model, cfg)
    baseline_model.summary(print_fn=logger.info)

    logger.info("Building pulsatile model...")
    pulsatile_model, pl_stats = build_pulsatile_model(cfg)
    _compile_model(pulsatile_model, cfg)
    pulsatile_model.summary(print_fn=logger.info)

    # Train baseline
    bl_history = _train_stream(
        baseline_model, train_bl_ds, val_bl_ds,
        cfg=cfg, stream_name="baseline", run_dir=run_dir,
    )

    # Train pulsatile
    pl_history = _train_stream(
        pulsatile_model, train_pl_ds, val_pl_ds,
        cfg=cfg, stream_name="pulsatile", run_dir=run_dir,
    )

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    logger.info("Running evaluation...")

    # Primary metrics (full reconstruction)
    recon_results = _evaluate_reconstruction(
        baseline_model, pulsatile_model, val_segments, cfg=cfg,
    )
    primary = recon_results["primary_metrics"]
    logger.info(
        "Primary: MSE=%.6f PRD=%.2f%% cos=%.4f",
        primary["mse"], primary["prd_percent"], primary["cosine_similarity"],
    )

    # HR/HRV metrics
    physio_results = _evaluate_physiokit(
        recon_results["originals"], recon_results["reconstructions"], cfg=cfg,
    )
    if physio_results and physio_results["summary"]:
        ps = physio_results["summary"]
        logger.info(
            "Physiokit: HR_MAE=%.2f bpm, SDNN_MAE=%.2f ms, RMSSD_MAE=%.2f ms",
            ps.get("hr_mae_bpm", float("nan")),
            ps.get("sdnn_mae_ms", float("nan")),
            ps.get("rmssd_mae_ms", float("nan")),
        )

    # Spectral metrics
    spectral_results = _evaluate_spectral(
        recon_results["originals"], recon_results["reconstructions"], cfg=cfg,
    )
    if spectral_results:
        logger.info(
            "Spectral: band_total_rel_error=%.4f",
            spectral_results.get("band_total_rel_error", float("nan")),
        )

    # Stitching evaluation
    stitch_results = _evaluate_stitching(baseline_model, pulsatile_model, cfg=cfg)
    if stitch_results:
        for method, stats in stitch_results["methods"].items():
            logger.info(
                "Stitching [%s]: PRD=%.2f%% cos=%.4f seam_ratio=%.3f",
                method,
                stats["mean_prd_percent"],
                stats["mean_cosine_similarity"],
                stats["mean_seam_ratio"],
            )

    # -----------------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------------
    report = {
        "run_name": cfg.run_name,
        "compression_stats": cr_stats,
        "baseline_stream_stats": bl_stats,
        "pulsatile_stream_stats": pl_stats,
        "primary_metrics": primary,
        "physiokit_metrics": physio_results,
        "spectral_metrics": spectral_results,
        "stitching_metrics": stitch_results,
        "baseline_final_val_loss": float(bl_history.history.get("val_loss", [float("nan")])[-1]),
        "pulsatile_final_val_loss": float(pl_history.history.get("val_loss", [float("nan")])[-1]),
    }

    report_path = run_dir / "two_stream_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", report_path)

    return report
