"""ECG RVQ training orchestration.

Wires together datasets, preprocessing, model build, training, evaluation,
and export into one coherent flow driven by an ``EcgRvqConfig``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.datasets.ecg import (
    bandpass_filter_batch,
    build_ecg_tfrecord_cache,
    collect_random_samples,
    load_ecg_file_splits,
    load_ecg_splits,
    make_ecg_inmemory_dataset,
    make_ecg_stream_dataset,
    make_ecg_tfrecord_dataset,
)
from compressionkit.dsp import (
    DwtConfig,
    StftConfig,
    dwt_pack,
    dwt_unpack,
    stft_forward,
    stft_inverse,
    stft_output_shape,
)
from compressionkit.evaluation.artifacts import save_sample_artifacts
from compressionkit.evaluation.ecg_stitching import evaluate_stitching
from compressionkit.evaluation.metrics import (
    TruePRD,
    compute_signal_metrics,
    summarize_ecg_alignment,
)
from compressionkit.models.rvq_autoencoder import (
    build_rvq_autoencoder,
    build_rvq_autoencoder_2d_spatial,
    compute_compression_stats,
)
from compressionkit.preprocessing.ecg import (
    generate_synthetic_ecg_batch,
)
from compressionkit.losses import (
    build_derivative_loss as _build_derivative_loss,
    build_dwt_loss as _build_dwt_loss,
    build_filtered_mse_loss as _build_filtered_mse_loss,
    build_multi_scale_spectral_loss as _build_multi_scale_spectral_loss,
)
from compressionkit.trainers.utils import (
    build_learning_rate,
)

logger = logging.getLogger("ecg-rvq-trainer")


# ---------------------------------------------------------------------------
# Transform-domain dataset wrapping
# ---------------------------------------------------------------------------

# STFT pad target: zero-pad the spectrogram from (33, 33, 2) to (64, 64, 2)
# for clean power-of-2 spatial downsampling.  4 stages of stride-2 give
# (4, 4, emb) = 16 latent positions.  With 2 RVQ levels × 8 bits = 256 bits
# → 32× CR (matching the raw-signal baseline).
STFT_PAD_HEIGHT = 64
STFT_PAD_WIDTH = 64


def _build_transform_configs(cfg: EcgRvqConfig) -> tuple[str, StftConfig | DwtConfig | None]:
    """Return (domain, transform_config) from the pipeline config."""
    tcfg = cfg.data.transform
    domain = tcfg.domain.strip().lower()
    if domain == "raw":
        return domain, None
    if domain == "dwt":
        return domain, DwtConfig(levels=tcfg.dwt_levels, wavelet=tcfg.dwt_wavelet)
    if domain == "stft":
        return domain, StftConfig(
            n_fft=tcfg.stft_n_fft,
            hop_length=tcfg.stft_hop_length,
            window=tcfg.stft_window,
        )
    raise ValueError(f"Unknown transform domain: {domain!r}")


def _wrap_dataset_with_transform(
    ds: tf.data.Dataset,
    domain: str,
    transform_cfg: StftConfig | DwtConfig | None,
    frame_size: int,
) -> tf.data.Dataset:
    """Wrap a dataset to apply forward transform to both input and target.

    The upstream dataset yields ``(input, target)`` of shape
    ``(B, 1, frame_size, 1)`` for raw/DWT or the same for raw before STFT.
    After wrapping:
      - DWT: ``(B, 1, frame_size, 1)`` (same shape, packed coefficients).
      - STFT: ``(B, STFT_PAD_HEIGHT, STFT_PAD_WIDTH, 2)`` (zero-padded spectrogram).
    """
    if domain == "raw" or transform_cfg is None:
        return ds

    if domain == "dwt":
        dwt_cfg = transform_cfg

        def _dwt_forward_batch(x: np.ndarray) -> np.ndarray:
            """Apply dwt_pack to a batch of shape (B, 1, T, 1)."""
            B = x.shape[0]
            out = np.empty_like(x)
            for i in range(B):
                sig = x[i, 0, :, 0]
                out[i, 0, :, 0] = dwt_pack(sig, dwt_cfg)
            return out

        def _apply_dwt(inp, target):
            inp_t = tf.numpy_function(_dwt_forward_batch, [inp], tf.float32)
            tgt_t = tf.numpy_function(_dwt_forward_batch, [target], tf.float32)
            inp_t.set_shape(inp.shape)
            tgt_t.set_shape(target.shape)
            return inp_t, tgt_t

        return ds.map(_apply_dwt, num_parallel_calls=tf.data.AUTOTUNE)

    if domain == "stft":
        stft_cfg = transform_cfg

        def _stft_forward_batch(x: np.ndarray) -> np.ndarray:
            """Apply stft_forward, then zero-pad to (PAD_H, PAD_W, 2)."""
            B = x.shape[0]
            out = np.zeros((B, STFT_PAD_HEIGHT, STFT_PAD_WIDTH, 2), dtype=np.float32)
            for i in range(B):
                sig = x[i, 0, :, 0]
                spec = stft_forward(sig, stft_cfg)  # (frames, freq_bins, 2)
                h, w = spec.shape[0], spec.shape[1]
                out[i, :h, :w, :] = spec
            return out

        out_shape = (None, STFT_PAD_HEIGHT, STFT_PAD_WIDTH, 2)

        def _apply_stft(inp, target):
            inp_t = tf.numpy_function(_stft_forward_batch, [inp], tf.float32)
            tgt_t = tf.numpy_function(_stft_forward_batch, [target], tf.float32)
            inp_t.set_shape(out_shape)
            tgt_t.set_shape(out_shape)
            return inp_t, tgt_t

        return ds.map(_apply_stft, num_parallel_calls=tf.data.AUTOTUNE)

    raise ValueError(f"Unknown transform domain: {domain!r}")


def _inverse_transform_batch(
    recon: np.ndarray,
    domain: str,
    transform_cfg: StftConfig | DwtConfig | None,
    frame_size: int,
) -> np.ndarray:
    """Inverse-transform reconstructed output back to time domain.

    Args:
        recon: Reconstructed batch. Shape depends on domain:
            - raw/dwt: ``(B, 1, T, 1)``
            - stft: ``(B, H, W, 2)``
        domain: Transform domain name.
        transform_cfg: Transform config.
        frame_size: Original signal length (for STFT inverse).

    Returns:
        Time-domain signals of shape ``(B, 1, frame_size, 1)``.
    """
    if domain == "raw" or transform_cfg is None:
        return recon

    B = recon.shape[0]
    out = np.empty((B, 1, frame_size, 1), dtype=np.float32)

    if domain == "dwt":
        for i in range(B):
            packed = recon[i, 0, :, 0]
            out[i, 0, :, 0] = dwt_unpack(packed, transform_cfg, frame_size)
        return out

    if domain == "stft":
        stft_cfg = transform_cfg
        # Crop zero-padded spectrogram back to actual STFT shape before inversion
        full_shape = stft_output_shape(frame_size, stft_cfg)  # (frames, freq_bins, 2)
        h, w = full_shape[0], full_shape[1]
        for i in range(B):
            spec = recon[i, :h, :w, :]  # crop from (64, 64, 2) to (33, 33, 2)
            out[i, 0, :, 0] = stft_inverse(spec, stft_cfg, frame_size)
        return out

    raise ValueError(f"Unknown domain: {domain!r}")


# ---------------------------------------------------------------------------
# Synthetic ECG TFRecord helper
# ---------------------------------------------------------------------------

def _write_synthetic_ecg_tfrecord(
    *,
    cache_dir: Path,
    num_segments: int,
    segment_samples: int,
    sample_rate: int,
    lead_index: int,
    synth_cfg: Any,
) -> Path:
    """Generate synthetic ECG windows and write as a TFRecord.

    Skips generation if a matching file already exists in *cache_dir*.

    Returns the path to the written ``synthetic_train.tfrecord``.
    """
    tfrecord_path = cache_dir / "synthetic_train.tfrecord"
    meta_path = cache_dir / "synthetic_meta.json"

    # Reuse existing synthetic TFRecord if params match
    if tfrecord_path.exists() and meta_path.exists():
        import json as _json
        existing = _json.load(meta_path.open())
        if (
            existing.get("num_segments") == num_segments
            and existing.get("segment_samples") == segment_samples
            and existing.get("sample_rate") == sample_rate
            and existing.get("lead_index") == lead_index
            and existing.get("seed") == synth_cfg.seed
        ):
            logger.info(
                "Reusing existing synthetic TFRecord (%d segments) at %s",
                num_segments, tfrecord_path,
            )
            return tfrecord_path

    logger.info(
        "Generating %d synthetic ECG segments (lead %d, sr=%d, len=%d)...",
        num_segments, lead_index, sample_rate, segment_samples,
    )
    synth_data = generate_synthetic_ecg_batch(
        num_segments=num_segments,
        signal_length=segment_samples,
        sample_rate=sample_rate,
        lead_index=lead_index,
        heart_rate_bpm=synth_cfg.heart_rate_bpm,
        noise_multiplier=synth_cfg.noise_multiplier,
        impedance=synth_cfg.impedance,
        seed=synth_cfg.seed,
    )
    writer = tf.io.TFRecordWriter(str(tfrecord_path))
    for i in range(synth_data.shape[0]):
        feature = {
            "signal": tf.train.Feature(
                float_list=tf.train.FloatList(value=synth_data[i].ravel().tolist())
            ),
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        writer.write(example.SerializeToString())
    writer.close()

    # Write metadata for reuse detection
    import json as _json
    meta = {
        "num_segments": num_segments,
        "segment_samples": segment_samples,
        "sample_rate": sample_rate,
        "lead_index": lead_index,
        "seed": synth_cfg.seed,
    }
    with meta_path.open("w") as f:
        _json.dump(meta, f, indent=2)

    logger.info("Wrote %d synthetic segments to %s", num_segments, tfrecord_path)
    return tfrecord_path


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_datasets(
    cfg: EcgRvqConfig,
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

        cache_dir, cache_meta = build_ecg_tfrecord_cache(
            datasets_dir=Path(data.datasets_dir),
            glob_pattern=data.dataset_glob,
            cache_root=cache_root,
            lead_index=data.lead_index,
            leads=data.leads,
            segment_samples=effective_segment_samples,
            frame_size=data.frame_size,
            shuffle_seed=data.shuffle_seed,
            train_ratio=cache_cfg.train_ratio,
            val_ratio=cache_cfg.val_ratio,
            windows_per_subject_train=cache_cfg.windows_per_subject_train,
            windows_per_subject_val=cache_cfg.windows_per_subject_val,
            force_rebuild=cache_cfg.force_rebuild,
            source_sample_rate=data.sampling_rate,
            target_sample_rate=data.target_sample_rate,
        )
        info["cache_dir"] = str(cache_dir)
        train_path = cache_dir / str(cache_meta["train_tfrecord"])
        val_path = cache_dir / str(cache_meta["val_tfrecord"])
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"Missing TFRecord files in cache dir: {cache_dir}")

        # Optional synthetic ECG mixing (cache mode)
        train_paths: list[Path] = [train_path]
        synth_cfg = data.synthetic_mix
        info["synthetic_added"] = 0
        info["synthetic_fraction_effective"] = 0.0
        if synth_cfg.enabled:
            if not 0.0 < synth_cfg.fraction < 1.0:
                raise ValueError(f"synthetic_mix.fraction must be in (0, 1), got {synth_cfg.fraction}")
            n_real = int(cache_meta.get("train_examples", 0))
            synthetic_added = int(np.ceil((synth_cfg.fraction * n_real) / (1.0 - synth_cfg.fraction)))
            synth_path = _write_synthetic_ecg_tfrecord(
                cache_dir=cache_dir,
                num_segments=synthetic_added,
                segment_samples=effective_segment_samples,
                sample_rate=data.effective_sample_rate,
                lead_index=data.lead_index,
                synth_cfg=synth_cfg,
            )
            train_paths.append(synth_path)
            info["synthetic_added"] = synthetic_added
            info["synthetic_fraction_effective"] = float(
                synthetic_added / max(n_real + synthetic_added, 1)
            )
            logger.info(
                "Synthetic mix: %d real + %d synthetic = %.1f%% synthetic",
                n_real, synthetic_added, info["synthetic_fraction_effective"] * 100,
            )

        train_ds = make_ecg_tfrecord_dataset(
            train_paths,
            frame_size=data.frame_size,
            segment_samples=effective_segment_samples,
            batch_size=data.batch_size,
            shuffle_buffer_size=data.buffer_size,
            preprocessor=preprocessor,
            augmenter=augmenter,
            input_filter_cfg=input_filter_dict if data.input_filter.enabled else None,
            target_filter_cfg=target_filter_dict if data.target_filter.enabled else None,
            sample_rate=data.effective_sample_rate,
            num_leads=data.num_leads,
            shuffle=True,
            seed=data.shuffle_seed,
        )
        val_ds = make_ecg_tfrecord_dataset(
            [val_path],
            frame_size=data.frame_size,
            segment_samples=effective_segment_samples,
            batch_size=data.batch_size,
            shuffle_buffer_size=data.buffer_size,
            preprocessor=preprocessor,
            augmenter=None,
            input_filter_cfg=input_filter_dict if data.input_filter.enabled else None,
            target_filter_cfg=target_filter_dict if data.target_filter.enabled else None,
            sample_rate=data.effective_sample_rate,
            num_leads=data.num_leads,
            shuffle=False,
            seed=data.shuffle_seed,
        )
        if validation_steps is None:
            val_examples = int(cache_meta.get("val_examples", 0))
            validation_steps = max(1, val_examples // data.batch_size)

    elif streaming_cfg.enabled:
        info["mode"] = "streaming"
        train_files, val_files, _ = load_ecg_file_splits(
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
            "lead_index": data.lead_index,
            "preprocessor": preprocessor,
            "input_filter_cfg": input_filter_dict if data.input_filter.enabled else None,
            "target_filter_cfg": target_filter_dict if data.target_filter.enabled else None,
            "seed": data.shuffle_seed,
        }
        train_ds = make_ecg_stream_dataset(
            train_files,
            subject_buffer_size=streaming_cfg.subject_buffer_size,
            window_buffer_size=streaming_cfg.window_buffer_size,
            windows_per_subject=streaming_cfg.windows_per_subject_train,
            augmenter=augmenter,
            shuffle=True,
            **stream_kwargs,
        )
        val_ds = make_ecg_stream_dataset(
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
        train_data, val_data, _ = load_ecg_splits(
            Path(data.datasets_dir),
            data.dataset_glob,
            seed=data.shuffle_seed,
            lead_index=data.lead_index,
        )

        # Optional synthetic ECG mixing (in-memory mode)
        synth_cfg = data.synthetic_mix
        info["synthetic_added"] = 0
        info["synthetic_fraction_effective"] = 0.0
        if synth_cfg.enabled:
            if not 0.0 < synth_cfg.fraction < 1.0:
                raise ValueError(f"synthetic_mix.fraction must be in (0, 1), got {synth_cfg.fraction}")
            n_real = train_data.shape[0]
            synthetic_added = int(np.ceil((synth_cfg.fraction * n_real) / (1.0 - synth_cfg.fraction)))
            synth_data = generate_synthetic_ecg_batch(
                num_segments=synthetic_added,
                signal_length=data.segment_samples,
                sample_rate=data.effective_sample_rate,
                lead_index=data.lead_index,
                heart_rate_bpm=synth_cfg.heart_rate_bpm,
                noise_multiplier=synth_cfg.noise_multiplier,
                impedance=synth_cfg.impedance,
                seed=synth_cfg.seed,
            )
            train_data = np.concatenate([train_data, synth_data], axis=0)
            info["synthetic_added"] = synthetic_added
            info["synthetic_fraction_effective"] = float(
                synthetic_added / max(train_data.shape[0], 1)
            )
            logger.info(
                "Synthetic mix: %d real + %d synthetic = %.1f%% synthetic",
                n_real, synthetic_added, info["synthetic_fraction_effective"] * 100,
            )

        train_data = train_data[:, :, np.newaxis]
        val_data = val_data[:, :, np.newaxis]

        # Optional filtering
        train_target_data = None
        val_target_data = None
        if data.input_filter.enabled:
            train_data = bandpass_filter_batch(
                train_data, sample_rate=data.effective_sample_rate,
                low_hz=data.input_filter.low_hz, high_hz=data.input_filter.high_hz,
                order=data.input_filter.order,
            )
            val_data = bandpass_filter_batch(
                val_data, sample_rate=data.effective_sample_rate,
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
                    raw_train, sample_rate=data.effective_sample_rate,
                    low_hz=data.target_filter.low_hz, high_hz=data.target_filter.high_hz,
                    order=data.target_filter.order,
                )
                val_target_data = bandpass_filter_batch(
                    raw_val, sample_rate=data.effective_sample_rate,
                    low_hz=data.target_filter.low_hz, high_hz=data.target_filter.high_hz,
                    order=data.target_filter.order,
                )

        train_ds = make_ecg_inmemory_dataset(
            train_data, frame_size=data.frame_size, batch_size=data.batch_size,
            buffer_size=data.buffer_size, preprocessor=preprocessor,
            augmenter=augmenter, target_data=train_target_data, shuffle=True,
        )
        val_ds = make_ecg_inmemory_dataset(
            val_data, frame_size=data.frame_size, batch_size=data.batch_size,
            buffer_size=data.buffer_size, preprocessor=preprocessor,
            augmenter=augmenter, target_data=val_target_data, shuffle=False,
        )

    # Apply forward transform if configured (DWT / STFT)
    domain, transform_cfg = _build_transform_configs(cfg)
    if domain != "raw":
        logger.info("Wrapping datasets with forward transform: %s", domain)
        train_ds = _wrap_dataset_with_transform(train_ds, domain, transform_cfg, data.frame_size)
        val_ds = _wrap_dataset_with_transform(val_ds, domain, transform_cfg, data.frame_size)
        info["transform_domain"] = domain

    return train_ds, val_ds, validation_steps, info


# NOTE: build_callbacks is imported from compressionkit.trainers.utils


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    cfg: EcgRvqConfig,
    *,
    model: keras.Model,
    val_ds: tf.data.Dataset,
    run_dir: Path,
    validation_steps: int | None,
) -> dict[str, Any]:
    """Run post-training evaluation: sample reconstruction + band metrics.

    When a transform domain is active, reconstructions are inverse-transformed
    back to the time domain before computing signal-quality metrics and saving
    sample artifacts.
    """
    data = cfg.data
    eval_cfg = cfg.evaluation
    rng = np.random.default_rng(data.shuffle_seed)
    sample_inputs, sample_targets = collect_random_samples(val_ds, eval_cfg.num_samples, rng)
    reconstructions = model.predict(sample_inputs, verbose=0)

    # Inverse-transform back to time domain for signal-level evaluation
    domain, transform_cfg = _build_transform_configs(cfg)
    if domain != "raw":
        sample_targets = _inverse_transform_batch(
            sample_targets, domain, transform_cfg, data.frame_size,
        )
        reconstructions = _inverse_transform_batch(
            reconstructions, domain, transform_cfg, data.frame_size,
        )
        logger.info("Inverse-transformed evaluation samples from %s → time domain", domain)

    band_cfg = eval_cfg.band_metrics

    band_sample_targets = None
    band_reconstructions = None
    best_band_metrics = None

    if band_cfg.enabled and data.num_leads == 1:
        targets_seq = sample_targets.reshape(sample_targets.shape[0], -1).astype(np.float32)
        recon_seq = reconstructions.reshape(reconstructions.shape[0], -1).astype(np.float32)
        band_sample_targets = bandpass_filter_batch(
            targets_seq, sample_rate=data.effective_sample_rate,
            low_hz=band_cfg.low_hz, high_hz=band_cfg.high_hz, order=band_cfg.order,
        )
        band_reconstructions = bandpass_filter_batch(
            recon_seq, sample_rate=data.effective_sample_rate,
            low_hz=band_cfg.low_hz, high_hz=band_cfg.high_hz, order=band_cfg.order,
        )
        best_band_metrics = compute_signal_metrics(band_sample_targets, band_reconstructions)

    num_leads = data.num_leads

    if num_leads > 1:
        # Multi-lead: save per-lead artifacts, aggregate metrics across leads
        plot_cap = max(0, min(eval_cfg.num_plot_samples, len(sample_targets)))
        sample_results = {}
        for idx, (target, recon) in enumerate(zip(sample_targets, reconstructions)):
            # target/recon shape: (1, T, C) → squeeze to (T, C)
            t2d = target.squeeze()  # (T, C)
            r2d = recon.squeeze()   # (T, C)
            save_plot_this = idx < plot_cap
            lead_metrics_list = []
            for lead_idx in range(num_leads):
                lead_result = save_sample_artifacts(
                    sample_id=idx * 100 + lead_idx,
                    original=t2d[:, lead_idx],
                    reconstructed=r2d[:, lead_idx],
                    sampling_rate=data.effective_sample_rate,
                    run_dir=run_dir,
                    save_plot=save_plot_this,
                )
                lead_metrics_list.append(lead_result.get("metrics", {}))
            # Also save lead II (index 1) as the "primary" sample artifact
            primary_result = save_sample_artifacts(
                sample_id=idx,
                original=t2d[:, 1] if num_leads > 1 else t2d[:, 0],
                reconstructed=r2d[:, 1] if num_leads > 1 else r2d[:, 0],
                sampling_rate=data.effective_sample_rate,
                run_dir=run_dir,
                save_plot=save_plot_this,
            )
            # Average metrics across leads
            avg_metrics = {}
            for key in lead_metrics_list[0]:
                vals = [m[key] for m in lead_metrics_list if key in m]
                avg_metrics[key] = float(np.mean(vals))
            primary_result["metrics"] = avg_metrics
            sample_results[str(idx)] = primary_result
    else:
        plot_cap = max(0, min(eval_cfg.num_plot_samples, len(sample_targets)))
        sample_results = {
            str(idx): save_sample_artifacts(
                idx, target.squeeze(), recon.squeeze(), data.effective_sample_rate, run_dir,
                band_original=None if band_sample_targets is None else band_sample_targets[idx],
                band_reconstructed=None if band_reconstructions is None else band_reconstructions[idx],
                save_plot=(idx < plot_cap),
            )
            for idx, (target, recon) in enumerate(zip(sample_targets, reconstructions))
        }

    # --- ECG physiology (HR / HRV / peak-timing) -------------------------
    ecg_physiology: dict[str, Any] | None = None
    if num_leads == 1:
        targets_seq = sample_targets.reshape(sample_targets.shape[0], -1).astype(np.float32)
        recon_seq = reconstructions.reshape(reconstructions.shape[0], -1).astype(np.float32)
        ecg_summary, _ = summarize_ecg_alignment(
            targets_seq, recon_seq, sample_rate=data.effective_sample_rate,
        )
        if ecg_summary is not None:
            ecg_physiology = ecg_summary
            logger.info(
                "ECG alignment: HR MAE %.2f bpm, peak timing MAE %.1f ms",
                ecg_summary.get("hr_mae_bpm", float("nan")),
                ecg_summary.get("peak_timing_mae_ms", float("nan")),
            )

    stitching_results: dict[str, Any] | None = None
    stitching_cfg = eval_cfg.stitching
    if stitching_cfg.enabled:
        if data.num_leads != 1:
            logger.warning(
                "Stitching evaluation only supports single-lead data (num_leads=%d); skipping.",
                data.num_leads,
            )
        else:
            logger.info(
                "Running stitching evaluation (methods=%s, %d recordings, %.0fs each)...",
                stitching_cfg.methods, stitching_cfg.num_recordings, stitching_cfg.duration_sec,
            )
            stitching_results = evaluate_stitching(
                model,
                datasets_dir=Path(data.datasets_dir),
                dataset_glob=data.dataset_glob,
                frame_size=data.frame_size,
                sample_rate=data.effective_sample_rate,
                duration_sec=stitching_cfg.duration_sec,
                epsilon=data.epsilon,
                methods=stitching_cfg.methods,
                hop_ratio=stitching_cfg.hop_ratio,
                num_recordings=stitching_cfg.num_recordings,
                batch_size=stitching_cfg.batch_size,
                seed=data.shuffle_seed,
                lead_index=data.lead_index if hasattr(data, "lead_index") else 1,
                seam_radius=stitching_cfg.seam_radius,
            )

    return {
        "samples": sample_results,
        "sample_inputs": sample_inputs,
        "sample_targets": sample_targets,
        "sample_reconstructions": reconstructions,
        "band_metrics": best_band_metrics,
        "stitching": stitching_results,
        "ecg_physiology": ecg_physiology,
    }


# ---------------------------------------------------------------------------
# Model build & compile
# ---------------------------------------------------------------------------


def build_model(cfg: EcgRvqConfig) -> keras.Model:
    """Build the ECG RVQ autoencoder — 1-D or 2-D spatial depending on domain.

    The encoder/decoder/RVQ submodules are attached to the returned model as
    ``model.encoder``, ``model.decoder``, ``model.vq``.
    """
    data = cfg.data
    mcfg = cfg.model
    prefix_cfg = cfg.training.rvq_prefix_loss
    domain, _ = _build_transform_configs(cfg)

    if domain == "stft":
        logger.info(
            "Building 2-D spatial model for STFT domain (input=%dx%dx2)",
            STFT_PAD_HEIGHT, STFT_PAD_WIDTH,
        )
        _enc, _rvq, _dec, model = build_rvq_autoencoder_2d_spatial(
            input_height=STFT_PAD_HEIGHT,
            input_width=STFT_PAD_WIDTH,
            in_ch=2,
            out_ch=2,
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
        return model

    _enc, _rvq, _dec, model = build_rvq_autoencoder(
        frame_size=data.frame_size,
        in_ch=data.num_leads,
        out_ch=data.num_leads,
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
        use_residual=mcfg.use_residual,
        use_ema=mcfg.use_ema,
        ema_decay=mcfg.ema_decay,
        encoder_type=mcfg.encoder_type,
        expand_ratio=mcfg.expand_ratio,
        causal=mcfg.causal,
        discard_tail=mcfg.discard_tail,
        bottleneck_type=mcfg.bottleneck_type,
        fsq_levels=mcfg.fsq_levels,
        decoder_type=mcfg.decoder_type,
        decoder_state_size=mcfg.decoder_state_size,
        decoder_num_ssm_blocks=mcfg.decoder_num_ssm_blocks,
        hier_detail_scale=mcfg.hier_detail_scale,
        revive_dead_codes=mcfg.revive_dead_codes,
        revive_threshold=mcfg.revive_threshold,
        kmeans_init=mcfg.kmeans_init,
        structured_dropout=mcfg.structured_dropout,
        dropout_levels=mcfg.dropout_levels,
        decoder_activation=mcfg.decoder_activation,
        encoder_blocks_per_stage=mcfg.encoder_blocks_per_stage,
        codebook_sizes=mcfg.codebook_sizes,
        prefix_loss_weights=prefix_cfg.weights if prefix_cfg.enabled else None,
        prefix_loss_target=prefix_cfg.target,
        prefix_loss_lowpass_kernel=prefix_cfg.lowpass_kernel,
        prefix_loss_initial_scale=(
            0.0 if prefix_cfg.enabled and (prefix_cfg.start_epoch > 0 or prefix_cfg.ramp_epochs > 0) else 1.0
        ),
    )
    return model


def build_extra_losses(cfg: EcgRvqConfig) -> list[callable]:
    """Assemble the optional auxiliary losses enabled in *cfg*."""
    data = cfg.data
    extra: list[callable] = []

    dloss = cfg.training.derivative_loss
    if dloss.enabled:
        extra.append(_build_derivative_loss(dloss.weight))
        logger.info("Derivative loss enabled, weight=%.4f", dloss.weight)

    floss = cfg.training.filtered_loss
    if floss.enabled:
        eff_sr = data.effective_sample_rate
        extra.append(
            _build_filtered_mse_loss(
                weight=floss.weight,
                sample_rate=eff_sr,
                cutoff_hz=floss.cutoff_hz,
                num_taps=floss.num_taps,
                num_leads=data.num_leads,
            )
        )
        logger.info(
            "Filtered MSE loss enabled, weight=%.2f, cutoff=%.1f Hz, taps=%d, sr=%d",
            floss.weight, floss.cutoff_hz, floss.num_taps, eff_sr,
        )

    sloss = cfg.training.spectral_loss
    if sloss.enabled:
        extra.append(
            _build_multi_scale_spectral_loss(weight=sloss.weight, fft_sizes=sloss.fft_sizes)
        )
        logger.info(
            "Multi-scale spectral loss enabled, weight=%.2f, fft_sizes=%s",
            sloss.weight, sloss.fft_sizes,
        )

    dwt_cfg = cfg.training.dwt_loss
    if dwt_cfg.enabled:
        extra.append(
            _build_dwt_loss(
                weight=dwt_cfg.weight,
                levels=dwt_cfg.levels,
                band_weights=dwt_cfg.band_weights,
                num_leads=data.num_leads,
            )
        )
        logger.info(
            "DWT subband loss enabled, weight=%.2f, levels=%d, band_weights=%s",
            dwt_cfg.weight, dwt_cfg.levels, dwt_cfg.band_weights,
        )
    return extra


def compile_model(
    model: keras.Model,
    cfg: EcgRvqConfig,
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
# Compression stats & summary assembly
# ---------------------------------------------------------------------------


def build_compression_stats(cfg: EcgRvqConfig) -> dict[str, Any]:
    """Compute the compression ratio / bit budget for *cfg*.

    Accounts for the STFT 2-D spatial case which has a different effective
    downsample factor than the time-domain 1-D path.
    """
    data = cfg.data
    mcfg = cfg.model
    downsample_factor = 2 ** mcfg.num_stages
    domain, _ = _build_transform_configs(cfg)

    if domain == "stft":
        ds_per_dim = 2 ** mcfg.num_stages
        latent_positions = (STFT_PAD_HEIGHT // ds_per_dim) * (STFT_PAD_WIDTH // ds_per_dim)
        stats = compute_compression_stats(
            data.frame_size,
            bit_depth=cfg.evaluation.input_bit_depth,
            num_channels=data.num_leads,
            latent_width=mcfg.latent_width,
            num_levels=mcfg.num_levels,
            downsample_factor=data.frame_size // latent_positions,
            bottleneck_type=mcfg.bottleneck_type,
            fsq_levels=mcfg.fsq_levels,
            codebook_sizes=mcfg.codebook_sizes,
        )
    else:
        stats = compute_compression_stats(
            data.frame_size,
            bit_depth=cfg.evaluation.input_bit_depth,
            num_channels=data.num_leads,
            latent_width=mcfg.latent_width,
            num_levels=mcfg.num_levels,
            downsample_factor=downsample_factor,
            bottleneck_type=mcfg.bottleneck_type,
            fsq_levels=mcfg.fsq_levels,
            codebook_sizes=mcfg.codebook_sizes,
        )
    stats["configured_num_stages"] = mcfg.num_stages
    stats["effective_downsample_factor"] = downsample_factor
    return stats


def assemble_summary(
    cfg: EcgRvqConfig,
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
            "cache_dir": ds_info.get("cache_dir"),
            "synthetic_mix_num_added": ds_info.get("synthetic_added", 0),
            "synthetic_mix_fraction_effective": ds_info.get("synthetic_fraction_effective", 0.0),
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

    stitch_cfg = cfg.evaluation.stitching
    if stitch_cfg.enabled and eval_results.get("stitching") is not None:
        summary["metrics"]["stitching_config"] = stitch_cfg.model_dump()
        summary["metrics"]["stitching"] = eval_results["stitching"]

    if eval_results.get("ecg_physiology") is not None:
        summary["metrics"]["ecg_physiology"] = eval_results["ecg_physiology"]

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
