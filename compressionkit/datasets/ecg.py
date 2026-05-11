"""ECG dataset loading, caching, and tf.data pipeline construction.

Loads single-lead ECG segments from HDF5 files, optionally caches to
TFRecords, and builds tf.data pipelines with preprocessing and augmentation
for RVQ training.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import keras
import numpy as np
import physiokit as pk
import tensorflow as tf

# ---------------------------------------------------------------------------
# Single-file loading
# ---------------------------------------------------------------------------

def load_ecg_signal(
    h5_file: Path,
    *,
    lead_index: int = 1,
    leads: list[int] | None = None,
) -> np.ndarray:
    """Load ECG signal(s) from an HDF5 file.

    When *leads* is ``None`` (default), a single lead is extracted and a 1-D
    array of shape ``(samples,)`` is returned.  When *leads* is a list of
    lead indices, a 2-D array of shape ``(samples, num_leads)`` is returned.

    Args:
        h5_file: Path to HDF5 file with a ``"data"`` dataset.
        lead_index: Lead to extract when *leads* is ``None``.
        leads: List of lead indices to extract.  Overrides *lead_index*.

    Returns:
        1-D ``(samples,)`` or 2-D ``(samples, num_leads)`` float32 array.
    """
    with h5py.File(h5_file, "r") as h5:
        data = h5["data"][:]
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        return data
    if data.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D data, got shape {data.shape}")
    # Detect layout: smaller dimension is usually leads
    channel_first = data.shape[0] < data.shape[1]
    if leads is not None:
        if channel_first:
            return data[leads].T  # (C, T) → (T, C)
        else:
            return data[:, leads]  # (T, C)
    if channel_first:
        return data[lead_index]
    else:
        return data[:, lead_index]


def load_ecg_dataset(
    file_paths: Sequence[Path],
    *,
    lead_index: int = 1,
) -> np.ndarray:
    """Load multiple HDF5 files into a stacked numpy array ``[N, T]``.

    Args:
        file_paths: List of HDF5 file paths.
        lead_index: Lead to extract from each file.

    Returns:
        Array of shape ``[N, T]`` with ``float32`` dtype.
    """
    segments: list[np.ndarray] = []
    for path in file_paths:
        try:
            segment = load_ecg_signal(path, lead_index=lead_index)
            segments.append(segment)
        except Exception:
            continue
    if not segments:
        raise RuntimeError("No ECG segments could be loaded; check dataset paths.")
    return np.stack(segments)


# ---------------------------------------------------------------------------
# Train / val / test split helpers
# ---------------------------------------------------------------------------

def load_ecg_splits(
    datasets_dir: Path,
    glob_pattern: str,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
    lead_index: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val/test numpy arrays of single-lead ECG signals.

    Args:
        datasets_dir: Root directory containing dataset files.
        glob_pattern: Glob pattern for HDF5 files relative to *datasets_dir*.
        train_ratio: Fraction of files for training.
        val_ratio: Fraction of files for validation.
        seed: Random seed for reproducible splits.
        lead_index: Lead to extract.

    Returns:
        ``(train_data, val_data, test_data)`` arrays of shape ``[N, T]``.
    """
    files = sorted(Path(datasets_dir).glob(glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"No H5 files found for pattern {glob_pattern} in {datasets_dir}"
        )
    rng = np.random.default_rng(seed)
    files = list(rng.permutation(files))

    n_train = max(1, int(train_ratio * len(files)))
    n_val = max(1, int(val_ratio * len(files)))
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    train_data = load_ecg_dataset(train_files, lead_index=lead_index)
    val_data = load_ecg_dataset(val_files, lead_index=lead_index)
    test_data = load_ecg_dataset(test_files, lead_index=lead_index) if test_files else np.empty((0,))
    return train_data, val_data, test_data


def load_ecg_file_splits(
    datasets_dir: Path,
    glob_pattern: str,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Return train/val/test H5 file splits for subject-level separation."""
    files = sorted(Path(datasets_dir).glob(glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"No H5 files found for pattern {glob_pattern} in {datasets_dir}"
        )
    rng = np.random.default_rng(seed)
    files = list(rng.permutation(files))

    n_train = max(1, int(train_ratio * len(files)))
    n_val = max(1, int(val_ratio * len(files)))
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]
    return train_files, val_files, test_files


# ---------------------------------------------------------------------------
# H5 window sampling
# ---------------------------------------------------------------------------

def _sample_ecg_window_from_h5(
    *,
    h5_path: str,
    lead_index: int,
    window_samples: int,
    leads: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Read one random window from an H5 file.

    Args:
        h5_path: Path to HDF5 file.
        lead_index: Lead to extract (single-lead mode).
        window_samples: Number of samples in the window.
        leads: List of lead indices for multi-lead mode.
        rng: Optional random generator.

    Returns:
        1-D ``(window_samples,)`` or 2-D ``(window_samples, C)`` float32 array.
    """
    signal = load_ecg_signal(Path(h5_path), lead_index=lead_index, leads=leads)
    total = signal.shape[0]

    if total <= window_samples:
        if total < window_samples:
            pad_width = [(0, window_samples - total)]
            if signal.ndim == 2:
                pad_width.append((0, 0))
            signal = np.pad(signal, pad_width, mode="edge")
        return signal[:window_samples].astype(np.float32)

    max_start = total - window_samples
    if rng is None:
        start = int(np.random.randint(0, max_start + 1))
    else:
        start = int(rng.integers(0, max_start + 1))
    return signal[start : start + window_samples].astype(np.float32)


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def _resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample a 1-D signal from *source_rate* to *target_rate* Hz."""
    if source_rate == target_rate:
        return signal
    from scipy.signal import resample

    num_target = round(len(signal) * target_rate / source_rate)
    return resample(signal, num_target).astype(np.float32)


# ---------------------------------------------------------------------------
# Bandpass filter helpers
# ---------------------------------------------------------------------------

def _maybe_filter(
    signal: np.ndarray,
    *,
    sample_rate: int,
    cfg: dict[str, Any] | None,
) -> np.ndarray:
    """Apply an optional bandpass filter to a 1-D signal."""
    if cfg is None or not cfg.get("enabled", False):
        return signal
    return pk.signal.filter_signal(
        signal.astype(np.float32),
        sample_rate=sample_rate,
        lowcut=cfg["low_hz"],
        highcut=cfg["high_hz"],
        order=cfg.get("order", 3),
        forward_backward=cfg.get("forward_backward", True),
    ).astype(np.float32)


def bandpass_filter_batch(
    signals: np.ndarray,
    *,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    order: int = 3,
    forward_backward: bool = True,
) -> np.ndarray:
    """Apply a bandpass filter across a batch of signals.

    Args:
        signals: Input batch with shape ``[N, T]`` or ``[N, T, 1]``.
        sample_rate: Sampling rate in Hz.
        low_hz: Low cutoff frequency in Hz.
        high_hz: High cutoff frequency in Hz.
        order: Filter order.
        forward_backward: Use zero-phase (filtfilt) filtering.

    Returns:
        Filtered signals with the same shape as input.
    """
    if signals.ndim not in (2, 3):
        raise ValueError(f"signals must have shape [N,T] or [N,T,1], got {signals.shape}")

    squeeze_last = signals.ndim == 3
    base = signals[..., 0] if squeeze_last else signals
    filtered = np.vstack(
        [
            pk.signal.filter_signal(
                signal.astype(np.float32),
                sample_rate=sample_rate,
                lowcut=low_hz,
                highcut=high_hz,
                order=order,
                forward_backward=forward_backward,
            ).astype(np.float32)
            for signal in base
        ]
    )
    return filtered[..., np.newaxis] if squeeze_last else filtered


# ---------------------------------------------------------------------------
# TFRecord cache builder
# ---------------------------------------------------------------------------

def _cache_signature(params: dict[str, Any]) -> str:
    """Deterministic hash of cache configuration."""
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def build_ecg_tfrecord_cache(
    *,
    datasets_dir: Path,
    glob_pattern: str,
    cache_root: Path,
    lead_index: int = 1,
    leads: list[int] | None = None,
    segment_samples: int = 5000,
    frame_size: int = 1024,
    shuffle_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    windows_per_subject_train: int = 64,
    windows_per_subject_val: int = 16,
    force_rebuild: bool = False,
    source_sample_rate: int | None = None,
    target_sample_rate: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build a deterministic TFRecord cache of ECG windows.

    Windows are sampled from each subject's H5 file, serialised into
    TFRecord files, and a ``meta.json`` sidecar records provenance.

    When *target_sample_rate* differs from *source_sample_rate*, each window
    is resampled before being written to the cache.  ``segment_samples`` and
    ``frame_size`` are interpreted in **target-rate** terms; the source-rate
    window length is computed automatically.

    Returns:
        ``(cache_dir, meta_dict)`` with paths to the train/val TFRecords.
    """
    need_resample = (
        source_sample_rate is not None
        and target_sample_rate is not None
        and source_sample_rate != target_sample_rate
    )
    # Source-rate window length for H5 sampling
    if need_resample:
        source_window_samples = math.ceil(segment_samples * source_sample_rate / target_sample_rate)
    else:
        source_window_samples = segment_samples

    num_leads = len(leads) if leads is not None else 1

    sig_params = {
        "datasets_dir": str(datasets_dir),
        "glob_pattern": glob_pattern,
        "lead_index": lead_index,
        "segment_samples": segment_samples,
        "frame_size": frame_size,
        "shuffle_seed": shuffle_seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "wps_train": windows_per_subject_train,
        "wps_val": windows_per_subject_val,
    }
    if leads is not None:
        sig_params["leads"] = leads
    if need_resample:
        sig_params["source_sample_rate"] = source_sample_rate
        sig_params["target_sample_rate"] = target_sample_rate
    sig = _cache_signature(sig_params)
    cache_dir = Path(cache_root) / f"ecg_{sig}"
    meta_path = cache_dir / "meta.json"

    if meta_path.exists() and not force_rebuild:
        with meta_path.open("r") as f:
            meta = json.load(f)
        return cache_dir, meta

    cache_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files, _ = load_ecg_file_splits(
        datasets_dir, glob_pattern,
        train_ratio=train_ratio, val_ratio=val_ratio, seed=shuffle_seed,
    )

    def _write_split(
        files: list[Path],
        tfrecord_name: str,
        windows_per_subject: int,
    ) -> int:
        rng = np.random.default_rng(shuffle_seed)
        path = cache_dir / tfrecord_name
        writer = tf.io.TFRecordWriter(str(path))
        total = 0
        for h5_path in files:
            for _ in range(windows_per_subject):
                window = _sample_ecg_window_from_h5(
                    h5_path=str(h5_path),
                    lead_index=lead_index,
                    leads=leads,
                    window_samples=source_window_samples,
                    rng=rng,
                )
                if need_resample:
                    if window.ndim == 1:
                        window = _resample(window, source_sample_rate, target_sample_rate)
                        if len(window) > segment_samples:
                            window = window[:segment_samples]
                        elif len(window) < segment_samples:
                            window = np.pad(window, (0, segment_samples - len(window)), mode="edge")
                    else:
                        # Multi-lead: resample each lead independently
                        resampled = np.stack(
                            [_resample(window[:, c], source_sample_rate, target_sample_rate)
                             for c in range(window.shape[1])],
                            axis=-1,
                        )
                        if resampled.shape[0] > segment_samples:
                            resampled = resampled[:segment_samples]
                        elif resampled.shape[0] < segment_samples:
                            resampled = np.pad(
                                resampled,
                                [(0, segment_samples - resampled.shape[0]), (0, 0)],
                                mode="edge",
                            )
                        window = resampled
                # Flatten for TFRecord storage: (T,) or (T, C) → flat
                feature = {
                    "signal": tf.train.Feature(
                        float_list=tf.train.FloatList(value=window.ravel().tolist())
                    ),
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature)
                )
                writer.write(example.SerializeToString())
                total += 1
        writer.close()
        return total

    train_count = _write_split(train_files, "train.tfrecord", windows_per_subject_train)
    val_count = _write_split(val_files, "val.tfrecord", windows_per_subject_val)

    meta = {
        "signature": sig,
        "params": sig_params,
        "train_tfrecord": "train.tfrecord",
        "val_tfrecord": "val.tfrecord",
        "train_examples": train_count,
        "val_examples": val_count,
        "train_subjects": len(train_files),
        "val_subjects": len(val_files),
        "num_leads": num_leads,
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return cache_dir, meta


# ---------------------------------------------------------------------------
# tf.data pipeline builders
# ---------------------------------------------------------------------------

def _parse_tfrecord_fn(segment_samples: int, num_leads: int = 1):
    """Return a parse function for ECG TFRecord examples."""
    flat_len = segment_samples * num_leads
    def _parse(example_proto):
        feature_spec = {"signal": tf.io.FixedLenFeature([flat_len], tf.float32)}
        parsed = tf.io.parse_single_example(example_proto, feature_spec)
        signal = parsed["signal"]
        if num_leads > 1:
            signal = tf.reshape(signal, [segment_samples, num_leads])
        return signal
    return _parse


def make_ecg_tfrecord_dataset(
    tfrecord_paths: Sequence[Path],
    *,
    frame_size: int,
    segment_samples: int,
    batch_size: int,
    shuffle_buffer_size: int,
    preprocessor: keras.layers.Layer,
    augmenter: keras.layers.Layer | None,
    input_filter_cfg: dict[str, Any] | None = None,
    target_filter_cfg: dict[str, Any] | None = None,
    sample_rate: int = 500,
    num_leads: int = 1,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a tf.data pipeline from cached ECG TFRecords.

    Each example is a 1-D or multi-lead segment; preprocessing (random
    crop + norm) and optional augmentation are applied per-batch.
    """
    files = [str(p) for p in tfrecord_paths]
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(
        _parse_tfrecord_fn(segment_samples, num_leads=num_leads),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)

    if num_leads == 1:
        # Add channel dim: (B, T) -> (B, T, 1)
        ds = ds.map(lambda x: tf.expand_dims(x, -1), num_parallel_calls=tf.data.AUTOTUNE)
    # else: already (B, T, C) from parser

    # Preprocess: random crop + layer norm
    ds = ds.map(lambda x: preprocessor(x, training=True), num_parallel_calls=tf.data.AUTOTUNE)

    # Optional per-batch target filtering via numpy
    if target_filter_cfg is not None and target_filter_cfg.get("enabled", False):
        _tf_low = float(target_filter_cfg["low_hz"])
        _tf_high = float(target_filter_cfg["high_hz"])
        _tf_order = int(target_filter_cfg.get("order", 3))
        _tf_fb = bool(target_filter_cfg.get("forward_backward", True))
        _tf_sr = int(sample_rate)

        def _apply_target_filter(x):
            def _np_filter(batch):
                return bandpass_filter_batch(
                    batch.numpy(),
                    sample_rate=_tf_sr,
                    low_hz=_tf_low,
                    high_hz=_tf_high,
                    order=_tf_order,
                    forward_backward=_tf_fb,
                )
            return tf.py_function(_np_filter, [x], tf.float32)

        ds = ds.map(
            lambda x: (x, _apply_target_filter(x)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        # Set shape back after py_function
        ds = ds.map(
            lambda x, y: (x, tf.ensure_shape(y, x.shape)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)

    # Reshape for autoencoder: (B, frame_size, C) -> (B, 1, frame_size, C)
    reshape = keras.layers.Reshape((1, frame_size, num_leads))

    def _make_pair(inp, target):
        x_aug = augmenter(inp, training=True) if augmenter is not None else inp
        return reshape(x_aug), reshape(target)

    ds = ds.map(_make_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def make_ecg_inmemory_dataset(
    data: np.ndarray,
    *,
    frame_size: int,
    batch_size: int,
    buffer_size: int,
    preprocessor: keras.layers.Layer,
    augmenter: keras.layers.Layer | None,
    target_data: np.ndarray | None = None,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build an in-memory tf.data pipeline for ECG segments.

    Args:
        data: Input data of shape ``[N, T]`` or ``[N, T, 1]``.
        frame_size: Crop size applied by the preprocessor.
        batch_size: Batch size.
        buffer_size: Shuffle buffer size.
        preprocessor: Preprocessing layer (crop + norm).
        augmenter: Optional augmentation layer.
        target_data: If given, separate target array (for filtered targets).
        shuffle: Whether to shuffle.

    Returns:
        tf.data.Dataset yielding ``(input, target)`` batches of shape
        ``(B, 1, frame_size, 1)``.
    """
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    if target_data is None:
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)

        def _apply_prep(x):
            return preprocessor(x, training=True) if preprocessor is not None else x

        def _apply_aug(x):
            return augmenter(x, training=True) if augmenter is not None else x

        dataset = dataset.map(lambda x: _apply_prep(x), num_parallel_calls=tf.data.AUTOTUNE)

        reshape = keras.layers.Reshape((1, frame_size, 1))
        dataset = dataset.map(
            lambda x: (
                reshape(_apply_aug(x)),
                reshape(x),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return dataset.prefetch(tf.data.AUTOTUNE)

    # Separate target
    if target_data.ndim == 2:
        target_data = target_data[:, :, np.newaxis]

    dataset = tf.data.Dataset.from_tensor_slices((data, target_data))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)

    reshape = keras.layers.Reshape((1, frame_size, 1))
    epsilon = 1e-3
    if preprocessor is not None and hasattr(preprocessor, "layers"):
        for layer in preprocessor.layers:
            if hasattr(layer, "epsilon"):
                epsilon = float(layer.epsilon)
                break

    def _rand_crop_pair(x_in, x_tgt):
        max_start = tf.maximum(tf.shape(x_in)[1] - frame_size, 0)
        start = tf.cond(
            max_start > 0,
            lambda: tf.random.uniform(shape=(), minval=0, maxval=max_start + 1, dtype=tf.int32),
            lambda: tf.constant(0, dtype=tf.int32),
        )
        x_in = x_in[:, start : start + frame_size, :]
        x_tgt = x_tgt[:, start : start + frame_size, :]
        return x_in, x_tgt

    def _layer_norm_batch(x):
        mean = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=(1, 2), keepdims=True)
        return (x - mean) / tf.sqrt(var + epsilon)

    def _apply_aug(x):
        return augmenter(x, training=True) if augmenter is not None else x

    dataset = dataset.map(_rand_crop_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda x_in, x_tgt: (_layer_norm_batch(x_in), _layer_norm_batch(x_tgt)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda x_in, x_tgt: (
            reshape(_apply_aug(x_in)),
            reshape(x_tgt),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def make_ecg_stream_dataset(
    file_paths: list[Path],
    *,
    frame_size: int,
    window_samples: int,
    batch_size: int,
    lead_index: int = 1,
    preprocessor: keras.layers.Layer,
    augmenter: keras.layers.Layer | None = None,
    interleave_cycle_length: int = 16,
    subject_buffer_size: int = 512,
    window_buffer_size: int = 20_000,
    windows_per_subject: int = 8,
    input_filter_cfg: dict[str, Any] | None = None,
    target_filter_cfg: dict[str, Any] | None = None,
    seed: int = 42,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a streaming tf.data pipeline that samples windows from H5 files.

    Each subject file yields ``windows_per_subject`` random windows, then
    results are interleaved across subjects.
    """
    rng = np.random.default_rng(seed)

    def _subject_generator(path_bytes):
        path = path_bytes.numpy().decode("utf-8")
        for _ in range(windows_per_subject):
            window = _sample_ecg_window_from_h5(
                h5_path=path,
                lead_index=lead_index,
                window_samples=window_samples,
                rng=rng,
            )
            window = _maybe_filter(window, sample_rate=500, cfg=input_filter_cfg)
            yield window.astype(np.float32)

    paths = [str(p) for p in file_paths]
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        path_ds = path_ds.shuffle(subject_buffer_size, seed=seed, reshuffle_each_iteration=True)

    ds = path_ds.interleave(
        lambda p: tf.data.Dataset.from_generator(
            lambda path_arg=p: _subject_generator(path_arg),
            output_signature=tf.TensorSpec(shape=(window_samples,), dtype=tf.float32),
        ),
        cycle_length=min(interleave_cycle_length, len(paths)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    if shuffle:
        ds = ds.shuffle(window_buffer_size, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: tf.expand_dims(x, -1), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: preprocessor(x, training=True), num_parallel_calls=tf.data.AUTOTUNE)

    reshape = keras.layers.Reshape((1, frame_size, 1))

    def _make_pair(x):
        x_aug = augmenter(x, training=True) if augmenter is not None else x
        return reshape(x_aug), reshape(x)

    ds = ds.map(_make_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def collect_random_samples(
    dataset: tf.data.Dataset,
    sample_count: int,
    rng: np.random.Generator,
    pool_multiplier: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Grab a subset of (input, target) tensors from a tf.data pipeline."""
    pool: list[tuple[np.ndarray, np.ndarray]] = []
    for batch_inputs, batch_targets in dataset:
        batch_x = batch_inputs.numpy()
        batch_y = batch_targets.numpy()
        for idx in range(batch_x.shape[0]):
            pool.append((batch_x[idx], batch_y[idx]))
        if len(pool) >= sample_count * pool_multiplier:
            break
    if not pool:
        raise RuntimeError("Unable to collect any samples from the dataset.")
    effective_count = min(sample_count, len(pool))
    indices = rng.choice(len(pool), size=effective_count, replace=False)
    inputs = np.stack([pool[i][0] for i in indices])
    targets = np.stack([pool[i][1] for i in indices])
    return inputs, targets


__all__ = [
    "bandpass_filter_batch",
    "build_ecg_tfrecord_cache",
    "collect_random_samples",
    "load_ecg_dataset",
    "load_ecg_file_splits",
    "load_ecg_signal",
    "load_ecg_splits",
    "make_ecg_inmemory_dataset",
    "make_ecg_stream_dataset",
    "make_ecg_tfrecord_dataset",
]
