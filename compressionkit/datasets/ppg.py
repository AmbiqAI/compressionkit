"""PPG dataset loading, caching, and tf.data pipeline construction.

Loads PPG segments from EDF files, optionally caches to TFRecords, and builds
tf.data pipelines with preprocessing and augmentation for RVQ training.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import keras
import numpy as np
import physiokit as pk
import pyedflib
import tensorflow as tf

from compressionkit.datasets.null_augmentation import (
    PairedNullAugmentationConfig,
    apply_paired_null_augmentation_batch,
)

# ---------------------------------------------------------------------------
# Single-file loading
# ---------------------------------------------------------------------------


def load_ppg_signal(
    edf_file: Path,
    *,
    target_rate: int = 64,
    offset_samples: int = 0,
    num_samples: int | None = None,
    target_label: str = "Pleth",
) -> np.ndarray:
    """Load a single PPG signal from an EDF file, resampling to *target_rate*."""
    with pyedflib.EdfReader(str(edf_file)) as reader:
        labels = reader.getSignalLabels()
        if target_label not in labels:
            raise ValueError(f"Channel '{target_label}' not found in {edf_file}")
        idx = labels.index(target_label)
        source_rate = int(reader.samplefrequency(idx))
        total_samples = reader.getNSamples()[idx]
        start = min(offset_samples, max(total_samples - 1, 0))

        read_len = total_samples - start
        if num_samples is not None:
            read_len = min(read_len, math.ceil(num_samples * source_rate / target_rate))

        signal = reader.readSignal(idx, start, read_len, digital=False).astype(np.float32)

    if source_rate != target_rate:
        signal = pk.signal.resample_signal(signal, source_rate, target_rate)

    if num_samples is not None:
        if signal.shape[0] < num_samples:
            raise ValueError(f"Signal {edf_file} shorter ({signal.shape[0]}) than requested {num_samples} samples.")
        signal = signal[:num_samples]

    return signal


def load_ppg_dataset(
    file_paths: Sequence[Path],
    *,
    target_rate: int = 64,
    offset_samples: int = 0,
    num_samples: int | None = None,
    target_label: str = "Pleth",
) -> np.ndarray:
    """Load multiple EDF files into a stacked numpy array ``[N, num_samples]``."""
    segments: list[np.ndarray] = []
    for path in file_paths:
        try:
            segment = load_ppg_signal(
                path,
                target_rate=target_rate,
                offset_samples=offset_samples,
                num_samples=num_samples,
                target_label=target_label,
            )
            segments.append(segment)
        except Exception:
            continue
    if not segments:
        raise RuntimeError("No PPG segments could be loaded; check dataset paths.")
    return np.stack(segments)


# ---------------------------------------------------------------------------
# Train / val / test split helpers
# ---------------------------------------------------------------------------


def load_ppg_splits(
    datasets_dir: Path,
    glob_pattern: str,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
    **load_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val/test numpy arrays of resampled PPG signals."""
    files = sorted(Path(datasets_dir).glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No EDF files found for pattern {glob_pattern} in {datasets_dir}")
    rng = np.random.default_rng(seed)
    files = list(rng.permutation(files))

    n_train = max(1, int(train_ratio * len(files)))
    n_val = max(1, int(val_ratio * len(files)))
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    train_data = load_ppg_dataset(train_files, **load_kwargs)
    val_data = load_ppg_dataset(val_files, **load_kwargs)
    test_data = load_ppg_dataset(test_files, **load_kwargs) if test_files else np.empty((0,))
    return train_data, val_data, test_data


def load_ppg_file_splits(
    datasets_dir: Path,
    glob_pattern: str,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Return train/val/test EDF file splits for subject-level separation."""
    files = sorted(Path(datasets_dir).glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No EDF files found for pattern {glob_pattern} in {datasets_dir}")
    rng = np.random.default_rng(seed)
    files = list(rng.permutation(files))

    n_train = max(1, int(train_ratio * len(files)))
    n_val = max(1, int(val_ratio * len(files)))
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]
    return train_files, val_files, test_files


# ---------------------------------------------------------------------------
# EDF window sampling
# ---------------------------------------------------------------------------


def _sample_ppg_window_from_edf(
    *,
    edf_path: str,
    target_rate: int,
    target_label: str,
    window_samples: int,
    offset_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Read one random window from an EDF file and resample to *target_rate*."""
    with pyedflib.EdfReader(edf_path) as reader:
        labels = reader.getSignalLabels()
        if target_label not in labels:
            raise ValueError(f"Channel '{target_label}' not found in {edf_path}")
        idx = labels.index(target_label)
        source_rate = float(reader.samplefrequency(idx))
        total_samples = int(reader.getNSamples()[idx])
        src_window = max(1, math.ceil(window_samples * source_rate / float(target_rate)))
        src_offset = max(0, math.ceil(offset_samples * source_rate / float(target_rate)))

        max_start = max(0, total_samples - src_window)
        if max_start <= src_offset:
            start = max_start
        elif rng is None:
            start = int(np.random.randint(src_offset, max_start + 1))
        else:
            start = int(rng.integers(src_offset, max_start + 1))
        signal = reader.readSignal(idx, start, src_window, digital=False).astype(np.float32)

    if source_rate != float(target_rate):
        signal = pk.signal.resample_signal(signal, int(source_rate), int(target_rate))
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size < window_samples:
        signal = np.pad(signal, (0, window_samples - int(signal.size)), mode="edge")
    return signal[:window_samples].astype(np.float32)


# ---------------------------------------------------------------------------
# Bandpass filter helpers
# ---------------------------------------------------------------------------


def _maybe_filter(
    signal: np.ndarray,
    *,
    sample_rate: int,
    cfg: dict[str, Any] | None,
) -> np.ndarray:
    """Apply an optional bandpass filter to a 1D signal."""
    if not cfg or not cfg.get("enabled", False):
        return signal
    return pk.signal.filter_signal(
        signal.astype(np.float32),
        sample_rate=sample_rate,
        lowcut=float(cfg.get("low_hz", 0.5)),
        highcut=float(cfg.get("high_hz", 8.0)),
        order=int(cfg.get("order", 3)),
    ).astype(np.float32)


def bandpass_filter_batch(
    signals: np.ndarray,
    *,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    order: int = 3,
) -> np.ndarray:
    """Apply a physiokit bandpass filter across a batch of signals.

    Args:
        signals: Input batch with shape ``[N, T]`` or ``[N, T, 1]``.
        sample_rate: Sampling rate in Hz.
        low_hz: Low cutoff frequency in Hz.
        high_hz: High cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Filtered signals with the same shape as input and ``float32`` dtype.
    """
    if signals.ndim not in (2, 3):
        raise ValueError(f"signals must have shape [N,T] or [N,T,1], got {signals.shape}")

    squeeze_last = signals.ndim == 3
    base = signals[..., 0] if squeeze_last else signals
    filtered = np.vstack(
        [
            pk.signal.filter_signal(
                sig.astype(np.float32),
                sample_rate=sample_rate,
                lowcut=low_hz,
                highcut=high_hz,
                order=order,
            ).astype(np.float32)
            for sig in base
        ]
    )
    return filtered[..., np.newaxis] if squeeze_last else filtered


# ---------------------------------------------------------------------------
# TFRecord cache
# ---------------------------------------------------------------------------


def _build_ppg_cache_signature(
    *,
    datasets_dir: Path,
    glob_pattern: str,
    target_rate: int,
    target_label: str,
    offset_samples: int,
    segment_samples: int,
    frame_size: int,
    shuffle_seed: int,
    train_ratio: float,
    val_ratio: float,
    windows_per_subject_train: int,
    windows_per_subject_val: int,
    file_paths: Sequence[Path],
) -> tuple[dict[str, Any], str]:
    """Create a deterministic cache signature and short hash."""
    signature: dict[str, Any] = {
        "datasets_dir": str(datasets_dir.resolve()),
        "dataset_glob": glob_pattern,
        "sampling_rate": int(target_rate),
        "target_label": str(target_label),
        "offset_samples": int(offset_samples),
        "segment_samples": int(segment_samples),
        "frame_size": int(frame_size),
        "shuffle_seed": int(shuffle_seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "windows_per_subject_train": int(windows_per_subject_train),
        "windows_per_subject_val": int(windows_per_subject_val),
        "source_file_count": len(file_paths),
        "source_files": [str(p) for p in sorted(file_paths)],
    }
    signature_json = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    signature_hash = hashlib.sha1(signature_json.encode("utf-8")).hexdigest()[:12]
    return signature, signature_hash


def _serialize_signal_example(signal: np.ndarray) -> bytes:
    """Serialize one 1D float32 signal into a TF Example."""
    flat = np.asarray(signal, dtype=np.float32).reshape(-1)
    feat = {
        "signal": tf.train.Feature(float_list=tf.train.FloatList(value=flat.tolist())),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feat))
    return example.SerializeToString()


def _write_windows_tfrecord(
    *,
    output_path: Path,
    file_paths: Sequence[Path],
    windows_per_subject: int,
    target_rate: int,
    target_label: str,
    offset_samples: int,
    segment_samples: int,
    rng: np.random.Generator,
) -> int:
    """Write random EDF windows to one TFRecord file and return count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for path in file_paths:
            for _ in range(int(max(1, windows_per_subject))):
                try:
                    signal = _sample_ppg_window_from_edf(
                        edf_path=str(path),
                        target_rate=int(target_rate),
                        target_label=str(target_label),
                        window_samples=int(segment_samples),
                        offset_samples=int(offset_samples),
                        rng=rng,
                    )
                except Exception:
                    continue
                writer.write(_serialize_signal_example(signal))
                total += 1
    return total


def build_ppg_tfrecord_cache(
    *,
    datasets_dir: Path,
    glob_pattern: str,
    cache_root: Path,
    target_rate: int,
    target_label: str,
    offset_samples: int,
    segment_samples: int,
    frame_size: int,
    shuffle_seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    windows_per_subject_train: int = 64,
    windows_per_subject_val: int = 16,
    force_rebuild: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Build a subject-split TFRecord cache for PPG windows.

    Returns:
        Tuple of cache directory path and metadata dictionary.
    """
    all_files = sorted(Path(datasets_dir).glob(glob_pattern))
    if not all_files:
        raise FileNotFoundError(f"No EDF files found for pattern {glob_pattern} in {datasets_dir}")

    signature, signature_hash = _build_ppg_cache_signature(
        datasets_dir=Path(datasets_dir),
        glob_pattern=glob_pattern,
        target_rate=int(target_rate),
        target_label=str(target_label),
        offset_samples=int(offset_samples),
        segment_samples=int(segment_samples),
        frame_size=int(frame_size),
        shuffle_seed=int(shuffle_seed),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        windows_per_subject_train=int(windows_per_subject_train),
        windows_per_subject_val=int(windows_per_subject_val),
        file_paths=all_files,
    )
    cache_dir = Path(cache_root) / (
        f"ppg_sr{int(target_rate)}_frm{int(frame_size)}_seg{int(segment_samples)}_{signature_hash}"
    )
    metadata_path = cache_dir / "metadata.json"

    if metadata_path.exists() and not force_rebuild:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return cache_dir, metadata

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_files, val_files, _ = load_ppg_file_splits(
        Path(datasets_dir),
        glob_pattern,
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        seed=int(shuffle_seed),
    )
    rng = np.random.default_rng(int(shuffle_seed))
    train_path = cache_dir / "train.tfrecord"
    val_path = cache_dir / "val.tfrecord"
    train_examples = _write_windows_tfrecord(
        output_path=train_path,
        file_paths=train_files,
        windows_per_subject=int(windows_per_subject_train),
        target_rate=int(target_rate),
        target_label=str(target_label),
        offset_samples=int(offset_samples),
        segment_samples=int(segment_samples),
        rng=rng,
    )
    val_examples = _write_windows_tfrecord(
        output_path=val_path,
        file_paths=val_files,
        windows_per_subject=int(windows_per_subject_val),
        target_rate=int(target_rate),
        target_label=str(target_label),
        offset_samples=int(offset_samples),
        segment_samples=int(segment_samples),
        rng=rng,
    )
    metadata: dict[str, Any] = {
        "version": 1,
        "signature": signature,
        "signature_hash": signature_hash,
        "train_tfrecord": train_path.name,
        "val_tfrecord": val_path.name,
        "train_examples": int(train_examples),
        "val_examples": int(val_examples),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return cache_dir, metadata


# ---------------------------------------------------------------------------
# tf.data pipeline builders
# ---------------------------------------------------------------------------


def _get_epsilon(preprocessor: keras.layers.Layer | None) -> float:
    """Extract epsilon from preprocessor if available."""
    epsilon = 1e-3
    if preprocessor is not None and hasattr(preprocessor, "layers"):
        for layer in preprocessor.layers:
            if hasattr(layer, "epsilon"):
                epsilon = float(layer.epsilon)
                break
    return epsilon


def make_ppg_inmemory_dataset(
    data: np.ndarray,
    *,
    frame_size: int,
    batch_size: int,
    buffer_size: int,
    preprocessor: keras.layers.Layer | None,
    augmenter: keras.layers.Layer | None,
    target_data: np.ndarray | None = None,
    null_aug_cfg: PairedNullAugmentationConfig | None = None,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data pipeline from in-memory numpy arrays.

    This wraps the same logic as the legacy ``make_ecg_dataset``.
    """
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    if target_data is None:
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)

        def _apply_prep(x: tf.Tensor) -> tf.Tensor:
            return preprocessor(x, training=True) if preprocessor is not None else x

        def _apply_aug(x: tf.Tensor) -> tf.Tensor:
            if augmenter is None:
                return x
            x = tf.ensure_shape(x, [batch_size, frame_size, 1])
            return augmenter(x, training=True)

        dataset = dataset.map(lambda x: _apply_prep(x), num_parallel_calls=tf.data.AUTOTUNE)

        reshape = keras.layers.Reshape((1, frame_size, 1))

        def _make_pair(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            x_aug, x_tgt = apply_paired_null_augmentation_batch(_apply_aug(x), x, null_aug_cfg)
            return reshape(x_aug), reshape(x_tgt)

        dataset = dataset.map(_make_pair, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.prefetch(tf.data.AUTOTUNE)

    # Paired input / target path (for filtered targets)
    if target_data.ndim == 2:
        target_data = target_data[:, :, np.newaxis]
    if target_data.shape != data.shape:
        raise ValueError(f"target_data shape {target_data.shape} must match data shape {data.shape}")

    dataset = tf.data.Dataset.from_tensor_slices((data, target_data))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)

    reshape = keras.layers.Reshape((1, frame_size, 1))
    epsilon = _get_epsilon(preprocessor)

    def _rand_crop_pair(
        x_in: tf.Tensor,
        x_tgt: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        max_start = tf.maximum(tf.shape(x_in)[1] - frame_size, 0)
        start = tf.cond(
            max_start > 0,
            lambda: tf.random.uniform(shape=(), minval=0, maxval=max_start + 1, dtype=tf.int32),
            lambda: tf.constant(0, dtype=tf.int32),
        )
        x_in = x_in[:, start : start + frame_size, :]
        x_tgt = x_tgt[:, start : start + frame_size, :]
        return x_in, x_tgt

    def _layer_norm_batch(x: tf.Tensor) -> tf.Tensor:
        mean = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=(1, 2), keepdims=True)
        return (x - mean) / tf.sqrt(var + epsilon)

    def _apply_aug(x: tf.Tensor) -> tf.Tensor:
        if augmenter is None:
            return x
        x = tf.ensure_shape(x, [batch_size, frame_size, 1])
        return augmenter(x, training=True)

    dataset = dataset.map(_rand_crop_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda x_in, x_tgt: (_layer_norm_batch(x_in), _layer_norm_batch(x_tgt)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    def _augment_pair(x_in: tf.Tensor, x_tgt: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x_aug, x_tgt = apply_paired_null_augmentation_batch(_apply_aug(x_in), x_tgt, null_aug_cfg)
        return reshape(x_aug), reshape(x_tgt)

    dataset = dataset.map(_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def make_ppg_tfrecord_dataset(
    tfrecord_paths: Sequence[Path],
    *,
    frame_size: int,
    segment_samples: int,
    batch_size: int,
    shuffle_buffer_size: int,
    preprocessor: keras.layers.Layer | None,
    augmenter: keras.layers.Layer | None,
    input_filter_cfg: dict[str, Any] | None,
    target_filter_cfg: dict[str, Any] | None,
    sample_rate: int,
    shuffle: bool,
    seed: int = 42,
    null_aug_cfg: PairedNullAugmentationConfig | None = None,
) -> tf.data.Dataset:
    """Build dataset from TFRecord windows compatible with RVQ trainer."""
    if not tfrecord_paths:
        raise ValueError("tfrecord_paths must not be empty")

    segment_samples = int(segment_samples)
    frame_size = int(frame_size)
    sample_rate = int(sample_rate)
    input_filter_cfg = dict(input_filter_cfg or {})
    target_filter_cfg = dict(target_filter_cfg or {})

    ds = tf.data.TFRecordDataset(
        [str(p) for p in tfrecord_paths],
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    if shuffle:
        ds = ds.shuffle(
            buffer_size=max(1, int(shuffle_buffer_size)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    ds = ds.repeat()

    spec = {"signal": tf.io.FixedLenFeature([segment_samples], tf.float32)}

    def _parse(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        rec = tf.io.parse_single_example(x, spec)
        sig = rec["signal"][:, tf.newaxis]
        return sig, sig

    def _maybe_filter_tf(
        x_in: tf.Tensor,
        x_tgt: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if not input_filter_cfg.get("enabled", False) and not target_filter_cfg.get("enabled", False):
            return x_in, x_tgt

        def _filter_np(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            in_sig = np.asarray(a, dtype=np.float32).reshape(-1)
            tgt_sig = np.asarray(b, dtype=np.float32).reshape(-1)
            if input_filter_cfg.get("enabled", False):
                in_sig = _maybe_filter(in_sig, sample_rate=sample_rate, cfg=input_filter_cfg)
            if target_filter_cfg.get("enabled", False):
                tgt_sig = _maybe_filter(tgt_sig, sample_rate=sample_rate, cfg=target_filter_cfg)
            return in_sig[:, np.newaxis].astype(np.float32), tgt_sig[:, np.newaxis].astype(np.float32)

        x_in_f, x_tgt_f = tf.py_function(_filter_np, [x_in, x_tgt], Tout=[tf.float32, tf.float32])
        x_in_f.set_shape((segment_samples, 1))
        x_tgt_f.set_shape((segment_samples, 1))
        return x_in_f, x_tgt_f

    epsilon = _get_epsilon(preprocessor)
    reshape = keras.layers.Reshape((1, frame_size, 1))

    def _rand_crop_pair(
        x_in: tf.Tensor,
        x_tgt: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        max_start = tf.maximum(tf.shape(x_in)[0] - frame_size, 0)
        start = tf.cond(
            max_start > 0,
            lambda: tf.random.uniform(shape=(), minval=0, maxval=max_start + 1, dtype=tf.int32),
            lambda: tf.constant(0, dtype=tf.int32),
        )
        x_in = x_in[start : start + frame_size, :]
        x_tgt = x_tgt[start : start + frame_size, :]
        return x_in, x_tgt

    def _layer_norm_batch(x: tf.Tensor) -> tf.Tensor:
        mean = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=(1, 2), keepdims=True)
        return (x - mean) / tf.sqrt(var + epsilon)

    def _apply_aug(x: tf.Tensor) -> tf.Tensor:
        if augmenter is None:
            return x
        x = tf.ensure_shape(x, [batch_size, frame_size, 1])
        return augmenter(x, training=True)

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_maybe_filter_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_rand_crop_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda x_in, x_tgt: (_layer_norm_batch(x_in), _layer_norm_batch(x_tgt)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    def _augment_pair(x_in: tf.Tensor, x_tgt: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x_aug, x_tgt = apply_paired_null_augmentation_batch(_apply_aug(x_in), x_tgt, null_aug_cfg)
        return reshape(x_aug), reshape(x_tgt)

    ds = ds.map(_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def make_ppg_stream_dataset(
    file_paths: Sequence[Path],
    *,
    frame_size: int,
    window_samples: int,
    batch_size: int,
    subject_buffer_size: int,
    window_buffer_size: int,
    windows_per_subject: int,
    interleave_cycle_length: int,
    target_rate: int,
    target_label: str,
    offset_samples: int,
    preprocessor: keras.layers.Layer | None,
    augmenter: keras.layers.Layer | None,
    input_filter_cfg: dict[str, Any] | None,
    target_filter_cfg: dict[str, Any] | None,
    shuffle: bool,
    seed: int = 42,
    null_aug_cfg: PairedNullAugmentationConfig | None = None,
) -> tf.data.Dataset:
    """Build a streaming tf.data dataset that samples random windows per subject."""
    if not file_paths:
        raise ValueError("file_paths must not be empty for streaming PPG dataset")

    paths = tf.constant([str(p) for p in file_paths], dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=max(1, int(subject_buffer_size)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    ds = ds.repeat()

    windows_per_subject = max(1, int(windows_per_subject))
    ds = ds.interleave(
        lambda p: tf.data.Dataset.from_tensors(p).repeat(windows_per_subject),
        cycle_length=max(1, int(interleave_cycle_length)),
        block_length=1,
        num_parallel_calls=1,
        deterministic=True,
    )

    input_filter_cfg = dict(input_filter_cfg or {})
    target_filter_cfg = dict(target_filter_cfg or {})
    offset_samples = int(max(0, offset_samples))
    window_samples = int(window_samples)
    target_rate = int(target_rate)
    target_label = str(target_label)

    def _sample_pair_np(path_bytes: tf.Tensor) -> tuple[np.ndarray, np.ndarray]:
        path = path_bytes.numpy().decode("utf-8")
        base = _sample_ppg_window_from_edf(
            edf_path=path,
            target_rate=target_rate,
            target_label=target_label,
            window_samples=window_samples,
            offset_samples=offset_samples,
        )
        x_in = _maybe_filter(base, sample_rate=target_rate, cfg=input_filter_cfg)
        x_tgt = _maybe_filter(base, sample_rate=target_rate, cfg=target_filter_cfg)
        return x_in[:, np.newaxis], x_tgt[:, np.newaxis]

    def _sample_pair_tf(path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x_in, x_tgt = tf.py_function(_sample_pair_np, [path], Tout=[tf.float32, tf.float32])
        x_in.set_shape((window_samples, 1))
        x_tgt.set_shape((window_samples, 1))
        return x_in, x_tgt

    # pyedflib is not thread-safe for concurrent opens on the same EDF path.
    ds = ds.map(_sample_pair_tf, num_parallel_calls=1, deterministic=True)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=max(1, int(window_buffer_size)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)

    epsilon = _get_epsilon(preprocessor)
    reshape = keras.layers.Reshape((1, frame_size, 1))

    def _rand_crop_pair(
        x_in: tf.Tensor,
        x_tgt: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        max_start = tf.maximum(tf.shape(x_in)[1] - frame_size, 0)
        start = tf.cond(
            max_start > 0,
            lambda: tf.random.uniform(shape=(), minval=0, maxval=max_start + 1, dtype=tf.int32),
            lambda: tf.constant(0, dtype=tf.int32),
        )
        x_in = x_in[:, start : start + frame_size, :]
        x_tgt = x_tgt[:, start : start + frame_size, :]
        return x_in, x_tgt

    def _layer_norm_batch(x: tf.Tensor) -> tf.Tensor:
        mean = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=(1, 2), keepdims=True)
        return (x - mean) / tf.sqrt(var + epsilon)

    def _apply_aug(x: tf.Tensor) -> tf.Tensor:
        if augmenter is None:
            return x
        x = tf.ensure_shape(x, [batch_size, frame_size, 1])
        return augmenter(x, training=True)

    ds = ds.map(_rand_crop_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda x_in, x_tgt: (_layer_norm_batch(x_in), _layer_norm_batch(x_tgt)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    def _augment_pair(x_in: tf.Tensor, x_tgt: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x_aug, x_tgt = apply_paired_null_augmentation_batch(_apply_aug(x_in), x_tgt, null_aug_cfg)
        return reshape(x_aug), reshape(x_tgt)

    ds = ds.map(_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


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
    "build_ppg_tfrecord_cache",
    "collect_random_samples",
    "load_ppg_dataset",
    "load_ppg_file_splits",
    "load_ppg_signal",
    "load_ppg_splits",
    "make_ppg_inmemory_dataset",
    "make_ppg_stream_dataset",
    "make_ppg_tfrecord_dataset",
]
