"""Utility functions for loading ECG datasets and building preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import h5py
import keras
import helia_edge as helia
import numpy as np
import physiokit as pk
import tensorflow as tf


def load_h5_arrays(file_paths: Sequence[Path]) -> np.ndarray:
    """Load ECG arrays from a list of HDF5 files."""
    arrays = []
    for file_path in file_paths:
        with h5py.File(file_path, "r") as h5:
            arrays.append(h5["data"][:])
    if not arrays:
        raise ValueError("No data loaded; check file paths.")
    return np.concatenate(arrays, axis=0)


def load_dataset_splits(
    datasets_dir: Path,
    glob_pattern: str,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test numpy arrays loaded from HDF5 ECG files."""
    files = sorted(Path(datasets_dir).glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No dataset files found with pattern {glob_pattern} under {datasets_dir}")
    rng = np.random.default_rng(seed)
    files = list(rng.permutation(files))
    n_train = max(1, int(train_ratio * len(files)))
    n_val = max(1, int(val_ratio * len(files)))
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]
    return (
        load_h5_arrays(train_files),
        load_h5_arrays(val_files),
        load_h5_arrays(test_files) if test_files else np.empty((0,)),
    )


def build_preprocessor(frame_size: int, epsilon: float = 1e-3) -> keras.layers.Layer:
    """Create the preprocessing pipeline applied before augmentation."""
    return helia.layers.preprocessing.AugmentationPipeline(
        layers=[
            helia.layers.preprocessing.RandomCrop1D(duration=frame_size, name="RandomCrop"),
            helia.layers.preprocessing.LayerNormalization1D(epsilon=epsilon, name="LayerNorm"),
        ]
    )


def build_augmenter(noise_factor=(0.01, 0.1)) -> keras.layers.Layer:
    """Return a simple augmentation pipeline for ECG signals."""
    return helia.layers.preprocessing.AugmentationPipeline(
        layers=[
            helia.layers.preprocessing.RandomGaussianNoise1D(factor=noise_factor, name="GaussianNoise"),
        ]
    )


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
        signals: Input batch with shape [N, T] or [N, T, 1].
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
                signal.astype(np.float32),
                sample_rate=sample_rate,
                lowcut=low_hz,
                highcut=high_hz,
                order=order,
            ).astype(np.float32)
            for signal in base
        ]
    )
    return filtered[..., np.newaxis] if squeeze_last else filtered


def make_ecg_dataset(
    data: np.ndarray,
    *,
    frame_size: int,
    batch_size: int,
    buffer_size: int,
    preprocessor: Optional[keras.layers.Layer],
    augmenter: Optional[keras.layers.Layer],
    target_data: np.ndarray | None = None,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Convert numpy arrays into a tf.data pipeline with augmentation."""
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

    if target_data.ndim == 2:
        target_data = target_data[:, :, np.newaxis]
    if target_data.shape != data.shape:
        raise ValueError(f"target_data shape {target_data.shape} must match data shape {data.shape}")

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


def collect_random_samples(
    dataset: tf.data.Dataset,
    sample_count: int,
    rng: np.random.Generator,
    pool_multiplier: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Grab a subset of (input, target) tensors from a tf.data pipeline."""
    pool = []
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
    "load_dataset_splits",
    "build_preprocessor",
    "build_augmenter",
    "bandpass_filter_batch",
    "make_ecg_dataset",
    "collect_random_samples",
]
