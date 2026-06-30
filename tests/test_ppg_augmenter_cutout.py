"""Tests for paired null augmentation in the PPG training path."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

pytest.importorskip("keras")
pytest.importorskip("helia_edge")

from compressionkit.configs.ppg_rvq import AugmentationConfig
from compressionkit.datasets.null_augmentation import (
    PairedNullAugmentationConfig,
    apply_paired_null_augmentation_batch,
)
from compressionkit.preprocessing.ppg import build_augmenter


def _layer_names(pipeline) -> list[str]:
    return [getattr(layer, "name", layer.__class__.__name__) for layer in pipeline.layers]


def test_augmenter_keeps_cutout_out_of_pipeline() -> None:
    cfg = AugmentationConfig(random_cutout=True, cutout_factor=[0.05, 0.1])
    aug = build_augmenter((0.0, 0.0), aug_cfg=cfg)
    names = _layer_names(aug)
    assert "GaussianNoise" in names
    assert "RandomCutout" not in names


def test_short_paired_cutout_zeros_input_and_target_together() -> None:
    cfg = PairedNullAugmentationConfig(short_cutout_factor=(0.1, 0.2))
    x_in = tf.ones((8, 320, 1), dtype=tf.float32)
    x_tgt = tf.ones((8, 320, 1), dtype=tf.float32)

    y_in, y_tgt = apply_paired_null_augmentation_batch(x_in, x_tgt, cfg)
    y_in = y_in.numpy()
    y_tgt = y_tgt.numpy()

    np.testing.assert_array_equal(y_in == 0.0, y_tgt == 0.0)
    zero_counts = (y_in == 0.0).sum(axis=(1, 2))
    assert np.all(zero_counts >= 32)
    assert np.all(zero_counts <= 64)


def test_long_paired_cutout_creates_large_zero_runs() -> None:
    cfg = PairedNullAugmentationConfig(long_cutout_factor=(0.5, 0.75), long_cutout_prob=1.0)
    x_in = tf.ones((4, 320, 1), dtype=tf.float32)
    x_tgt = tf.ones((4, 320, 1), dtype=tf.float32)

    y_in, y_tgt = apply_paired_null_augmentation_batch(x_in, x_tgt, cfg)
    y_in = y_in.numpy()
    y_tgt = y_tgt.numpy()

    np.testing.assert_array_equal(y_in == 0.0, y_tgt == 0.0)
    zero_counts = (y_in == 0.0).sum(axis=(1, 2))
    assert np.all(zero_counts >= 160)
    assert np.all(zero_counts <= 240)


def test_null_frame_prob_zeroes_entire_frames() -> None:
    cfg = PairedNullAugmentationConfig(null_frame_prob=1.0)
    x_in = tf.random.normal((3, 320, 1), seed=0)
    x_tgt = tf.random.normal((3, 320, 1), seed=1)

    y_in, y_tgt = apply_paired_null_augmentation_batch(x_in, x_tgt, cfg)

    assert np.all(y_in.numpy() == 0.0)
    assert np.all(y_tgt.numpy() == 0.0)
