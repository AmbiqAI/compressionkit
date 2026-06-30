"""Tests for the contact-artifact bank and in-graph RVQ artifact augmenter."""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from compressionkit.configs.ecg_rvq import AugmentationConfig
from compressionkit.datasets.ecg import make_ecg_inmemory_dataset
from compressionkit.preprocessing.ecg import RandomArtifactNoise1D, build_augmenter
from compressionkit.synthetic.artifact_augment import build_artifact_bank


def test_build_artifact_bank_shape_and_finite() -> None:
    bank = build_artifact_bank(64, 512, 256.0, seed=0)
    assert bank.shape == (64, 512)
    assert np.isfinite(bank).all()
    # rows are unit-normalized artifact waveforms
    assert abs(float(bank.std()) - 1.0) < 0.3


def test_artifact_layer_preserves_shape_and_corrupts() -> None:
    bank = build_artifact_bank(64, 512, 256.0, seed=1)
    layer = RandomArtifactNoise1D(artifact_bank=bank, clean_prob=0.0, seed=3)
    batch = np.random.default_rng(0).standard_normal((8, 512, 1)).astype(np.float32)
    out = layer(tf.constant(batch), training=True).numpy()
    assert out.shape == batch.shape
    assert np.isfinite(out).all()
    # corruption actually changed the signal
    assert float(np.mean(np.abs(out - batch))) > 1e-3


def test_artifact_layer_identity_when_eval() -> None:
    bank = build_artifact_bank(16, 512, 256.0, seed=2)
    layer = RandomArtifactNoise1D(artifact_bank=bank, seed=4)
    batch = np.random.default_rng(0).standard_normal((4, 512, 1)).astype(np.float32)
    out = layer(tf.constant(batch), training=False).numpy()
    np.testing.assert_array_equal(out, batch)


def test_fraction_distribution_is_continuous() -> None:
    bank = build_artifact_bank(8, 512, 256.0, seed=0)
    layer = RandomArtifactNoise1D(artifact_bank=bank, clean_prob=0.08, seed=5)
    fracs = np.array([layer._sample_fraction() for _ in range(2000)])
    # spread across the mid-range (a bimodal {clean, severe} split would be empty here)
    assert float(np.mean((fracs > 0.2) & (fracs < 0.8))) > 0.3
    assert fracs.min() < 0.05


def test_build_augmenter_wires_artifact_layer() -> None:
    cfg = AugmentationConfig(artifact_noise_enabled=True, gaussian_noise=[0.0, 0.0])
    bank = build_artifact_bank(32, 512, 256.0, seed=0)
    aug = build_augmenter(aug_cfg=cfg, sample_rate=256, artifact_bank=bank)
    batch = np.random.default_rng(0).standard_normal((4, 512, 1)).astype(np.float32)
    out = np.asarray(aug(tf.constant(batch), training=True))
    assert out.shape == batch.shape
    assert float(np.mean(np.abs(out - batch))) > 1e-3


def test_inmemory_dataset_keeps_clean_targets_with_artifact_augmentation() -> None:
    cfg = AugmentationConfig(
        artifact_noise_enabled=True,
        artifact_clean_prob=0.0,
        gaussian_noise=[0.0, 0.0],
        empirical_noise_prob=0.0,
    )
    bank = build_artifact_bank(32, 128, 256.0, seed=7)
    aug = build_augmenter(aug_cfg=cfg, sample_rate=256, artifact_bank=bank)
    clean = np.random.default_rng(11).standard_normal((4, 128, 1)).astype(np.float32)

    ds = make_ecg_inmemory_dataset(
        clean,
        frame_size=128,
        batch_size=4,
        buffer_size=4,
        preprocessor=keras.layers.Identity(),
        augmenter=aug,
        shuffle=False,
    )

    batch_inputs, batch_targets = next(iter(ds))
    batch_inputs = np.asarray(batch_inputs)[:, 0, :, :]
    batch_targets = np.asarray(batch_targets)[:, 0, :, :]

    np.testing.assert_allclose(batch_targets, clean, atol=1e-6)
    assert float(np.mean(np.abs(batch_inputs - batch_targets))) > 1e-3
