"""Tests for the wide artifact augmenter and the v2 wavelet denoiser."""

from __future__ import annotations

import numpy as np

from compressionkit.models.wavelet_denoiser import (
    as_coeff_denoiser,
    build_wavelet_denoiser_v2,
    build_wavelet_gain_denoiser,
)
from compressionkit.synthetic.artifact_augment import WideArtifactAugmenter
from compressionkit.synthetic.ecg_mcsharry import ecg_mcsharry


def _clean_batch(n: int = 24) -> np.ndarray:
    return np.stack([ecg_mcsharry(2.0, 256, hr_mean=70, seed=i)[:512] for i in range(n)]).astype(np.float32)


def test_v2_denoiser_near_identity_at_init() -> None:
    model = build_wavelet_denoiser_v2(frame_size=512, width=48)
    assert model.output_names[0] == "denoised"
    den = as_coeff_denoiser(model, frame_size=512, wavelet="bior4.4", levels=6)
    x = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    # Near-identity initialization (gain ~ 1, residual ~ 0).
    assert float(np.mean(np.abs(den(x) - x))) < 0.1


def test_v2_has_more_capacity_than_gain_net() -> None:
    v1 = build_wavelet_gain_denoiser(frame_size=512)
    v2 = build_wavelet_denoiser_v2(frame_size=512)
    assert v2.count_params() > v1.count_params()


def test_as_coeff_denoiser_detects_gain_vs_direct() -> None:
    # gain model multiplies; direct model returns coeffs. Both preserve length.
    gain = build_wavelet_gain_denoiser(frame_size=512)
    direct = build_wavelet_denoiser_v2(frame_size=512)
    x = np.random.default_rng(1).standard_normal(512).astype(np.float32)
    for model in (gain, direct):
        out = as_coeff_denoiser(model, frame_size=512, wavelet="bior4.4", levels=6)(x)
        assert out.shape == x.shape


def test_wide_augmenter_is_not_bimodal() -> None:
    # Severities should spread continuously, not cluster at {0, high}.
    aug = WideArtifactAugmenter(sample_rate=256, clean_prob=0.08)
    rng = np.random.default_rng(0)
    sevs = np.array([aug._sample_severity(rng) for _ in range(2000)])
    # mass in the mid-range (0.2, 0.8) — a bimodal {clean-spike, high} would be near-empty here
    mid_fraction = float(np.mean((sevs > 0.2) & (sevs < 0.8)))
    assert mid_fraction > 0.3
    assert sevs.min() < 0.05 and sevs.max() > 0.8


def test_wide_augmenter_pairs_shapes_and_scale() -> None:
    clean = _clean_batch(16)
    target, noisy = WideArtifactAugmenter(sample_rate=256).corrupt_batch(clean, seed=0)
    assert target.shape == clean.shape == noisy.shape
    # both unit-normalized per window
    assert abs(float(target.std()) - 1.0) < 0.2
    assert abs(float(noisy.std()) - 1.0) < 0.2
