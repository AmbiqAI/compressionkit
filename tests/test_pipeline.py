"""Tests for the composable 4-stage pipeline and its stages."""

from __future__ import annotations

import numpy as np
import pytest

from compressionkit.pipeline import (
    BandpassPreprocessor,
    DeadzoneQuantizer,
    DeflateEntropy,
    DwtTransform,
    LearnedDenoisePreprocessor,
    PipelineCodec,
    RawEntropy,
    RawTransform,
    ZNormPreprocessor,
    build_dwt_deadzone_codec,
)


@pytest.fixture
def ecg_frame() -> np.ndarray:
    rng = np.random.default_rng(1)
    t = np.arange(512) / 256.0
    sig = np.zeros_like(t)
    for k in range(int(t[-1] * 1.2)):
        sig += np.exp(-((t - k / 1.2) ** 2) / (2 * 0.02**2))
    sig += 0.05 * rng.standard_normal(t.size)
    return sig.astype(np.float32)


def test_znorm_preprocessor_inverts() -> None:
    pre = ZNormPreprocessor()
    frame = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out, ctx = pre.forward(frame)
    assert abs(float(out.mean())) < 1e-5
    np.testing.assert_allclose(pre.inverse(out, ctx), frame, atol=1e-4)


def test_dwt_deadzone_codec_round_trip_and_cr(ecg_frame: np.ndarray) -> None:
    codec = build_dwt_deadzone_codec(modality="ecg", sample_rate=256, frame_size=512, target_cr=8.0)
    enc = codec.encode(ecg_frame)
    recon = codec.decode(enc)
    assert recon.shape == ecg_frame.shape
    raw_bits = 512 * 16
    true_cr = raw_bits / enc.nbits
    assert true_cr == pytest.approx(8.0, rel=0.25)


def test_pipeline_swappable_entropy_slot(ecg_frame: np.ndarray) -> None:
    # Deflate should never use more bits than raw fixed-width packing.
    common = {
        "transform": DwtTransform(),
        "encoder": DeadzoneQuantizer(scale=0.5),
        "modality": "ecg",
        "sample_rate": 256,
        "frame_size": 512,
        "target_cr": 8.0,
        "match_cr": False,
    }
    raw = PipelineCodec(preprocess=ZNormPreprocessor(), entropy=RawEntropy(), **common)
    deflate = PipelineCodec(preprocess=ZNormPreprocessor(), entropy=DeflateEntropy(), **common)
    assert deflate.encode(ecg_frame).nbits <= raw.encode(ecg_frame).nbits


def test_bandpass_vs_raw_transform_shapes(ecg_frame: np.ndarray) -> None:
    codec = PipelineCodec(
        preprocess=BandpassPreprocessor(sample_rate=256),
        transform=RawTransform(),
        encoder=DeadzoneQuantizer(),
        entropy=DeflateEntropy(),
        modality="ecg",
        sample_rate=256,
        frame_size=512,
        target_cr=4.0,
    )
    recon = codec.decode(codec.encode(ecg_frame))
    assert recon.shape == ecg_frame.shape


def test_learned_denoise_preprocessor_with_identity_gain(ecg_frame: np.ndarray) -> None:
    # An identity coeff_denoiser must reproduce the signal through DWT/inverse.
    pre = LearnedDenoisePreprocessor(coeff_denoiser=lambda packed: packed, frame_size=512)
    out, ctx = pre.forward(ecg_frame)
    assert out.shape == ecg_frame.shape
    np.testing.assert_allclose(out, ecg_frame, atol=1e-3)
    # inverse is identity (front-end conditioning)
    np.testing.assert_array_equal(pre.inverse(out, ctx), out)


def test_learned_denoise_rejects_length_change(ecg_frame: np.ndarray) -> None:
    pre = LearnedDenoisePreprocessor(coeff_denoiser=lambda packed: packed[:-1], frame_size=512)
    with pytest.raises(ValueError, match="preserve length"):
        pre.forward(ecg_frame)
