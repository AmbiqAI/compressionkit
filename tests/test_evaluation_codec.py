"""Tests for the uniform Codec interface and reference adapters."""

from __future__ import annotations

import numpy as np
import pytest

from compressionkit.evaluation.codec import (
    Codec,
    EncodedFrame,
    IdentityCodec,
    SpihtAcCodec,
    compression_ratio,
)


@pytest.fixture
def ppg_frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(320) / 64.0
    sig = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t) + 0.05 * rng.standard_normal(320)
    return sig.astype(np.float32)


def test_identity_codec_round_trip(ppg_frame: np.ndarray) -> None:
    codec = IdentityCodec(frame_size=320, sample_rate=64)
    assert isinstance(codec, Codec)
    enc = codec.encode(ppg_frame)
    assert isinstance(enc, EncodedFrame)
    assert enc.nbits == 320 * 32
    recon = codec.decode(enc)
    np.testing.assert_array_equal(recon, ppg_frame)


def test_compression_ratio_identity(ppg_frame: np.ndarray) -> None:
    codec = IdentityCodec(frame_size=320)
    enc = codec.encode(ppg_frame)
    # 320 * 16 raw bits / (320 * 32) encoded bits = 0.5
    assert compression_ratio(codec, enc, bits_per_sample=16) == pytest.approx(0.5)


def test_compression_ratio_handles_zero_bits() -> None:
    enc = EncodedFrame(payload=b"", nbits=0)
    codec = IdentityCodec(frame_size=320)
    assert compression_ratio(codec, enc) == float("inf")


@pytest.mark.parametrize("target_cr", [4.0, 8.0, 16.0])
def test_spiht_ac_codec_round_trip(ppg_frame: np.ndarray, target_cr: float) -> None:
    codec = SpihtAcCodec(
        modality="ppg",
        sample_rate=64,
        frame_size=320,
        target_cr=target_cr,
        wavelet="bior4.4",
        levels=5,
        use_ac=True,
    )
    assert isinstance(codec, Codec)
    enc = codec.encode(ppg_frame)
    # Bit budget respected
    assert enc.nbits <= codec.max_bits + 8  # last byte may pad
    recon = codec.decode(enc)
    assert recon.shape == ppg_frame.shape
    assert recon.dtype == np.float32
    # Reconstruction PRD should be reasonable at these CRs
    err = ppg_frame - recon
    prd = 100.0 * np.sqrt(np.sum(err**2) / (np.sum(ppg_frame**2) + 1e-12))
    assert prd < 60.0, f"PRD {prd:.1f}% too high at CR {target_cr}"


def test_spiht_ac_codec_metadata_required() -> None:
    codec = SpihtAcCodec(frame_size=320, target_cr=8.0)
    enc = codec.encode(np.zeros(320, dtype=np.float32) + 0.1)
    # Strip metadata
    bad = EncodedFrame(payload=enc.payload, nbits=enc.nbits, side={})
    with pytest.raises(ValueError, match="metadata"):
        codec.decode(bad)


def test_spiht_ac_codec_rejects_multichannel() -> None:
    codec = SpihtAcCodec(frame_size=320, target_cr=8.0)
    with pytest.raises(ValueError, match="1-D"):
        codec.encode(np.zeros((320, 3), dtype=np.float32))


def test_spiht_ac_invalid_cr_raises() -> None:
    with pytest.raises(ValueError):
        SpihtAcCodec(frame_size=320, target_cr=1e9)


def test_codec_protocol_runtime_check() -> None:
    """Both reference codecs must satisfy the runtime-checkable protocol."""
    assert isinstance(IdentityCodec(), Codec)
    assert isinstance(SpihtAcCodec(), Codec)
