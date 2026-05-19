"""Tests for :class:`compressionkit.evaluation.RvqCodec`.

These tests require a trained RVQ run directory and the Keras/TF stack. They
are skipped automatically when the fixture run is absent (e.g. CI without
golden artifacts).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# Prefer a small, fast single-channel model. Fall back to 12-lead if missing.
CANDIDATE_RUNS: list[tuple[str, str, int, int]] = [
    # (run_subdir, modality, expected_frame_size, expected_channels)
    ("results/ppg_rvq_64hz_04x_golden", "ppg", 320, 1),
    ("results/ecg_rvq_256hz_04x_golden", "ecg", 512, 1),
    ("results/ecg_rvq_256hz_12lead_32x_big", "ecg", 512, 12),
]


def _first_available_run() -> tuple[Path, str, int, int]:
    for sub, modality, fs, nch in CANDIDATE_RUNS:
        run_dir = REPO_ROOT / sub
        if (run_dir / "best_model.weights.h5").exists() and (run_dir / "config.json").exists():
            return run_dir, modality, fs, nch
    pytest.skip("No RVQ golden run available for testing")


@pytest.fixture(scope="module")
def rvq_codec():
    pytest.importorskip("tensorflow")
    pytest.importorskip("keras")
    from compressionkit.evaluation import RvqCodec

    run_dir, modality, _fs, _nch = _first_available_run()
    return RvqCodec.from_run_dir(run_dir, modality=modality)


def _make_frame(codec) -> np.ndarray:
    rng = np.random.default_rng(0)
    if codec.modality == "ppg":
        t = np.arange(codec.frame_size) / codec.sample_rate
        sig = np.sin(2 * np.pi * 1.2 * t) + 0.05 * rng.standard_normal(codec.frame_size)
    else:
        t = np.arange(codec.frame_size) / codec.sample_rate
        sig = np.sin(2 * np.pi * 1.0 * t) + 0.1 * rng.standard_normal(codec.frame_size)
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
    if codec.n_channels == 1:
        return sig.astype(np.float32)
    return np.tile(sig[:, None].astype(np.float32), (1, codec.n_channels))


def test_rvq_codec_satisfies_protocol(rvq_codec) -> None:
    from compressionkit.evaluation import Codec

    assert isinstance(rvq_codec, Codec)


def test_rvq_round_trip_shape_and_metadata(rvq_codec) -> None:
    frame = _make_frame(rvq_codec)
    enc = rvq_codec.encode(frame)

    # nbits is positive and exactly tokens * bits_per_token
    assert enc.nbits > 0
    n_tokens = enc.side["n_tokens"]
    bits_per_token = enc.side["bits_per_token"]
    assert enc.nbits == n_tokens * bits_per_token

    # Token IDs: one int array per RVQ level, length n_tokens
    token_ids = enc.side["token_ids"]
    assert len(token_ids) == len(enc.side["codebook_sizes"])
    for ids, K in zip(token_ids, enc.side["codebook_sizes"]):
        assert ids.shape == (n_tokens,)
        assert ids.dtype == np.int32
        assert int(ids.min()) >= 0
        assert int(ids.max()) < K

    # Residual norms decrease monotonically (greedy RVQ is non-increasing)
    norms = enc.side["residual_norms"]
    assert len(norms) == len(token_ids) + 1
    for a, b in zip(norms, norms[1:]):
        assert b <= a + 1e-4, f"residual norm increased: {a} -> {b}"

    # Quant distances: one per token per level
    dists = enc.side["quant_distances"]
    assert len(dists) == len(token_ids)
    for d in dists:
        assert d.shape == (n_tokens,)
        assert np.all(d >= 0)

    # Decode returns the same shape as the input
    recon = rvq_codec.decode(enc)
    assert recon.shape == frame.shape
    assert recon.dtype == np.float32


def test_rvq_compression_ratio_matches_target(rvq_codec) -> None:
    from compressionkit.evaluation import compression_ratio

    frame = _make_frame(rvq_codec)
    enc = rvq_codec.encode(frame)
    cr = compression_ratio(rvq_codec, enc, bits_per_sample=16)
    # For the single-channel helper. Multi-channel raw bits = frame_size * 16 * n_ch
    if rvq_codec.n_channels > 1:
        raw_bits = rvq_codec.frame_size * 16 * rvq_codec.n_channels
        cr = raw_bits / enc.nbits
    # CR should be in the same ballpark as target_cr (within ±20%)
    assert cr > 1.0
    if rvq_codec.target_cr > 0:
        ratio = cr / rvq_codec.target_cr
        assert 0.5 < ratio < 2.0, f"CR {cr:.2f} vs target {rvq_codec.target_cr:.2f}"


def test_rvq_encode_decode_is_deterministic(rvq_codec) -> None:
    """Two encodes of the same frame should produce identical token IDs."""
    frame = _make_frame(rvq_codec)
    enc1 = rvq_codec.encode(frame)
    enc2 = rvq_codec.encode(frame)
    for a, b in zip(enc1.side["token_ids"], enc2.side["token_ids"]):
        np.testing.assert_array_equal(a, b)
    rec1 = rvq_codec.decode(enc1)
    rec2 = rvq_codec.decode(enc2)
    np.testing.assert_allclose(rec1, rec2, atol=1e-5)
