"""Tests for :mod:`compressionkit.evaluation.stitching_report`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compressionkit.evaluation import (
    IdentityCodec,
    SpihtAcCodec,
    codec_predict_fn,
    compare_stitching_methods,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _long_ppg(n_samples: int = 4096, fs: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    sig = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t) + 0.05 * rng.standard_normal(n_samples)
    return ((sig - sig.mean()) / (sig.std() + 1e-6)).astype(np.float32)


def test_compare_stitching_identity_returns_zero_prd() -> None:
    codec = IdentityCodec(name="identity", modality="ppg", sample_rate=64, frame_size=320)
    sig = _long_ppg(n_samples=2048)
    df = compare_stitching_methods(codec, sig, hop_ratios=(0.5,))
    assert isinstance(df, pd.DataFrame)
    # Identity codec should round-trip perfectly for every method.
    assert (df["prd"] < 1e-3).all()


def test_compare_stitching_columns_and_methods() -> None:
    codec = IdentityCodec(name="identity", modality="ppg", sample_rate=64, frame_size=320)
    sig = _long_ppg(n_samples=2048)
    df = compare_stitching_methods(codec, sig, hop_ratios=(0.25, 0.5, 0.75))
    expected_cols = {
        "method",
        "hop_ratio",
        "prd",
        "seam_ratio",
        "seam_rms",
        "non_seam_rms",
        "num_seams",
        "n_samples",
    }
    assert expected_cols.issubset(df.columns)
    # All four methods appear.
    assert set(df["method"]) == {"hard_concat", "overlap_add", "linear_crossfade", "tukey_overlap_add"}
    # hard_concat reported once (hop_ratio = NaN), others 3x for the 3 hops.
    assert (df["method"] == "hard_concat").sum() == 1
    assert (df["method"] == "overlap_add").sum() == 3


def test_compare_stitching_signal_too_short_raises() -> None:
    codec = IdentityCodec(name="identity", modality="ppg", sample_rate=64, frame_size=320)
    with pytest.raises(ValueError):
        compare_stitching_methods(codec, np.zeros(100, dtype=np.float32))


def test_compare_stitching_unknown_method_raises() -> None:
    codec = IdentityCodec(name="identity", modality="ppg", sample_rate=64, frame_size=320)
    with pytest.raises(ValueError):
        compare_stitching_methods(codec, _long_ppg(), methods=["nope"])


def test_codec_predict_fn_shape_check() -> None:
    codec = IdentityCodec(name="identity", modality="ppg", sample_rate=64, frame_size=320)
    predict = codec_predict_fn(codec)
    bad = np.zeros((2, 320), dtype=np.float32)
    with pytest.raises(ValueError):
        predict(bad)


def test_compare_stitching_spiht_produces_finite_metrics() -> None:
    # SPIHT introduces real quantization error; just verify every row has a
    # finite PRD and seam metric — the comparative ordering between methods
    # is codec-specific and is what the customer-facing report exists to
    # surface (so we deliberately don't assert a fixed ordering here).
    codec = SpihtAcCodec(
        name="spiht_4x",
        modality="ppg",
        sample_rate=64,
        frame_size=320,
        target_cr=4.0,
    )
    sig = _long_ppg(n_samples=2048)
    df = compare_stitching_methods(codec, sig, hop_ratios=(0.5,))
    assert np.all(np.isfinite(df["prd"]))
    assert np.all(np.isfinite(df["seam_ratio"]))
    # Sanity: SPIHT 4x is lossy, so PRD must be > 0.
    assert (df["prd"] > 0.1).all()
