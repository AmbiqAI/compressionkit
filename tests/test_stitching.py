"""Tests for the frame-stitching primitives."""

from __future__ import annotations

import numpy as np
import pytest

from compressionkit.evaluation.stitching import (
    STITCH_METHODS,
    reconstruct_hard_concat,
    reconstruct_linear_crossfade,
    reconstruct_overlap_add,
    reconstruct_tukey_overlap_add,
    seam_discontinuity_ratio,
    stitch,
)

FRAME = 64


def identity_predict(batch: np.ndarray) -> np.ndarray:
    """Pass-through predictor: reconstruction = input."""
    return batch


def test_registry_keys_are_stable() -> None:
    assert set(STITCH_METHODS) == {
        "hard_concat",
        "overlap_add",
        "linear_crossfade",
        "tukey_overlap_add",
    }


def test_stitch_dispatcher_unknown_method() -> None:
    with pytest.raises(ValueError):
        stitch("bogus", identity_predict, np.zeros(256, dtype=np.float32), FRAME)


@pytest.mark.parametrize(
    "fn",
    [reconstruct_overlap_add, reconstruct_linear_crossfade, reconstruct_tukey_overlap_add],
)
def test_identity_predict_reconstructs_smooth_signal(fn) -> None:
    rng = np.random.default_rng(0)
    # Smooth signal (low-frequency sinusoid) so layer-norm round-trip is stable
    t = np.linspace(0, 8 * np.pi, 1024, dtype=np.float32)
    sig = 0.3 * np.sin(t) + 0.05 * rng.standard_normal(t.shape).astype(np.float32)

    recon = fn(identity_predict, sig, FRAME, hop_ratio=0.5)
    assert recon.shape == sig.shape
    # Trim frame-length edges where OLA ramp is not in full coverage
    inner = slice(FRAME, -FRAME)
    np.testing.assert_allclose(recon[inner], sig[inner], atol=1e-3)


def test_hard_concat_is_exact_for_identity() -> None:
    sig = np.linspace(-1.0, 1.0, 4 * FRAME, dtype=np.float32)
    recon = reconstruct_hard_concat(identity_predict, sig, FRAME)
    np.testing.assert_allclose(recon, sig, atol=1e-5)


def test_seam_metric_smooth_signal_ratio_near_one() -> None:
    t = np.linspace(0, 6 * np.pi, 2048, dtype=np.float32)
    sig = np.sin(t)
    stats = seam_discontinuity_ratio(sig, frame_size=FRAME, hop_ratio=0.5, radius=4)
    # A purely smooth signal has no special structure at seam locations,
    # so ratio should be close to 1.
    assert 0.5 < stats["ratio"] < 2.0
    assert stats["num_seams"] > 0


def test_seam_metric_detects_known_discontinuities() -> None:
    """Inject large jumps at frame boundaries and check ratio is large."""
    rng = np.random.default_rng(42)
    n_frames = 16
    sig = 0.01 * rng.standard_normal(n_frames * FRAME).astype(np.float32)
    # Hard jumps at every FRAME boundary
    for k in range(1, n_frames):
        sig[k * FRAME:] += 1.0 if k % 2 == 0 else -1.0
    stats = seam_discontinuity_ratio(sig, frame_size=FRAME, hop_ratio=1.0, radius=2)
    assert stats["ratio"] > 5.0


def test_stitch_dispatches_to_registry() -> None:
    sig = np.linspace(-1.0, 1.0, 4 * FRAME, dtype=np.float32)
    a = stitch("hard_concat", identity_predict, sig, FRAME)
    b = reconstruct_hard_concat(identity_predict, sig, FRAME)
    np.testing.assert_array_equal(a, b)
