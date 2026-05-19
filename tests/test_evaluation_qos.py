"""Tests for :mod:`compressionkit.evaluation.qos`.

QoS-from-side-channel computations are tested with hand-crafted
``EncodedFrame`` instances (no model required). The end-to-end calibration
behavior is tested against the real PPG/ECG golden RVQ run when available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compressionkit.evaluation import (
    EncodedFrame,
    RvqQoS,
    RvqQoSCalibrator,
    compute_rvq_qos,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_RUNS: list[tuple[str, str, int, int]] = [
    ("results/ppg_rvq_64hz_04x_golden", "ppg", 320, 1),
    ("results/ecg_rvq_256hz_04x_golden", "ecg", 512, 1),
    ("results/ecg_rvq_256hz_12lead_32x_big", "ecg", 512, 12),
]


# ---------------------------------------------------------------------------
# Unit tests (no model required)
# ---------------------------------------------------------------------------


def _fake_encoded(
    quant_distances: list[np.ndarray],
    residual_norms: list[float],
    token_ids: list[np.ndarray],
    codebook_sizes: list[int],
) -> EncodedFrame:
    payload = np.zeros(0, dtype=np.uint8)
    return EncodedFrame(
        payload=payload,
        nbits=0,
        side={
            "quant_distances": quant_distances,
            "residual_norms": residual_norms,
            "token_ids": token_ids,
            "codebook_sizes": codebook_sizes,
        },
    )


def test_compute_rvq_qos_basic_fields() -> None:
    enc = _fake_encoded(
        quant_distances=[np.array([0.1, 0.2, 0.3]), np.array([0.05, 0.05, 0.10])],
        residual_norms=[1.0, 0.5, 0.1],
        token_ids=[np.array([0, 1, 0]), np.array([2, 2, 2])],
        codebook_sizes=[4, 4],
    )
    qos = compute_rvq_qos(enc)
    assert isinstance(qos, RvqQoS)
    assert qos.mean_quant_distance == pytest.approx(np.mean([0.1, 0.2, 0.3, 0.05, 0.05, 0.10]))
    assert qos.max_quant_distance == pytest.approx(0.3)
    assert qos.per_level_mean_distance == pytest.approx([0.2, 0.2 / 3])
    assert qos.residual_norm_per_stage == [1.0, 0.5, 0.1]
    assert qos.final_residual_norm == pytest.approx(0.1)
    assert qos.relative_residual == pytest.approx(0.1)
    # Codebook perplexity: level 0 uses {0,1,0} → 2^H(2/3,1/3) ≈ 1.89; level 1 is degenerate → 1.0.
    assert qos.codebook_perplexity_per_level[0] == pytest.approx(
        2 ** (-(2 / 3) * np.log2(2 / 3) - (1 / 3) * np.log2(1 / 3)), rel=1e-6
    )
    assert qos.codebook_perplexity_per_level[1] == pytest.approx(1.0)
    assert qos.confidence is None


def test_compute_rvq_qos_requires_side_keys() -> None:
    bad = EncodedFrame(payload=np.zeros(0, dtype=np.uint8), nbits=0, side={})
    with pytest.raises(ValueError):
        compute_rvq_qos(bad)


def test_compute_rvq_qos_zero_initial_norm_is_safe() -> None:
    enc = _fake_encoded(
        quant_distances=[np.array([0.0])],
        residual_norms=[0.0, 0.0],
        token_ids=[np.array([0])],
        codebook_sizes=[2],
    )
    qos = compute_rvq_qos(enc)
    assert qos.relative_residual == 0.0
    assert qos.final_residual_norm == 0.0


def test_qos_to_dict_roundtrip() -> None:
    enc = _fake_encoded(
        quant_distances=[np.array([0.5])],
        residual_norms=[1.0, 0.5],
        token_ids=[np.array([0])],
        codebook_sizes=[2],
    )
    d = compute_rvq_qos(enc).to_dict()
    for key in (
        "mean_quant_distance",
        "max_quant_distance",
        "per_level_mean_distance",
        "residual_norm_per_stage",
        "final_residual_norm",
        "relative_residual",
        "codebook_perplexity_per_level",
        "initial_latent_norm",
        "confidence",
    ):
        assert key in d


def test_calibrator_requires_fit() -> None:
    cal = RvqQoSCalibrator()
    assert not cal.fitted
    qos = compute_rvq_qos(_fake_encoded([np.array([0.1])], [1.0, 0.5], [np.array([0])], [2]))
    with pytest.raises(RuntimeError):
        cal.score(qos)


def test_calibrator_empty_fit_raises() -> None:
    with pytest.raises(ValueError):
        RvqQoSCalibrator().fit([])


def _qos(mean_d: float, rel: float, latent_norm: float = 10.0) -> RvqQoS:
    return RvqQoS(
        mean_quant_distance=mean_d,
        max_quant_distance=mean_d * 1.5,
        per_level_mean_distance=[mean_d],
        residual_norm_per_stage=[latent_norm, rel * latent_norm],
        final_residual_norm=rel * latent_norm,
        relative_residual=rel,
        codebook_perplexity_per_level=[4.0],
        initial_latent_norm=latent_norm,
    )


def test_calibrator_ranks_excellent_above_median_above_ood() -> None:
    # In-dist samples: latent_norm ~ U(8, 12), mean_d ~ U(0.1, 0.2), rel ~ U(0.05, 0.15).
    rng = np.random.default_rng(0)
    samples = [
        _qos(
            mean_d=float(rng.uniform(0.1, 0.2)),
            rel=float(rng.uniform(0.05, 0.15)),
            latent_norm=float(rng.uniform(8.0, 12.0)),
        )
        for _ in range(200)
    ]
    cal = RvqQoSCalibrator().fit(samples)
    assert cal.fitted

    median = _qos(mean_d=0.15, rel=0.10, latent_norm=10.0)
    excellent = _qos(mean_d=0.05, rel=0.02, latent_norm=10.0)
    ood_residual = _qos(mean_d=5.0, rel=0.99, latent_norm=10.0)

    c_median = cal.score(median)
    c_excellent = cal.score(excellent)
    c_ood = cal.score(ood_residual)

    assert 0.0 <= c_ood < c_median < c_excellent <= 1.0
    assert c_excellent > 0.85
    assert c_ood < 0.05
    assert median.confidence == pytest.approx(c_median)


def test_calibrator_low_energy_guard_penalises_zero_input() -> None:
    # Zero-input lookalike: tiny initial latent norm, tiny residual, tiny distance.
    # Old 2-axis calibrator scored this 1.0 (perfect). The 3-axis norm guard
    # must drive confidence near 0.
    rng = np.random.default_rng(1)
    samples = [
        _qos(
            mean_d=float(rng.uniform(0.1, 0.2)),
            rel=float(rng.uniform(0.05, 0.15)),
            latent_norm=float(rng.uniform(8.0, 12.0)),
        )
        for _ in range(200)
    ]
    cal = RvqQoSCalibrator().fit(samples)

    zero_like = _qos(mean_d=1e-6, rel=1e-6, latent_norm=1e-4)
    huge = _qos(mean_d=0.15, rel=0.10, latent_norm=1000.0)
    assert cal.score(zero_like) < 0.05
    assert cal.score(huge) < 0.05


# ---------------------------------------------------------------------------
# Integration test against a real RVQ run (skipped without golden artifacts)
# ---------------------------------------------------------------------------


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


def _make_in_dist_frame(codec, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(codec.frame_size) / codec.sample_rate
    if codec.modality == "ppg":
        sig = np.sin(2 * np.pi * 1.2 * t) + 0.05 * rng.standard_normal(codec.frame_size)
    else:
        sig = np.sin(2 * np.pi * 1.0 * t) + 0.1 * rng.standard_normal(codec.frame_size)
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
    if codec.n_channels == 1:
        return sig.astype(np.float32)
    return np.tile(sig[:, None].astype(np.float32), (1, codec.n_channels))


def test_qos_in_dist_higher_than_ood(rvq_codec) -> None:
    # Calibrate on ~16 synthetic in-distribution frames; ensure that calibrated
    # confidence is higher on a fresh in-dist frame than on Gaussian noise.
    in_dist = [compute_rvq_qos(rvq_codec.encode(_make_in_dist_frame(rvq_codec, seed=s))) for s in range(16)]
    cal = RvqQoSCalibrator().fit(in_dist)

    fresh_in_dist = compute_rvq_qos(rvq_codec.encode(_make_in_dist_frame(rvq_codec, seed=999)))
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((rvq_codec.frame_size, rvq_codec.n_channels)).astype(np.float32)
    if rvq_codec.n_channels == 1:
        noise = noise[:, 0]
    ood = compute_rvq_qos(rvq_codec.encode(noise))

    c_in = cal.score(fresh_in_dist)
    c_ood = cal.score(ood)
    assert 0.0 <= c_ood <= 1.0
    assert 0.0 <= c_in <= 1.0
    # In-dist confidence must be strictly greater than pure-noise confidence.
    assert c_in > c_ood
