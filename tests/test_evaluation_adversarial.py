"""Tests for :mod:`compressionkit.evaluation.adversarial`."""

from __future__ import annotations

import numpy as np
import pytest

from compressionkit.evaluation import (
    PHYSIO_BANDS,
    AdversarialResult,
    IdentityCodec,
    SpihtAcCodec,
    inject_gaussian,
    make_dc_offset,
    make_gaussian_noise,
    make_signal_zero_mix,
    make_sinusoid,
    make_step,
    make_zero_burst,
    make_zero_input,
    run_adversarial_battery,
)

# ---------------------------------------------------------------------------
# Generator unit tests
# ---------------------------------------------------------------------------


def test_make_zero_input_shape() -> None:
    arr = make_zero_input(320, n_frames=4)
    assert arr.shape == (4, 320)
    assert arr.dtype == np.float32
    assert np.all(arr == 0.0)


def test_make_gaussian_noise_reproducible() -> None:
    a = make_gaussian_noise(320, n_frames=4, std=2.0, seed=42)
    b = make_gaussian_noise(320, n_frames=4, std=2.0, seed=42)
    np.testing.assert_array_equal(a, b)
    # std is approximately correct (loose check)
    assert 1.5 < float(np.std(a)) < 2.5


def test_make_dc_offset_values() -> None:
    arr = make_dc_offset(8, offsets=[-1.0, 0.0, 1.5])
    assert arr.shape == (3, 8)
    np.testing.assert_array_equal(arr[0], -1.0)
    np.testing.assert_array_equal(arr[1], 0.0)
    np.testing.assert_array_equal(arr[2], 1.5)


def test_make_sinusoid_frequency() -> None:
    fs = 64
    arr = make_sinusoid(fs * 4, fs=fs, freqs_hz=[2.0], amp=1.0)
    # Expect dominant FFT bin at 2 Hz
    spectrum = np.abs(np.fft.rfft(arr[0]))
    freqs = np.fft.rfftfreq(arr.shape[1], 1 / fs)
    assert freqs[np.argmax(spectrum)] == pytest.approx(2.0, abs=0.1)


def test_make_step_position() -> None:
    arr = make_step(16, step_positions=[8], step_amp=1.0)
    np.testing.assert_array_equal(arr[0, :8], 0.0)
    np.testing.assert_array_equal(arr[0, 8:], 1.0)


def test_make_zero_burst_creates_gaps() -> None:
    sig = np.ones((2, 64), dtype=np.float32)
    out = make_zero_burst(sig, burst_len=8, n_bursts=1, seed=0)
    for i in range(2):
        assert np.sum(out[i] == 0) >= 8


def test_make_signal_zero_mix_fraction() -> None:
    sig = np.ones((2, 100), dtype=np.float32)
    out = make_signal_zero_mix(sig, zero_fraction=0.25, seed=0)
    for i in range(2):
        assert int(np.sum(out[i] == 0)) == 25


def test_inject_gaussian_snr_calibration() -> None:
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((4, 1024)).astype(np.float32)
    noisy = inject_gaussian(sig, input_snr_db=10.0, seed=0)
    # Measure realised SNR
    noise = noisy - sig
    snr = 10.0 * np.log10(np.mean(sig**2) / np.mean(noise**2))
    assert snr == pytest.approx(10.0, abs=1.0)


def test_signal_zero_mix_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        make_signal_zero_mix(np.zeros((1, 8), dtype=np.float32), zero_fraction=1.5)


# ---------------------------------------------------------------------------
# Battery — IdentityCodec sanity
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_ppg() -> IdentityCodec:
    return IdentityCodec(modality="ppg", sample_rate=64, frame_size=320)


def test_battery_returns_results(identity_ppg: IdentityCodec) -> None:
    results = run_adversarial_battery(identity_ppg, n_frames=4, seed=0)
    assert len(results) >= 5  # zero, gaussian, dc, sinusoid, step
    assert all(isinstance(r, AdversarialResult) for r in results)
    names = {r.test_name for r in results}
    assert {"zero_input", "gaussian_noise", "dc_offset", "sinusoid", "step"} <= names


def test_identity_zero_input_no_hallucination(identity_ppg: IdentityCodec) -> None:
    """Identity codec on zero input must produce zero out → no hallucination."""
    results = run_adversarial_battery(identity_ppg, n_frames=4, seed=0)
    zero = next(r for r in results if r.test_name == "zero_input")
    assert zero.input_energy == 0.0
    assert zero.output_energy == 0.0
    assert zero.energy_ratio == 0.0
    assert zero.output_l2_when_input_zero == 0.0
    assert zero.hallucinated_peaks == 0.0


def test_identity_gaussian_energy_preserved(identity_ppg: IdentityCodec) -> None:
    """Identity codec on Gaussian noise: out energy == in energy."""
    results = run_adversarial_battery(identity_ppg, n_frames=4, seed=0)
    g = next(r for r in results if r.test_name == "gaussian_noise")
    assert g.energy_ratio == pytest.approx(1.0, abs=1e-3)


def test_battery_with_signal_frames_runs_snr_sweep(identity_ppg: IdentityCodec) -> None:
    rng = np.random.default_rng(0)
    t = np.arange(320) / 64.0
    sig = np.stack(
        [(np.sin(2 * np.pi * 1.2 * t) + 0.05 * rng.standard_normal(320)).astype(np.float32) for _ in range(4)]
    )
    results = run_adversarial_battery(
        identity_ppg,
        n_frames=4,
        seed=0,
        signal_frames=sig,
        input_snr_db_sweep=[20.0, 0.0],
    )
    names = {r.test_name for r in results}
    assert "signal_plus_gaussian_20dB" in names
    assert "signal_plus_gaussian_0dB" in names
    assert "signal_with_zero_bursts" in names
    assert "signal_zero_mix_30pct" in names


def test_to_dict_round_trip(identity_ppg: IdentityCodec) -> None:
    results = run_adversarial_battery(identity_ppg, n_frames=2, seed=0)
    d = results[0].to_dict()
    assert d["test_name"] == results[0].test_name
    assert d["n_frames"] == results[0].n_frames


# ---------------------------------------------------------------------------
# Battery — SPIHT-AC sanity (real lossy codec)
# ---------------------------------------------------------------------------


@pytest.fixture
def spiht_ppg() -> SpihtAcCodec:
    return SpihtAcCodec(
        modality="ppg",
        sample_rate=64,
        frame_size=320,
        target_cr=4.0,
        levels=5,
    )


def test_spiht_zero_input_does_not_hallucinate(spiht_ppg: SpihtAcCodec) -> None:
    """SPIHT-AC on zeros: output must have negligible energy (no hallucinated pulses)."""
    results = run_adversarial_battery(spiht_ppg, n_frames=4, seed=0)
    zero = next(r for r in results if r.test_name == "zero_input")
    band = PHYSIO_BANDS["ppg"]
    assert band is not None
    # Output L2 must be ~0
    assert zero.output_l2_when_input_zero is not None
    assert zero.output_l2_when_input_zero < 1e-3
    assert zero.hallucinated_peaks == 0.0


def test_physio_bands_table_completeness() -> None:
    assert "ppg" in PHYSIO_BANDS
    assert "ecg" in PHYSIO_BANDS
    for _k, (lo, hi) in PHYSIO_BANDS.items():
        assert lo < hi
        assert lo > 0
