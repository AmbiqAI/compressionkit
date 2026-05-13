"""Unit tests for compressionkit.evaluation.noise — noise floor estimators."""

from __future__ import annotations

import numpy as np

from compressionkit.evaluation.noise import (
    estimate_bandpass_residual_noise,
    estimate_ecg_noise_floor,
    estimate_hf_noise_power,
    estimate_ppg_noise_floor,
)

FS_ECG = 256
FS_PPG = 64


def _make_clean_sine(fs: int, duration_s: float = 4.0, freq_hz: float = 1.0) -> np.ndarray:
    """Pure sine at `freq_hz` — all energy in-band, zero noise."""
    t = np.arange(int(fs * duration_s)) / fs
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _add_hf_noise(signal: np.ndarray, fs: int, noise_freq: float, amplitude: float) -> np.ndarray:
    """Add a high-frequency sinusoid as synthetic 'noise'."""
    t = np.arange(signal.size) / fs
    return signal + amplitude * np.sin(2 * np.pi * noise_freq * t).astype(np.float32)


# ---------- estimate_hf_noise_power -----------------------------------------


class TestHfNoisePower:
    def test_clean_signal_has_low_hf_power(self) -> None:
        """A 1 Hz sine at 256 Hz sample rate has ~0 energy above 40 Hz."""
        sig = _make_clean_sine(FS_ECG, freq_hz=1.0)
        out = estimate_hf_noise_power(sig, FS_ECG, hf_band=(40.0, None))
        assert out["hf_noise_power"] < 1e-4
        assert out["hf_noise_rms"] < 0.02

    def test_noisy_signal_has_high_hf_power(self) -> None:
        """Adding a 60 Hz sinusoid should show up in HF band."""
        sig = _make_clean_sine(FS_ECG, freq_hz=1.0)
        noisy = _add_hf_noise(sig, FS_ECG, noise_freq=60.0, amplitude=0.5)
        out = estimate_hf_noise_power(noisy, FS_ECG, hf_band=(40.0, None))
        assert out["hf_noise_power"] > 0.01

    def test_short_signal_returns_zeros(self) -> None:
        out = estimate_hf_noise_power(np.zeros(4), FS_ECG, hf_band=(40.0, None))
        assert out["hf_noise_power"] == 0.0
        assert out["hf_noise_rms"] == 0.0

    def test_custom_hf_band(self) -> None:
        """Custom band limits are reflected in the output."""
        sig = _make_clean_sine(FS_ECG, freq_hz=1.0)
        out = estimate_hf_noise_power(sig, FS_ECG, hf_band=(50.0, 100.0))
        assert out["hf_band_low"] == 50.0
        assert out["hf_band_high"] == 100.0


# ---------- estimate_bandpass_residual_noise ---------------------------------


class TestBandpassResidualNoise:
    def test_inband_sine_has_low_residual(self) -> None:
        """A 2 Hz sine inside 0.5–8 Hz should have near-zero residual."""
        sig = _make_clean_sine(FS_PPG, duration_s=8.0, freq_hz=2.0)
        out = estimate_bandpass_residual_noise(sig, FS_PPG, lowcut=0.5, highcut=8.0)
        assert out["bp_noise_rms"] < 0.05
        assert out["bp_signal_rms"] > 0.3

    def test_outofband_energy_captured(self) -> None:
        """Adding a DC offset + HF tone should increase residual."""
        sig = _make_clean_sine(FS_PPG, duration_s=8.0, freq_hz=2.0)
        noisy = sig + 1.0 + 0.3 * np.sin(2 * np.pi * 20.0 * np.arange(sig.size) / FS_PPG).astype(np.float32)
        out = estimate_bandpass_residual_noise(noisy, FS_PPG, lowcut=0.5, highcut=8.0)
        assert out["bp_noise_rms"] > 0.1

    def test_short_signal_returns_zeros(self) -> None:
        out = estimate_bandpass_residual_noise(np.zeros(4), FS_PPG, lowcut=0.5, highcut=8.0)
        assert out["bp_noise_power"] == 0.0

    def test_output_keys(self) -> None:
        sig = _make_clean_sine(FS_PPG, duration_s=4.0, freq_hz=2.0)
        out = estimate_bandpass_residual_noise(sig, FS_PPG, lowcut=0.5, highcut=8.0)
        expected_keys = {
            "bp_noise_rms",
            "bp_noise_power",
            "bp_signal_rms",
            "bp_signal_power",
            "bp_lowcut",
            "bp_highcut",
        }
        assert set(out.keys()) == expected_keys


# ---------- Composite estimators --------------------------------------------


class TestEcgNoiseFloor:
    def test_returns_all_three_estimators(self) -> None:
        """ECG composite should include HF, bandpass, and QRS keys."""
        sig = _make_clean_sine(FS_ECG, duration_s=4.0, freq_hz=1.0)
        out = estimate_ecg_noise_floor(sig, FS_ECG)
        assert "hf_noise_rms" in out
        assert "bp_noise_rms" in out
        # QRS SNR may be NaN if no peaks detected on a simple sine
        assert "qrs_snr_db" in out


class TestPpgNoiseFloor:
    def test_returns_hf_and_bp(self) -> None:
        """PPG composite should include HF and bandpass keys (no QRS)."""
        sig = _make_clean_sine(FS_PPG, duration_s=4.0, freq_hz=1.5)
        out = estimate_ppg_noise_floor(sig, FS_PPG)
        assert "hf_noise_rms" in out
        assert "bp_noise_rms" in out
        assert "qrs_snr_db" not in out

    def test_ppg_default_bands(self) -> None:
        """PPG noise estimator uses 8 Hz HF band and 0.5-8 Hz bandpass."""
        sig = _make_clean_sine(FS_PPG, duration_s=4.0, freq_hz=1.5)
        out = estimate_ppg_noise_floor(sig, FS_PPG)
        assert out["hf_band_low"] == 8.0
        assert out["bp_lowcut"] == 0.5
        assert out["bp_highcut"] == 8.0


# ---------- PRDN-noise (from metrics.compute_signal_metrics) ----------------


class TestPrdnNoise:
    """Verify that PRDN-noise correctly accounts for noise floor."""

    def test_perfect_recon_of_noisy_signal(self) -> None:
        """Perfect reconstruction of a noisy signal → PRD = 0, PRDN = 0."""
        sig = _make_clean_sine(FS_ECG, freq_hz=5.0, duration_s=4.0)
        from compressionkit.evaluation.metrics import compute_signal_metrics

        out = compute_signal_metrics(sig, sig, noise_power=0.01)
        assert np.isclose(out["prd_percent"], 0.0, atol=1e-6)
        assert np.isclose(out["prdn_noise_percent"], 0.0, atol=1e-6)

    def test_denoising_codec_gets_low_prdn(self) -> None:
        """Codec that removes noise: high PRD but low PRDN-noise.

        Scenario: original = clean + noise.  Recon = clean (perfect denoiser).
        PRD should be ~50% (since noise is ~50% RMS of signal).
        PRDN-noise should be ~0% because error ≈ noise energy.
        """
        rng = np.random.default_rng(42)
        clean = _make_clean_sine(FS_ECG, freq_hz=5.0, duration_s=4.0)
        noise = 0.5 * rng.standard_normal(clean.size).astype(np.float32)
        noisy = clean + noise
        noise_power = float(np.mean(noise**2))

        from compressionkit.evaluation.metrics import compute_signal_metrics

        out = compute_signal_metrics(noisy, clean, noise_power=noise_power)

        # PRD is high because reconstruction differs from noisy original
        assert out["prd_percent"] > 20.0
        # PRDN-noise is near zero because the error IS the noise
        assert out["prdn_noise_percent"] < 5.0

    def test_no_noise_power_omits_prdn(self) -> None:
        """Without noise_power arg, prdn_noise_percent is absent."""
        sig = _make_clean_sine(FS_ECG, freq_hz=5.0)
        from compressionkit.evaluation.metrics import compute_signal_metrics

        out = compute_signal_metrics(sig, sig)
        assert "prdn_noise_percent" not in out
