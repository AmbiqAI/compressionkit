"""Unit tests for compressionkit.evaluation.spectral_metrics."""

from __future__ import annotations

import numpy as np

from compressionkit.evaluation.spectral_metrics import (
    ECG_DEFAULT_BANDS,
    PPG_DEFAULT_FREQ_WEIGHTS,
    psd_band_error,
    spectral_coherence,
    weighted_freq_prd,
)

FS = 256


def _sine(fs: int, freq: float, duration_s: float = 4.0) -> np.ndarray:
    t = np.arange(int(fs * duration_s)) / fs
    return np.sin(2 * np.pi * freq * t).astype(np.float64)


# ---------- psd_band_error ---------------------------------------------------


class TestPsdBandError:
    def test_identical_signals_zero_error(self) -> None:
        sig = _sine(FS, 5.0)
        out = psd_band_error(sig, sig, fs=FS, bands=ECG_DEFAULT_BANDS)
        assert np.isclose(out["band_total_rel_error"], 0.0, atol=1e-6)
        for k, v in out.items():
            if k.endswith("_rel_error"):
                assert np.isclose(v, 0.0, atol=1e-6)

    def test_notched_band_shows_high_error(self) -> None:
        """Zeroing a frequency band should produce high error in that band."""
        sig = _sine(FS, 10.0)  # 10 Hz → in (5, 15) ECG band
        # Remove the 10 Hz component by using a zero signal
        recon = np.zeros_like(sig)
        out = psd_band_error(sig, recon, fs=FS, bands=ECG_DEFAULT_BANDS)
        # The (5, 15) band should have near-1.0 relative error
        assert out["band_5_15_rel_error"] > 0.95
        # Total error should be high
        assert out["band_total_rel_error"] > 0.2

    def test_out_of_band_noise_removal_is_benign(self) -> None:
        """Removing out-of-band noise shouldn't hurt clinical band error."""
        sig = _sine(FS, 10.0)
        # Add HF noise
        rng = np.random.default_rng(42)
        noise = 0.3 * rng.standard_normal(sig.size)
        noisy = sig + noise
        # Recon = clean signal (perfectly denoised)
        out = psd_band_error(noisy, sig, fs=FS, bands=ECG_DEFAULT_BANDS)
        # Clinical bands (0.5-5, 5-15, 15-40) should have low error
        assert out["band_5_15_rel_error"] < 0.15

    def test_output_keys_per_band(self) -> None:
        sig = _sine(FS, 2.0, duration_s=2.0)
        bands = [(0.5, 5.0), (5.0, 15.0)]
        out = psd_band_error(sig, sig, fs=FS, bands=bands)
        for lo, hi in bands:
            tag = f"band_{lo:g}_{hi:g}"
            assert f"{tag}_orig_power" in out
            assert f"{tag}_recon_power" in out
            assert f"{tag}_rel_error" in out
        assert "band_total_rel_error" in out


# ---------- weighted_freq_prd ------------------------------------------------


class TestWeightedFreqPrd:
    def test_identical_signals_zero_prd(self) -> None:
        sig = _sine(FS, 5.0)
        out = weighted_freq_prd(sig, sig, fs=FS, weights=PPG_DEFAULT_FREQ_WEIGHTS)
        assert np.isclose(out["weighted_freq_prd_percent"], 0.0, atol=1e-6)

    def test_scaled_signal_nonzero_prd(self) -> None:
        sig = _sine(FS, 5.0)
        out = weighted_freq_prd(sig, 0.5 * sig, fs=FS, weights=PPG_DEFAULT_FREQ_WEIGHTS)
        assert out["weighted_freq_prd_percent"] > 0.0

    def test_higher_weight_band_dominates(self) -> None:
        """Error concentrated in a high-weight band should yield higher PRD."""
        sig_lo = _sine(FS, 1.0, duration_s=4.0)  # 1 Hz → in (0.5, 3) band, weight=2.0
        sig_hi = _sine(FS, 5.0, duration_s=4.0)  # 5 Hz → in (3, 8) band, weight=1.0
        # Distort only the low-freq component
        recon_lo_distorted = sig_lo + sig_hi + 0.3 * _sine(FS, 1.0, duration_s=4.0)
        orig = sig_lo + sig_hi
        prd_lo = weighted_freq_prd(
            orig,
            recon_lo_distorted,
            fs=FS,
            weights=[(0.5, 3.0, 2.0), (3.0, 8.0, 1.0)],
        )["weighted_freq_prd_percent"]
        # Distort only the high-freq component
        recon_hi_distorted = sig_lo + sig_hi + 0.3 * _sine(FS, 5.0, duration_s=4.0)
        prd_hi = weighted_freq_prd(
            orig,
            recon_hi_distorted,
            fs=FS,
            weights=[(0.5, 3.0, 2.0), (3.0, 8.0, 1.0)],
        )["weighted_freq_prd_percent"]
        # Equal amplitude distortion in high-weight band → larger wf-PRD
        assert prd_lo > prd_hi

    def test_short_signal_returns_zero(self) -> None:
        out = weighted_freq_prd(np.zeros(2), np.zeros(2), fs=FS, weights=PPG_DEFAULT_FREQ_WEIGHTS)
        assert out["weighted_freq_prd_percent"] == 0.0


# ---------- spectral_coherence -----------------------------------------------


class TestSpectralCoherence:
    def test_identical_signals_perfect_coherence(self) -> None:
        sig = _sine(FS, 5.0)
        out = spectral_coherence(sig, sig, fs=FS, band=(0.5, 40.0))
        assert out["coherence_0.5_40"] > 0.99

    def test_uncorrelated_noise_low_coherence(self) -> None:
        """Two independent white noise signals should have near-zero coherence."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(2048)
        b = rng.standard_normal(2048)
        out = spectral_coherence(a, b, fs=FS, band=(1.0, 100.0))
        assert out["coherence_1_100"] < 0.3

    def test_scaled_copy_high_coherence(self) -> None:
        """Coherence is scale-invariant: 0.5*x should give ~1.0 coherence."""
        sig = _sine(FS, 10.0, duration_s=4.0)
        out = spectral_coherence(sig, 0.5 * sig, fs=FS, band=(5.0, 15.0))
        assert out["coherence_5_15"] > 0.99

    def test_short_signal_returns_zero(self) -> None:
        out = spectral_coherence(np.zeros(8), np.zeros(8), fs=FS, band=(1.0, 40.0))
        assert out["coherence_1_40"] == 0.0

    def test_band_key_format(self) -> None:
        """Output key should encode band boundaries."""
        sig = _sine(FS, 5.0)
        out = spectral_coherence(sig, sig, fs=FS, band=(0.5, 8.0))
        assert "coherence_0.5_8" in out
