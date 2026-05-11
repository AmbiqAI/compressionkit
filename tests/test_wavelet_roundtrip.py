"""Round-trip tests for the wavelet compression helpers."""

from __future__ import annotations

import numpy as np

from compressionkit.dsp import dwt_forward, dwt_inverse


def test_haar_roundtrip_reconstructs_signal() -> None:
    """Forward + inverse Haar DWT should reconstruct the input within tolerance."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(512).astype(np.float32)

    coeffs = dwt_forward(signal, levels=4, wavelet="haar")
    recon = dwt_inverse(coeffs, wavelet="haar")

    assert recon.shape == signal.shape
    assert np.allclose(recon, signal, atol=1e-4)
