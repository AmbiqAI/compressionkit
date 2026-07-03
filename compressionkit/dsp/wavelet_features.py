"""Keras-free numpy helpers for the wavelet-gain denoiser's noise-level features.

These are the "twin" numpy implementations of the Keras feature layers used
during training (:class:`~compressionkit.models.wavelet_denoiser.FinestLevelFeature`
and :class:`~compressionkit.models.wavelet_denoiser.BandRatioFeature`). They live
here (rather than in ``compressionkit.models``) so they can be imported by
keras/TensorFlow-free consumers — in particular the LiteRT-only runtime path
in :mod:`compressionkit.pipeline.learned_stages`, which must not pull in a
full Keras/TF dependency to run a quantized denoiser on an embedded target.

``compressionkit.models.wavelet_denoiser`` re-exports these for backward
compatibility; import from either location.
"""

from __future__ import annotations

import numpy as np

from compressionkit.dsp.wavelet import dwt_forward

__all__ = [
    "band_ratio_feature_np",
    "finest_band_indices",
    "level_feature_np",
]


def finest_band_indices(frame_size: int, wavelet: str, levels: int) -> tuple[int, int]:
    """Return ``(start, length)`` of the finest detail band in packed order.

    Packing order is ``[approx, cD_1 (finest), cD_2, ..., cD_L]`` to match
    :class:`~compressionkit.evaluation.codec.LearnedShrinkSpihtCodec`, so the
    finest band immediately follows the approximation coefficients.
    """
    coeffs = dwt_forward(np.zeros(frame_size, dtype=np.float32), levels=levels, wavelet=wavelet)
    approx_len = len(coeffs.approx)
    finest_len = len(coeffs.details[0])
    return approx_len, finest_len


def level_feature_np(packed: np.ndarray, start: int, length: int) -> np.ndarray:
    """NumPy twin of ``FinestLevelFeature`` for the eval / codec inference path."""
    arr = np.asarray(packed, dtype=np.float32)
    finest = arr[start : start + length]
    level = np.float32(np.log1p(np.std(finest)))
    return np.stack([arr, np.full_like(arr, level)], axis=-1)  # (T, 2)


def band_ratio_feature_np(packed: np.ndarray, start: int, length: int) -> np.ndarray:
    """NumPy twin of ``BandRatioFeature`` for eval / codec inference."""
    arr = np.asarray(packed, dtype=np.float32)
    finest = arr[start : start + length]
    coarse = np.concatenate([arr[:start], arr[start + length :]], axis=0)
    fine_rms = np.sqrt(np.mean(finest**2) + 1e-6)
    coarse_rms = np.sqrt(np.mean(coarse**2) + 1e-6)
    ratio = np.float32(np.log1p(fine_rms / (coarse_rms + 1e-6)))
    return np.stack([arr, np.full_like(arr, ratio)], axis=-1)
