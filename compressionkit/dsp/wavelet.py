"""Wavelet utilities for lightweight DSP compression baselines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:  # Optional dependency for robust wavelet transforms.
    import pywt  # type: ignore
except Exception:  # pragma: no cover - fallback when PyWavelets is unavailable.
    pywt = None


@dataclass(frozen=True)
class WaveletCoeffs:
    """Container for multilevel 1D wavelet coefficients."""

    approx: np.ndarray
    details: list[np.ndarray]


@dataclass(frozen=True)
class WaveletFilters:
    """Analysis/synthesis filters for a wavelet."""

    dec_lo: np.ndarray
    dec_hi: np.ndarray
    rec_lo: np.ndarray
    rec_hi: np.ndarray


def _get_wavelet_filters(name: str) -> WaveletFilters:
    """Return orthonormal wavelet filters for a given name."""
    if name == "haar":
        s = 1.0 / np.sqrt(2.0)
        dec_lo = np.array([s, s], dtype=np.float32)
        dec_hi = np.array([-s, s], dtype=np.float32)
        rec_lo = dec_lo[::-1].copy()
        rec_hi = dec_hi[::-1].copy()
        return WaveletFilters(dec_lo=dec_lo, dec_hi=dec_hi, rec_lo=rec_lo, rec_hi=rec_hi)
    if name == "db4":
        sqrt3 = np.sqrt(3.0)
        denom = 4.0 * np.sqrt(2.0)
        dec_lo = np.array(
            [
                (1 + sqrt3) / denom,
                (3 + sqrt3) / denom,
                (3 - sqrt3) / denom,
                (1 - sqrt3) / denom,
            ],
            dtype=np.float32,
        )
        dec_hi = np.array(
            [
                (1 - sqrt3) / denom,
                -(3 - sqrt3) / denom,
                (3 + sqrt3) / denom,
                -(1 + sqrt3) / denom,
            ],
            dtype=np.float32,
        )
        rec_lo = dec_lo[::-1].copy()
        rec_hi = dec_hi[::-1].copy()
        return WaveletFilters(dec_lo=dec_lo, dec_hi=dec_hi, rec_lo=rec_lo, rec_hi=rec_hi)
    raise ValueError(f"Unsupported wavelet: {name}")


def _analysis_step(signal: np.ndarray, filters: WaveletFilters) -> tuple[np.ndarray, np.ndarray]:
    """One level of periodic wavelet analysis."""
    n = signal.size
    if n % 2 != 0:
        signal = np.append(signal, signal[-1])
        n += 1
    half = n // 2
    approx = np.zeros(half, dtype=np.float32)
    detail = np.zeros(half, dtype=np.float32)
    L = filters.dec_lo.size
    for i in range(half):
        idx = (2 * i + np.arange(L)) % n
        segment = signal[idx]
        approx[i] = float(np.sum(filters.dec_lo * segment))
        detail[i] = float(np.sum(filters.dec_hi * segment))
    return approx, detail


def _synthesis_step(approx: np.ndarray, detail: np.ndarray, filters: WaveletFilters) -> np.ndarray:
    """One level of periodic wavelet synthesis."""
    half = approx.size
    n = half * 2
    signal = np.zeros(n, dtype=np.float32)
    L = filters.rec_lo.size
    for i in range(half):
        base_idx = 2 * i
        for k in range(L):
            idx = (base_idx + k) % n
            signal[idx] += filters.rec_lo[k] * approx[i] + filters.rec_hi[k] * detail[i]
    return signal


def dwt_forward(x: np.ndarray, levels: int = 4, wavelet: str = "haar") -> WaveletCoeffs:
    """Compute a multilevel 1D DWT using periodic extension."""
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if pywt is not None:
        coeffs = pywt.wavedec(x, wavelet, level=levels, mode="periodization")
        approx = np.asarray(coeffs[0], dtype=np.float32)
        # pywt returns details from highest to lowest level.
        details = [np.asarray(band, dtype=np.float32) for band in reversed(coeffs[1:])]
        return WaveletCoeffs(approx=approx, details=details)
    filters = _get_wavelet_filters(wavelet)
    coeffs: list[np.ndarray] = []
    current = np.asarray(x, dtype=np.float32)
    for _ in range(levels):
        approx, detail = _analysis_step(current, filters)
        coeffs.append(detail)
        current = approx
    return WaveletCoeffs(approx=current, details=coeffs)


def dwt_inverse(coeffs: WaveletCoeffs, wavelet: str = "haar") -> np.ndarray:
    """Reconstruct signal from multilevel 1D DWT coefficients."""
    if pywt is not None:
        details = [np.asarray(band, dtype=np.float32) for band in coeffs.details]
        coeffs_desc = [np.asarray(coeffs.approx, dtype=np.float32), *list(reversed(details))]
        return np.asarray(pywt.waverec(coeffs_desc, wavelet, mode="periodization"), dtype=np.float32)
    filters = _get_wavelet_filters(wavelet)
    approx = np.asarray(coeffs.approx, dtype=np.float32)
    for detail in reversed(coeffs.details):
        detail = np.asarray(detail, dtype=np.float32)
        approx = _synthesis_step(approx, detail, filters)
    return approx


def compute_thresholds(
    details: Sequence[np.ndarray],
    method: str = "energy",
    factor: float = 0.1,
    level_factors: Sequence[float] | None = None,
) -> list[float]:
    """Compute per-band thresholds for adaptive wavelet denoising."""
    thresholds = []
    if level_factors is not None and len(level_factors) != len(details):
        raise ValueError("level_factors length must match number of detail bands")
    for idx, band in enumerate(details):
        band = np.asarray(band)
        if method == "energy":
            thr = np.sqrt(np.mean(band**2)) * factor
        elif method == "percentile":
            thr = np.percentile(np.abs(band), 100 * (1 - factor))
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        if level_factors is not None:
            thr *= float(level_factors[idx])
        thresholds.append(float(thr))
    return thresholds


def apply_threshold(details: Sequence[np.ndarray], thresholds: Sequence[float], mode: str = "soft") -> list[np.ndarray]:
    """Apply per-band soft or hard thresholding."""
    if len(details) != len(thresholds):
        raise ValueError("details and thresholds must have the same length")
    output = []
    for band, thr in zip(details, thresholds):
        band = np.asarray(band, dtype=np.float32)
        if mode == "hard":
            band = band * (np.abs(band) >= thr)
        elif mode == "soft":
            band = np.sign(band) * np.maximum(np.abs(band) - thr, 0.0)
        else:
            raise ValueError(f"Unknown threshold mode: {mode}")
        output.append(band)
    return output


def compute_step_sizes(details: Sequence[np.ndarray], scale: float = 0.5) -> list[float]:
    """Compute per-band quantization step sizes from detail energy."""
    steps = []
    for band in details:
        band = np.asarray(band, dtype=np.float32)
        rms = np.sqrt(np.mean(band**2)) + 1e-12
        steps.append(float(rms * scale))
    return steps


def quantize_coeffs(details: Sequence[np.ndarray], step_sizes: Sequence[float]) -> list[np.ndarray]:
    """Uniformly quantize wavelet coefficients per band."""
    if len(details) != len(step_sizes):
        raise ValueError("details and step_sizes must have the same length")
    quantized = []
    for band, step in zip(details, step_sizes):
        band = np.asarray(band, dtype=np.float32)
        quantized.append(np.round(band / step).astype(np.int32))
    return quantized


def dequantize_coeffs(details: Sequence[np.ndarray], step_sizes: Sequence[float]) -> list[np.ndarray]:
    """Invert uniform quantization for wavelet coefficients per band."""
    if len(details) != len(step_sizes):
        raise ValueError("details and step_sizes must have the same length")
    dequantized = []
    for band, step in zip(details, step_sizes):
        dequantized.append(np.asarray(band, dtype=np.float32) * step)
    return dequantized


def pack_coeffs(coeffs: WaveletCoeffs) -> dict[str, list[np.ndarray]]:
    """Pack coefficients into a serializable dict."""
    return {"approx": coeffs.approx, "details": coeffs.details}


def unpack_coeffs(payload: dict[str, list[np.ndarray]]) -> WaveletCoeffs:
    """Restore coefficients from a packed dict."""
    return WaveletCoeffs(approx=np.asarray(payload["approx"]), details=[np.asarray(d) for d in payload["details"]])


def compute_prd(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute PRD (%) between original and reconstructed signals."""
    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    numerator = np.sum((original - reconstructed) ** 2)
    denominator = np.sum(original**2) + 1e-12
    return float(100.0 * np.sqrt(numerator / denominator))


@dataclass(frozen=True)
class WaveletCompressed:
    """Packed representation for wavelet+thresholding compression."""

    approx: np.ndarray
    details: list[np.ndarray]
    thresholds: list[float]
    step_sizes: list[float]
    levels: int
    wavelet: str


def compress_signal(
    x: np.ndarray,
    *,
    levels: int = 4,
    wavelet: str = "haar",
    threshold_method: str = "energy",
    threshold_factor: float = 0.1,
    threshold_mode: str = "soft",
    quant_scale: float = 0.5,
    threshold_factors: Sequence[float] | None = None,
) -> WaveletCompressed:
    """Compress a signal via wavelet decomposition + adaptive thresholding."""
    coeffs = dwt_forward(x, levels=levels, wavelet=wavelet)
    thresholds = compute_thresholds(
        coeffs.details,
        method=threshold_method,
        factor=threshold_factor,
        level_factors=threshold_factors,
    )
    thresholded = apply_threshold(coeffs.details, thresholds, mode=threshold_mode)
    if quant_scale <= 0:
        step_sizes = [1.0] * len(thresholded)
        quantized = [np.asarray(band, dtype=np.float32) for band in thresholded]
    else:
        step_sizes = compute_step_sizes(thresholded, scale=quant_scale)
        quantized = quantize_coeffs(thresholded, step_sizes)
    return WaveletCompressed(
        approx=coeffs.approx,
        details=quantized,
        thresholds=thresholds,
        step_sizes=step_sizes,
        levels=levels,
        wavelet=wavelet,
    )


def decompress_signal(payload: WaveletCompressed) -> np.ndarray:
    """Reconstruct a signal from a WaveletCompressed payload."""
    dequantized = dequantize_coeffs(payload.details, payload.step_sizes)
    coeffs = WaveletCoeffs(approx=payload.approx, details=dequantized)
    return dwt_inverse(coeffs, wavelet=payload.wavelet)


__all__ = [
    "WaveletCoeffs",
    "WaveletCompressed",
    "WaveletFilters",
    "apply_threshold",
    "compress_signal",
    "compute_prd",
    "compute_step_sizes",
    "compute_thresholds",
    "decompress_signal",
    "dequantize_coeffs",
    "dwt_forward",
    "dwt_inverse",
    "pack_coeffs",
    "quantize_coeffs",
    "unpack_coeffs",
]
