"""Reusable IIR filtering helpers (denoising bands, clean-signal proxies).

These are embedded-portable building blocks: zero-phase Butterworth
second-order sections with fixed coefficients and no dynamic allocation.
They are shared by codecs (e.g. :class:`~compressionkit.evaluation.codec.FilterSpihtCodec`)
and by evaluation harnesses that need a clean ground-truth *proxy* to score
reconstructions against when the true clean signal is unknown.

All functions are CPU-only and safe to import without TensorFlow.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ECG_PROXY_BAND",
    "PPG_PROXY_BAND",
    "bandpass",
    "butter_bandpass_sos",
    "clean_proxy",
]

# Default clean-proxy passbands per modality, in Hz. These bracket the
# physiologically meaningful content (P-QRS-T for ECG, pulse + harmonics for
# PPG) while rejecting baseline wander and high-frequency noise.
ECG_PROXY_BAND: tuple[float, float] = (0.5, 40.0)
PPG_PROXY_BAND: tuple[float, float] = (0.5, 8.0)


def butter_bandpass_sos(
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    order: int = 3,
):
    """Design a Butterworth band-pass filter as second-order sections.

    Args:
        sample_rate: Sampling rate in Hz.
        low_hz: Lower cutoff in Hz (must be > 0).
        high_hz: Upper cutoff in Hz; clamped to ``0.95 * nyquist``.
        order: Filter order.

    Returns:
        SOS array suitable for :func:`scipy.signal.sosfiltfilt`.

    Raises:
        ValueError: If the resulting passband is degenerate.
    """
    from scipy import signal as scipy_signal

    nyq = sample_rate / 2.0
    high = min(high_hz, nyq * 0.95)
    if not 0.0 < low_hz < high:
        raise ValueError(f"Invalid bandpass cutoffs: low_hz={low_hz}, high_hz={high_hz} (nyquist={nyq})")
    return scipy_signal.butter(order, [low_hz / nyq, high / nyq], btype="bandpass", output="sos")


def bandpass(
    signal_array: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    *,
    order: int = 3,
    forward_backward: bool = True,
) -> np.ndarray:
    """Band-pass filter a 1-D signal.

    Args:
        signal_array: Input samples, shape ``(n,)``.
        sample_rate: Sampling rate in Hz.
        low_hz: Lower cutoff in Hz.
        high_hz: Upper cutoff in Hz.
        order: Filter order.
        forward_backward: When True, use zero-phase :func:`sosfiltfilt`;
            otherwise causal :func:`sosfilt`. Falls back to causal filtering
            automatically for signals too short for the zero-phase padding.

    Returns:
        Filtered signal as ``float32``, same length as the input.
    """
    from scipy import signal as scipy_signal

    arr = np.asarray(signal_array, dtype=np.float32).reshape(-1)
    sos = butter_bandpass_sos(sample_rate, low_hz, high_hz, order)
    if forward_backward and arr.size > 3 * (2 * sos.shape[0] + 1):
        return scipy_signal.sosfiltfilt(sos, arr).astype(np.float32)
    return scipy_signal.sosfilt(sos, arr).astype(np.float32)


def clean_proxy(
    signal_array: np.ndarray,
    sample_rate: float,
    modality: str,
    *,
    order: int = 3,
) -> np.ndarray:
    """Return a band-limited clean-signal proxy for scoring reconstructions.

    The proxy is a zero-phase band-pass using the modality default band
    (:data:`ECG_PROXY_BAND` or :data:`PPG_PROXY_BAND`). Scoring against this
    proxy measures *truth* fidelity (how well a codec recovers the underlying
    physiological signal), complementing scoring against the raw noisy input
    (*faithfulness*).

    Args:
        signal_array: Input samples, shape ``(n,)``.
        sample_rate: Sampling rate in Hz.
        modality: ``"ecg"`` or ``"ppg"``.
        order: Filter order.

    Returns:
        The band-limited proxy as ``float32``.
    """
    band = ECG_PROXY_BAND if modality == "ecg" else PPG_PROXY_BAND
    return bandpass(signal_array, sample_rate, band[0], band[1], order=order)
