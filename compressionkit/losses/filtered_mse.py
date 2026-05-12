"""Lowpass-filtered MSE loss for bandwidth-limited compression."""

from __future__ import annotations

import keras.ops as ops
import numpy as np


def _design_fir_lowpass(
    cutoff_hz: float,
    sample_rate: int,
    num_taps: int,
) -> np.ndarray:
    """Design a Hamming-windowed sinc lowpass FIR filter.

    Returns the filter coefficients as a float32 numpy array of shape
    ``(num_taps,)`` normalised to unit DC gain.

    Args:
        cutoff_hz: Filter cutoff frequency in Hz.
        sample_rate: Signal sample rate in Hz.
        num_taps: Number of FIR filter taps (odd is preferred).

    Returns:
        1-D float32 array of filter coefficients.
    """
    n = np.arange(num_taps)
    mid = (num_taps - 1) / 2.0
    fc = cutoff_hz / sample_rate  # normalised cutoff (cycles/sample)

    # windowed-sinc
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(
            n == mid,
            2.0 * fc,
            np.sin(2.0 * np.pi * fc * (n - mid)) / (np.pi * (n - mid)),
        )
    window = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (num_taps - 1))
    h = h * window
    h = h / h.sum()  # normalise to unit gain
    return h.astype(np.float32)


def build_filtered_mse_loss(
    weight: float,
    sample_rate: int,
    cutoff_hz: float,
    num_taps: int = 65,
    num_leads: int = 1,
) -> callable:
    """MSE loss computed on lowpass-filtered signals.

    A fixed FIR lowpass filter is applied to both ``y_true`` and ``y_pred``
    before computing MSE.  This prevents the model from being penalised for
    high-frequency content that the temporal bottleneck cannot represent.

    Input shape is ``(B, 1, T, C)``; the convolution runs along axis 2.
    For multi-lead inputs (``num_leads > 1``), depthwise convolution applies
    the same filter independently to each channel.

    Args:
        weight: Scalar multiplier applied to the filtered MSE term.
        sample_rate: Signal sample rate in Hz.
        cutoff_hz: Lowpass cutoff frequency in Hz.
        num_taps: Number of FIR filter taps.
        num_leads: Number of input channels/leads.

    Returns:
        A callable ``filtered_mse_loss(y_true, y_pred) -> scalar``.
    """
    h = _design_fir_lowpass(cutoff_hz, sample_rate, num_taps)
    if num_leads == 1:
        kernel = ops.convert_to_tensor(h.reshape(1, num_taps, 1, 1))

        def _lowpass(x):
            return ops.conv(x, kernel, strides=1, padding="same")
    else:
        kernel = ops.convert_to_tensor(
            np.tile(h.reshape(1, num_taps, 1, 1), (1, 1, num_leads, 1))
        )

        def _lowpass(x):
            return ops.depthwise_conv(x, kernel, strides=1, padding="same")

    def filtered_mse_loss(y_true, y_pred):
        return weight * ops.mean(ops.square(_lowpass(y_true) - _lowpass(y_pred)))

    filtered_mse_loss.__name__ = "filtered_mse_loss"
    return filtered_mse_loss
