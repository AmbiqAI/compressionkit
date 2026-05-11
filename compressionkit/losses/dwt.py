"""Haar wavelet (DWT) subband-weighted MSE loss."""

from __future__ import annotations

import numpy as np
import keras.ops as ops


def build_dwt_loss(
    weight: float,
    levels: int,
    band_weights: list[float],
    num_leads: int = 1,
) -> callable:
    """Frequency-weighted MSE via Haar wavelet decomposition.

    Performs a multi-level Haar DWT on both ``y_true`` and ``y_pred``,
    then computes a weighted MSE per subband.  ``band_weights`` has
    ``levels + 1`` entries: ``[approx, detail_coarsest, ..., detail_finest]``.

    Input shape is ``(B, 1, T, C)``; the decomposition runs along axis 2.

    Args:
        weight: Global scalar multiplier for the total DWT loss.
        levels: Number of DWT decomposition levels.
        band_weights: Per-subband weights, length ``levels + 1``.
        num_leads: Number of input channels/leads.

    Returns:
        A callable ``dwt_loss(y_true, y_pred) -> scalar``.
    """
    assert len(band_weights) == levels + 1, (
        f"band_weights length {len(band_weights)} != levels + 1 = {levels + 1}"
    )

    s2 = np.float32(1.0 / np.sqrt(2.0))
    if num_leads == 1:
        lo_np = np.array([[s2, s2]], dtype=np.float32).reshape(1, 2, 1, 1)
        hi_np = np.array([[s2, -s2]], dtype=np.float32).reshape(1, 2, 1, 1)
        lo_kernel = ops.convert_to_tensor(lo_np)
        hi_kernel = ops.convert_to_tensor(hi_np)

        def _haar_step(x):
            approx = ops.conv(x, lo_kernel, strides=(1, 2), padding="valid")
            detail = ops.conv(x, hi_kernel, strides=(1, 2), padding="valid")
            return approx, detail
    else:
        lo_np = np.tile(
            np.array([[s2, s2]], dtype=np.float32).reshape(1, 2, 1, 1),
            (1, 1, num_leads, 1),
        )
        hi_np = np.tile(
            np.array([[s2, -s2]], dtype=np.float32).reshape(1, 2, 1, 1),
            (1, 1, num_leads, 1),
        )
        lo_kernel = ops.convert_to_tensor(lo_np)
        hi_kernel = ops.convert_to_tensor(hi_np)

        def _haar_step(x):
            approx = ops.depthwise_conv(x, lo_kernel, strides=(1, 2), padding="valid")
            detail = ops.depthwise_conv(x, hi_kernel, strides=(1, 2), padding="valid")
            return approx, detail

    bw = [ops.convert_to_tensor(float(w)) for w in band_weights]

    def dwt_loss(y_true, y_pred):
        yt = y_true
        yp = y_pred
        total = ops.convert_to_tensor(0.0)

        # Collect detail band MSEs from finest to coarsest
        detail_losses = []
        for _ in range(levels):
            yt_a, yt_d = _haar_step(yt)
            yp_a, yp_d = _haar_step(yp)
            detail_losses.append(ops.mean(ops.square(yt_d - yp_d)))
            yt = yt_a
            yp = yp_a

        # Approximation band MSE
        total = total + bw[0] * ops.mean(ops.square(yt - yp))

        # Detail bands: detail_losses[0] is finest, [-1] is coarsest
        # band_weights[1] = coarsest detail, band_weights[-1] = finest detail
        for i, d_loss in enumerate(reversed(detail_losses)):
            total = total + bw[i + 1] * d_loss

        return weight * total

    dwt_loss.__name__ = "dwt_loss"
    return dwt_loss
