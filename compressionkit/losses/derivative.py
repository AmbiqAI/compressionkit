"""First-difference (derivative) penalty loss."""

from __future__ import annotations

import keras.ops as ops


def build_derivative_loss(weight: float) -> callable:
    """Build a first-difference penalty scaled by *weight*.

    The loss measures how well the reconstruction preserves the first
    derivative (sample-to-sample differences) of the target signal.
    Input shape is ``(B, 1, T, C)``; the diff is computed along the time
    axis (axis=2).

    Args:
        weight: Scalar multiplier applied to the derivative MSE term.

    Returns:
        A callable ``derivative_loss(y_true, y_pred) -> scalar``.
    """

    def derivative_loss(y_true, y_pred):
        dt = ops.subtract(y_true[:, :, 1:, :], y_true[:, :, :-1, :])
        dp = ops.subtract(y_pred[:, :, 1:, :], y_pred[:, :, :-1, :])
        return weight * ops.mean(ops.square(ops.subtract(dt, dp)))

    derivative_loss.__name__ = "derivative_loss"
    return derivative_loss
