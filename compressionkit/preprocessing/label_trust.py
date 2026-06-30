"""Reference-free label-trust weighting for ECG/PPG reconstruction training.

The supervised target for a real recording already contains inherent
corruption (``n0``): motion, mains pickup, baseline wander, EMG, lead-off, etc.
Reconstructing such a target perfectly means reproducing its noise. This module
estimates ``n0`` per target window with a fully TF-native, reference-free
signal-quality proxy and converts it into a per-sample (or per-timestep) loss
weight: trustworthy (low-``n0``) targets become *strong* labels with high
weight, while corrupted (high-``n0``) targets become *weak* labels with relaxed
weight.

This is a training-time-only signal. It shapes the objective via
``sample_weight`` and never appears in the deployed (LiteRT/INT8) graph, so it
is portability-neutral.

The corruption proxy is the local out-of-band residual power fraction: the
signal is split into a band-limited physiological estimate (a short moving
average removes high-frequency content, a long moving average removes baseline
drift) and a residual that captures high-frequency noise plus low-frequency
drift. The smoothed residual-to-signal power ratio is squashed to ``[0, 1]``.
Score smoothing is deliberately coarse so brief broadband QRS energy is not
mistaken for sustained corruption.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf


def _ms_to_odd_window(sample_rate: float, ms: float) -> int:
    """Convert a duration in milliseconds to an odd moving-average window."""
    win = int(round(float(sample_rate) * float(ms) / 1000.0))
    win = max(1, win)
    return win if win % 2 == 1 else win + 1


def _moving_average(x: tf.Tensor, win: int) -> tf.Tensor:
    """Length-preserving moving average over the time axis.

    Args:
        x: Tensor of shape ``(N, T, C)``.
        win: Odd window length in samples. ``win <= 1`` is a no-op.

    Returns:
        Tensor of shape ``(N, T, C)``.
    """
    if win <= 1:
        return x
    pad = win // 2
    x_padded = tf.pad(x, [[0, 0], [pad, pad], [0, 0]], mode="REFLECT")
    return tf.nn.avg_pool1d(x_padded, ksize=win, strides=1, padding="VALID")


def estimate_corruption_score(
    y: tf.Tensor,
    *,
    sample_rate: float,
    hf_window_ms: float = 40.0,
    baseline_window_ms: float = 900.0,
    smooth_ms: float = 300.0,
    half_sat_ratio: float = 0.25,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Estimate per-timestep inherent corruption (``n0``) of a target batch.

    Args:
        y: Target batch of shape ``(B, 1, T, C)``.
        sample_rate: Effective sample rate of ``y`` in Hz.
        hf_window_ms: Short moving-average window; content faster than this is
            treated as out-of-band (high-frequency noise).
        baseline_window_ms: Long moving-average window capturing baseline drift.
        smooth_ms: Smoothing window applied to the power ratio so localized QRS
            energy does not register as sustained corruption.
        half_sat_ratio: Residual/signal power ratio mapped to a score of 0.5.
        eps: Numerical floor for the signal power denominator.

    Returns:
        Corruption score of shape ``(B, 1, T)`` in ``[0, 1]`` (0 = clean).
    """
    shape = tf.shape(y)
    batch, length, channels = shape[0], shape[2], shape[3]
    sig = tf.reshape(y, (batch, length, channels))

    hf_win = _ms_to_odd_window(sample_rate, hf_window_ms)
    baseline_win = _ms_to_odd_window(sample_rate, baseline_window_ms)
    smooth_win = _ms_to_odd_window(sample_rate, smooth_ms)

    low_pass = _moving_average(sig, hf_win)
    baseline = _moving_average(sig, baseline_win)
    physiological = low_pass - baseline
    residual = sig - physiological

    residual_power = _moving_average(tf.square(residual), smooth_win)
    signal_power = _moving_average(tf.square(sig), smooth_win)
    ratio = residual_power / (signal_power + eps)

    score = ratio / (ratio + half_sat_ratio)
    score = tf.reduce_mean(score, axis=-1)
    return tf.reshape(score, (batch, 1, length))


def corruption_to_weight(
    score: tf.Tensor,
    *,
    w_min: float = 0.25,
    gamma: float = 1.0,
    granularity: str = "window",
    normalize: bool = True,
) -> tf.Tensor:
    """Map a corruption score to a reconstruction-loss weight.

    Args:
        score: Corruption score of shape ``(B, 1, T)`` in ``[0, 1]``.
        w_min: Floor weight for fully corrupted targets.
        gamma: Sharpness of the trust-to-weight curve (``trust ** gamma``).
        granularity: ``"timestep"`` keeps per-sample weights; ``"window"``
            reduces to one weight per window.
        normalize: Rescale weights to mean ~1 so the effective learning rate is
            preserved.

    Returns:
        Weight tensor broadcastable against the per-element reconstruction loss
        of shape ``(B, 1, T)`` — ``(B, 1, T)`` for timestep granularity or
        ``(B, 1, 1)`` for window granularity.
    """
    trust = tf.clip_by_value(1.0 - score, 0.0, 1.0)
    weight = w_min + (1.0 - w_min) * tf.pow(trust, gamma)

    if granularity == "window":
        weight = tf.reduce_mean(weight, axis=-1, keepdims=True)
    elif granularity != "timestep":
        raise ValueError(f"granularity must be 'window' or 'timestep', got {granularity!r}")

    if normalize:
        scale = tf.cast(tf.size(weight), weight.dtype) / (tf.reduce_sum(weight) + 1e-8)
        weight = weight * scale
    return weight


def build_label_trust_map(
    *,
    sample_rate: float,
    w_min: float = 0.25,
    gamma: float = 1.0,
    granularity: str = "window",
    hf_window_ms: float = 40.0,
    baseline_window_ms: float = 900.0,
    smooth_ms: float = 300.0,
    half_sat_ratio: float = 0.25,
    normalize: bool = True,
) -> Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """Build a ``tf.data`` map that appends an ``n0``-aware ``sample_weight``.

    The returned mapper turns a ``(input, target)`` element into
    ``(input, target, sample_weight)``, where the weight is derived from the
    inherent corruption of the *target*.

    Args:
        sample_rate: Effective sample rate of the windows in Hz.
        w_min: Floor weight for fully corrupted targets.
        gamma: Sharpness of the trust-to-weight curve.
        granularity: ``"window"`` or ``"timestep"``.
        hf_window_ms: Short moving-average window (high-frequency split).
        baseline_window_ms: Long moving-average window (baseline-drift split).
        smooth_ms: Power-ratio smoothing window.
        half_sat_ratio: Residual/signal power ratio mapped to a score of 0.5.
        normalize: Rescale weights to mean ~1 per batch.

    Returns:
        A callable suitable for ``tf.data.Dataset.map``.
    """

    def _map(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        score = estimate_corruption_score(
            y,
            sample_rate=sample_rate,
            hf_window_ms=hf_window_ms,
            baseline_window_ms=baseline_window_ms,
            smooth_ms=smooth_ms,
            half_sat_ratio=half_sat_ratio,
        )
        weight = corruption_to_weight(
            score,
            w_min=w_min,
            gamma=gamma,
            granularity=granularity,
            normalize=normalize,
        )
        return x, y, weight

    return _map
