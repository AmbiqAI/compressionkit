"""Shared paired null-augmentation helpers for training datasets.

These helpers zero both the input and target tensors so the model learns
abstention rather than inpainting when signal is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class PairedNullAugmentationConfig:
    """Configuration for paired null augmentation.

    Attributes:
        short_cutout_factor: Contiguous zero-span fraction applied to every
            sample not claimed by a stronger null regime.
        long_cutout_factor: Contiguous zero-span fraction for occasional long
            dropout regimes.
        long_cutout_prob: Per-sample probability of using the long-dropout
            regime.
        null_frame_prob: Per-sample probability of zeroing the full frame.
    """

    short_cutout_factor: tuple[float, float] | None = None
    long_cutout_factor: tuple[float, float] | None = None
    long_cutout_prob: float = 0.0
    null_frame_prob: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "long_cutout_prob", float(self.long_cutout_prob))
        object.__setattr__(self, "null_frame_prob", float(self.null_frame_prob))
        if not 0.0 <= self.long_cutout_prob <= 1.0:
            raise ValueError(f"long_cutout_prob must be in [0, 1], got {self.long_cutout_prob}")
        if not 0.0 <= self.null_frame_prob <= 1.0:
            raise ValueError(f"null_frame_prob must be in [0, 1], got {self.null_frame_prob}")
        if self.long_cutout_prob + self.null_frame_prob > 1.0:
            raise ValueError(
                "long_cutout_prob + null_frame_prob must be <= 1.0 so regimes remain exclusive"
            )
        for name, factor in (
            ("short_cutout_factor", self.short_cutout_factor),
            ("long_cutout_factor", self.long_cutout_factor),
        ):
            if factor is None:
                continue
            low, high = float(factor[0]), float(factor[1])
            if low <= 0.0 or high <= 0.0 or low > high or high > 1.0:
                raise ValueError(
                    f"{name} must satisfy 0 < low <= high <= 1.0, got {(low, high)}"
                )

    @property
    def enabled(self) -> bool:
        return (
            self.short_cutout_factor is not None
            or (self.long_cutout_factor is not None and self.long_cutout_prob > 0.0)
            or self.null_frame_prob > 0.0
        )


def _build_span_mask(
    batch_size: tf.Tensor,
    duration: tf.Tensor,
    factor: tuple[float, float],
) -> tf.Tensor:
    """Return a contiguous boolean span mask of shape ``(B, T, 1)``."""
    duration_f = tf.cast(duration, tf.float32)
    min_cut = tf.maximum(1, tf.cast(duration_f * float(factor[0]), tf.int32))
    max_cut = tf.maximum(min_cut + 1, tf.cast(duration_f * float(factor[1]), tf.int32) + 1)
    cut_size = tf.random.uniform(shape=(batch_size,), minval=min_cut, maxval=max_cut, dtype=tf.int32)
    max_start = tf.maximum(duration - cut_size + 1, 1)
    rand = tf.random.uniform(shape=(batch_size,), dtype=tf.float32)
    cut_start = tf.cast(rand * tf.cast(max_start, tf.float32), tf.int32)

    time_idx = tf.range(duration)[tf.newaxis, :]
    start_exp = cut_start[:, tf.newaxis]
    end_exp = (cut_start + cut_size)[:, tf.newaxis]
    mask = tf.logical_and(time_idx >= start_exp, time_idx < end_exp)
    return tf.expand_dims(mask, -1)


def apply_paired_null_augmentation_batch(
    x_in: tf.Tensor,
    x_tgt: tf.Tensor,
    cfg: PairedNullAugmentationConfig | None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply paired short/long/full null regimes to a batch.

    The same mask is applied to both input and target so the model is trained
    to output silence for missing signal rather than reconstruct from context.
    """
    if cfg is None or not cfg.enabled:
        return x_in, x_tgt

    shape = tf.shape(x_in)
    batch_size = shape[0]
    duration = shape[1]
    mask = tf.zeros_like(x_in, dtype=tf.bool)
    remaining = tf.ones((batch_size,), dtype=tf.bool)

    if cfg.null_frame_prob > 0.0:
        full_selector = tf.random.uniform((batch_size,), dtype=tf.float32) < cfg.null_frame_prob
        full_mask = tf.broadcast_to(full_selector[:, tf.newaxis, tf.newaxis], shape)
        mask = tf.logical_or(mask, full_mask)
        remaining = tf.logical_and(remaining, tf.logical_not(full_selector))

    if cfg.long_cutout_factor is not None and cfg.long_cutout_prob > 0.0:
        long_selector = tf.random.uniform((batch_size,), dtype=tf.float32) < cfg.long_cutout_prob
        long_selector = tf.logical_and(remaining, long_selector)
        long_mask = _build_span_mask(batch_size, duration, cfg.long_cutout_factor)
        long_mask = tf.logical_and(
            long_mask,
            tf.broadcast_to(long_selector[:, tf.newaxis, tf.newaxis], shape),
        )
        mask = tf.logical_or(mask, long_mask)
        remaining = tf.logical_and(remaining, tf.logical_not(long_selector))

    if cfg.short_cutout_factor is not None:
        short_mask = _build_span_mask(batch_size, duration, cfg.short_cutout_factor)
        short_mask = tf.logical_and(
            short_mask,
            tf.broadcast_to(remaining[:, tf.newaxis, tf.newaxis], shape),
        )
        mask = tf.logical_or(mask, short_mask)

    x_in = tf.where(mask, tf.zeros_like(x_in), x_in)
    x_tgt = tf.where(mask, tf.zeros_like(x_tgt), x_tgt)
    return x_in, x_tgt


def apply_paired_cutout_batch(
    x_in: tf.Tensor,
    x_tgt: tf.Tensor,
    factor: tuple[float, float],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Backward-compatible short paired cutout wrapper."""
    return apply_paired_null_augmentation_batch(
        x_in,
        x_tgt,
        PairedNullAugmentationConfig(short_cutout_factor=factor),
    )


__all__ = [
    "PairedNullAugmentationConfig",
    "apply_paired_cutout_batch",
    "apply_paired_null_augmentation_batch",
]
