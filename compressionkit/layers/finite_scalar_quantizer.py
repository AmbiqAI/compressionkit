"""Finite Scalar Quantization (FSQ).

Paper: "Finite Scalar Quantization: VQ-VAE Made Simple",
       Mentzer et al., ICLR 2024.  https://arxiv.org/abs/2309.15505

A drop-in replacement for ``VectorQuantizer`` / ``ResidualVectorQuantizer``
that:

- Quantizes each latent dimension independently to a small fixed number of
  levels.  The implicit codebook is the Cartesian product of per-dim levels
  (e.g. ``[8, 5, 5, 5]`` => 1000 codes).
- Has **no learned codebook**, no commitment loss, no EMA updates, and no
  codebook-collapse pathology.
- Is highly edge-friendly: the only ops needed at inference are ``tanh``,
  scalar multiply, ``round``, and per-dim integer modulo for index packing.
  No codebook table needs to be shipped on-device.

Designed to be portable to ``helia_edge.layers`` later — it depends only on
``keras`` and uses ``keras.ops`` exclusively (no backend-specific code).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import keras
from keras import ops


def _round_ste(x):
    """Round to nearest integer with a straight-through estimator gradient."""
    return x + ops.stop_gradient(ops.round(x) - x)


class FiniteScalarQuantizer(keras.layers.Layer):
    """Finite Scalar Quantization bottleneck.

    The layer accepts a continuous latent ``z`` of shape ``(..., D)`` where
    ``D == len(levels)``, bounds it to a per-dimension interval, rounds to one
    of ``L_d`` discrete levels per dimension, and returns the rescaled
    quantized latent in roughly ``[-1, 1]`` per dimension.

    The straight-through estimator is used so gradients flow through the
    quantization step.  No commitment or codebook loss is added.

    Args:
        levels: Per-dimension number of quantization levels.  Each must be
            ``>= 2``.  The latent dimensionality is ``D = len(levels)``.  Total
            implicit codebook size is ``prod(levels)``.  The FSQ paper
            recommends mildly asymmetric values such as ``[8, 5, 5, 5]`` for
            ~1k codes or ``[8, 8, 8, 5, 5, 5]`` for ~64k codes.
        eps: Numerical-stability epsilon used when bounding the latent.

    Drop-in compatibility with ``helia_edge.trainers.VQAutoencoder``:
        ``call(z, training=None, return_indices=False)`` returns either the
        quantized latent (same shape as input) or ``(zhat, indices)`` where
        ``indices`` has shape ``(..., D)`` of int32, one per dimension.
    """

    def __init__(
        self,
        levels: Sequence[int],
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(levels, (list, tuple)) or len(levels) == 0:
            raise ValueError("levels must be a non-empty list/tuple")
        if not all(int(L) >= 2 for L in levels):
            raise ValueError("each level must be >= 2")
        self.levels: list[int] = [int(L) for L in levels]
        self.D: int = len(self.levels)
        self.eps: float = float(eps)

        # Implicit codebook size and per-dimension info bits
        self._codebook_size: int = int(math.prod(self.levels))
        self._bits_per_dim: list[float] = [math.log2(L) for L in self.levels]
        self._bits_per_index: float = float(sum(self._bits_per_dim))

        # Per-batch metrics (mean over batches inside an epoch)
        self._perplexity = keras.metrics.Mean(name="fsq_perplexity_mean")
        self._usage = keras.metrics.Mean(name="fsq_usage_mean")
        self._bpi = keras.metrics.Mean(name="fsq_bits_per_index_sum")

    # ------------------------------------------------------------------
    # Public properties (helpful for callers / logging)
    # ------------------------------------------------------------------
    @property
    def embedding_dim(self) -> int:
        return self.D

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def bits_per_index(self) -> float:
        """Theoretical max bits per latent position (sum of log2(L_d))."""
        return self._bits_per_index

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, input_shape):
        last = input_shape[-1]
        if last is not None and int(last) != self.D:
            raise ValueError(
                f"FSQ input last dim {int(last)} does not match D={self.D} "
                f"(levels={self.levels})"
            )
        super().build(input_shape)

    # ------------------------------------------------------------------
    # Core FSQ math (matches the official Algorithm 1)
    # ------------------------------------------------------------------
    def _bound(self, z):
        """Smoothly bound ``z`` to the open interval covering the levels."""
        levels = ops.convert_to_tensor(self.levels, dtype=self.compute_dtype)
        half_l = (levels - 1.0) * (1.0 - self.eps) / 2.0
        # For even L, shift so 0 sits between two levels (avoids "dead" zero)
        is_even = ops.cast(ops.equal(ops.mod(levels, 2.0), 0.0), self.compute_dtype)
        offset = is_even * 0.5
        shift = ops.tan(offset / half_l)
        return ops.tanh(z + shift) * half_l - offset

    def _quantize(self, z):
        """Quantize ``z`` to discrete codes, returned in roughly ``[-1, 1]``."""
        bounded = self._bound(z)
        rounded = _round_ste(bounded)
        levels = ops.convert_to_tensor(self.levels, dtype=self.compute_dtype)
        # Rescale by half-width so output is in [-1, 1]
        half_width = ops.floor(levels / 2.0)
        return rounded / half_width

    def _to_int_levels(self, zhat):
        """Convert quantized ``zhat`` (in [-1, 1]) to integer levels [0, L-1]."""
        levels = ops.convert_to_tensor(self.levels, dtype=self.compute_dtype)
        half_width = ops.floor(levels / 2.0)
        zint = ops.round(zhat * half_width + half_width)
        zint = ops.clip(zint, 0.0, levels - 1.0)
        return ops.cast(zint, "int32")

    # ------------------------------------------------------------------
    # Layer API
    # ------------------------------------------------------------------
    def call(self, x, training=None, return_indices=False):
        x = ops.convert_to_tensor(x, dtype=self.compute_dtype)
        zhat = self._quantize(x)

        # Per-dim metrics — sum-of-marginal-entropies upper-bounds true entropy
        # but is far cheaper than enumerating prod(levels) codes.
        zint = self._to_int_levels(zhat)  # (..., D), int32
        flat_zint = ops.reshape(zint, (-1, self.D))

        eps_t = ops.convert_to_tensor(1e-10, dtype=self.compute_dtype)
        log2 = ops.log(ops.convert_to_tensor(2.0, self.compute_dtype))

        bpi_total = ops.convert_to_tensor(0.0, dtype=self.compute_dtype)
        perp_sum = ops.convert_to_tensor(0.0, dtype=self.compute_dtype)
        usage_sum = ops.convert_to_tensor(0.0, dtype=self.compute_dtype)
        for d, L in enumerate(self.levels):
            col = flat_zint[:, d]
            one_hot = ops.one_hot(col, L)
            p = ops.mean(ops.cast(one_hot, self.compute_dtype), axis=0)
            H = -ops.sum(p * (ops.log(p + eps_t) / log2))
            bpi_total = bpi_total + H
            perp_sum = perp_sum + ops.exp(H * log2)
            usage_sum = usage_sum + ops.sum(ops.cast(p > 0, self.compute_dtype)) / float(L)

        self._bpi.update_state(bpi_total)
        self._perplexity.update_state(perp_sum / float(self.D))
        self._usage.update_state(usage_sum / float(self.D))

        if return_indices:
            return zhat, zint
        return zhat

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self):
        config = super().get_config()
        config.update({"levels": list(self.levels), "eps": self.eps})
        return config
