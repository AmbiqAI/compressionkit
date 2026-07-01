"""Significance-based quantization bottleneck (SPIHT-style).

Replaces RVQ with a bottleneck that does adaptive bit allocation:
- During training: top-K masking + quantization noise (simulates SPIHT)
- During inference: hard quantization + significance-ordered bit-plane coding

The key insight: SPIHT keeps the K largest coefficients and zeros the rest.
We simulate this during training so the encoder learns to concentrate
important information into a FEW large latent values.

Training:
  1. Encoder → latent (N values)
  2. Sort by magnitude → keep top-K → zero the rest (differentiable via soft-top-K)
  3. Add uniform noise to kept values (simulates finite precision)
  4. Decoder reconstructs from sparse latent

This forces the encoder to produce SPARSE latents where only ~K values
carry all the signal information — exactly what SPIHT can code efficiently.
"""

from __future__ import annotations

import keras
import numpy as np


@keras.saving.register_keras_serializable(package="compressionkit")
class SignificanceQuantizer(keras.layers.Layer):
    """SPIHT-style top-K + quantization bottleneck.

    During training:
        - Keeps top-K latent values by magnitude (soft differentiable masking)
        - Adds uniform noise to kept values (simulates quantization)
        - Forces sparsity: encoder must put all info into K values

    During inference:
        - Hard top-K selection + scalar quantization
        - Or SPIHT-style bit-plane coding to a fixed bit budget

    Args:
        keep_ratio: Fraction of latent values to keep (top-K). E.g., 0.5 = keep
            half of values. This controls the effective bit rate.
        quant_bits: Bits per kept value for quantization noise simulation.
        rate_lambda: Weight for L1 penalty on non-top-K values (helps sparsity).
        temperature: Softmax temperature for differentiable top-K during training.
            Lower = harder selection (less gradient flow). Higher = softer.
    """

    def __init__(
        self,
        keep_ratio: float = 0.5,
        quant_bits: int = 8,
        rate_lambda: float = 0.001,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.keep_ratio = keep_ratio
        self.quant_bits = quant_bits
        self.rate_lambda = rate_lambda
        self.temperature = temperature

        # Metrics
        self._rate_metric = keras.metrics.Mean(name="sig_rate")
        self._sparsity_metric = keras.metrics.Mean(name="sig_sparsity")

    def call(
        self,
        z: keras.KerasTensor,
        training: bool = False,
        return_indices: bool = False,
    ) -> keras.KerasTensor | tuple[keras.KerasTensor, list[keras.KerasTensor]]:
        input_shape = keras.ops.shape(z)
        batch_size = input_shape[0]
        z_flat = keras.ops.reshape(z, (batch_size, -1))  # (B, N)
        # Use static shape for k computation (works in graph mode)
        static_shape = z.shape  # e.g. (None, 1, 160, 2)
        n_latents_static = 1
        for dim in static_shape[1:]:
            n_latents_static *= dim
        k = max(1, int(n_latents_static * self.keep_ratio))

        if training:
            # --- Hard top-K masking with straight-through gradient ---
            abs_z = keras.ops.abs(z_flat)

            # Find threshold: k-th largest magnitude
            top_vals = keras.ops.top_k(abs_z, k=k)[0]  # (B, k)
            threshold = keras.ops.stop_gradient(keras.ops.expand_dims(top_vals[:, -1], axis=-1))  # (B, 1)

            # Hard mask: exactly K values kept (correct forward simulation)
            hard_mask = keras.ops.cast(abs_z >= threshold, "float32")  # (B, N), exactly K ones per row

            # Soft mask for gradient flow
            soft_mask = keras.ops.sigmoid((abs_z - threshold) / (self.temperature + 1e-8))

            # STE: hard in forward, soft gradient in backward
            mask = soft_mask + keras.ops.stop_gradient(hard_mask - soft_mask)

            # Apply mask — decoder sees exactly K non-zero values
            z_masked = z_flat * mask

            # Add quantization noise to kept values only
            z_range = keras.ops.stop_gradient(keras.ops.max(keras.ops.abs(z_flat), axis=-1, keepdims=True))
            noise_scale = z_range / (2.0**self.quant_bits)
            noise = keras.random.uniform(shape=keras.ops.shape(z_flat), minval=-0.5, maxval=0.5)
            z_hat = z_masked + keras.ops.stop_gradient(noise * noise_scale * hard_mask)

            # Rate loss: penalize below-threshold values
            below_threshold = (1.0 - hard_mask) * abs_z
            rate = keras.ops.mean(below_threshold)
            self.add_loss(self.rate_lambda * rate)

            # Metrics
            self._rate_metric.update_state(rate)
            actual_sparsity = keras.ops.mean(1.0 - hard_mask)
            self._sparsity_metric.update_state(actual_sparsity)

        else:
            # --- Hard top-K at inference ---
            abs_z = keras.ops.abs(z_flat)
            top_vals = keras.ops.top_k(abs_z, k=k)[0]
            threshold = keras.ops.expand_dims(top_vals[:, -1], axis=-1)
            mask = keras.ops.cast(abs_z >= threshold, "float32")

            # Hard quantization of kept values
            z_range = keras.ops.max(keras.ops.abs(z_flat), axis=-1, keepdims=True)
            step = z_range / (2.0 ** (self.quant_bits - 1))  # step size
            z_hat = keras.ops.round(z_flat / (step + 1e-8)) * step
            z_hat = z_hat * mask  # zero out non-top-K

        z_hat = keras.ops.reshape(z_hat, input_shape)

        if return_indices:
            indices = keras.ops.cast(
                keras.ops.round(z_flat / (z_range / (2.0 ** (self.quant_bits - 1)) + 1e-8)),
                "int32",
            )
            indices = keras.ops.reshape(indices, input_shape)
            return z_hat, [indices]

        return z_hat

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        return [self._rate_metric, self._sparsity_metric]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "keep_ratio": self.keep_ratio,
                "quant_bits": self.quant_bits,
                "rate_lambda": self.rate_lambda,
                "temperature": self.temperature,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Inference codec: SPIHT-style bit-plane coding of quantized latents
# ---------------------------------------------------------------------------


def encode_significance(
    latent: np.ndarray,
    bit_budget: int,
    max_bitplanes: int = 16,
) -> tuple[np.ndarray, dict]:
    """Encode a quantized latent vector using significance-based bit-plane coding.

    Emulates SPIHT's progressive refinement on a flat vector:
    1. Find maximum magnitude → determines number of bit-planes
    2. Significance pass (MSB first): for each bit-plane, mark which positions
       are newly significant (magnitude crosses threshold)
    3. Refinement pass: refine previously significant values with next bit

    Args:
        latent: Quantized integer latent vector of shape (N,).
        bit_budget: Maximum number of bits to spend.
        max_bitplanes: Maximum bit-planes to scan.

    Returns:
        Tuple of (reconstructed_latent, metadata_dict).
    """
    N = latent.shape[0]
    latent_int = np.round(latent).astype(np.int32)
    signs = np.sign(latent_int)
    magnitudes = np.abs(latent_int)

    max_mag = int(magnitudes.max()) if magnitudes.max() > 0 else 0
    if max_mag == 0:
        return np.zeros_like(latent, dtype=np.float32), {"bits_used": 0}

    # Determine number of bit-planes
    num_planes = int(np.ceil(np.log2(max_mag + 1)))
    num_planes = min(num_planes, max_bitplanes)

    # Reconstruct progressively
    recon_mag = np.zeros(N, dtype=np.float32)
    significant = np.zeros(N, dtype=bool)  # which positions are "discovered"
    bits_used = 0

    for plane in range(num_planes - 1, -1, -1):
        threshold = 1 << plane

        # Significance pass: check unsignificant positions
        for i in range(N):
            if bits_used >= bit_budget:
                break
            if not significant[i]:
                # 1 bit: is this position significant at this plane?
                bits_used += 1
                if magnitudes[i] >= threshold:
                    significant[i] = True
                    recon_mag[i] = threshold + (threshold >> 1)  # midpoint estimate
                    # 1 bit for sign
                    bits_used += 1

        if bits_used >= bit_budget:
            break

        # Refinement pass: refine already-significant positions
        for i in range(N):
            if bits_used >= bit_budget:
                break
            if significant[i] and recon_mag[i] < (threshold << 1):
                continue  # just became significant, skip refinement this round
            if significant[i]:
                # 1 bit: refine magnitude
                bits_used += 1
                if magnitudes[i] & threshold:
                    recon_mag[i] += threshold >> 1
                else:
                    recon_mag[i] -= threshold >> 1

        if bits_used >= bit_budget:
            break

    # Apply signs
    recon = recon_mag * signs.astype(np.float32)
    return recon, {"bits_used": bits_used, "significant_count": int(significant.sum())}


def decode_significance(
    latent: np.ndarray,
    bit_budget: int,
    scale: float = 1.0,
) -> np.ndarray:
    """Convenience: encode + return reconstruction (for evaluation loops).

    Args:
        latent: Float latent vector (will be rounded to integers).
        bit_budget: Bit budget constraint.
        scale: If the latent was pre-scaled, undo scaling after decode.

    Returns:
        Reconstructed latent vector (float32).
    """
    recon, _ = encode_significance(latent / scale, bit_budget)
    return (recon * scale).astype(np.float32)
