"""Independent references from the training RVQ layer, before deployment."""

from __future__ import annotations

import keras
import numpy as np

from compressionkit.layers.ema_residual_vector_quantizer import EmaResidualVectorQuantizer
from compressionkit.layers.residual_vector_quantizer import ResidualVectorQuantizer


def build_rvq_reference(
    encoder: keras.Model,
    decoder: keras.Model,
    weights: list[np.ndarray],
    inputs: np.ndarray,
    *,
    num_levels: int,
    use_ema: bool = False,
    kmeans_init: bool = False,
) -> dict[str, np.ndarray]:
    """Restore the source quantizer and generate discrete-path reference arrays.

    This deliberately uses the training layer's weight restoration and
    encode/decode methods, independently of export extraction and runtime VQ.
    No training calls or EMA updates occur. Source-latent comparisons isolate
    codebook export from changes caused by encoder conversion/quantization.

    Args:
        encoder: Float Keras encoder from the training run.
        decoder: Float Keras decoder from the training run.
        weights: Original, ordered ``vq.get_weights()`` state.
        inputs: Nonempty batch of preprocessed input frames.
        num_levels: Trained quantizer level count.
        use_ema: Whether to restore the EMA quantizer.
        kmeans_init: Whether the EMA checkpoint has a warm-start flag.

    Returns:
        Source latents, indices, quantized latents, and decoded waveforms.
    """
    stride = 3 if use_ema else 1
    kwargs = {
        "num_levels": num_levels,
        "num_embeddings": [weights[stride * i].shape[0] for i in range(num_levels)],
        "embedding_dim": weights[0].shape[1],
    }
    vq = EmaResidualVectorQuantizer(**kwargs, kmeans_init=kmeans_init) if use_ema else ResidualVectorQuantizer(**kwargs)
    vq.build(encoder.output_shape)
    vq.set_weights(weights)
    latent = np.asarray(encoder(inputs, training=False), dtype=np.float32)
    indices = vq.encode(latent)
    quantized = keras.ops.convert_to_numpy(vq.decode(indices, latent.shape))
    decoded = decoder(quantized, training=False)
    if isinstance(decoded, dict):
        decoded = decoded.get("reconstruction", decoded.get("output"))
    return {
        "source_latents": latent,
        "source_indices": np.stack([keras.ops.convert_to_numpy(i) for i in indices], axis=-1).reshape(
            *latent.shape[:-1], num_levels
        ),
        "source_quantized_latents": np.asarray(quantized, dtype=np.float32),
        "source_reconstructions": np.asarray(decoded, dtype=np.float32),
    }
