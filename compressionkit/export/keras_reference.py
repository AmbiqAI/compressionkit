"""Float Keras decoder references for export-time validation."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf


def load_keras_reference_decoder(path: str | Path) -> keras.Model:
    """Load a decoder companion on CPU for full float32 reference math."""
    with tf.device("/CPU:0"):
        return keras.models.load_model(path)


def decode_keras_reference(decoder: keras.Model, latents: np.ndarray) -> np.ndarray:
    """Decode a latent batch with float reference math, without GPU TF32."""
    with tf.device("/CPU:0"):
        output = decoder(latents, training=False)
        if isinstance(output, dict):
            output = output.get("reconstruction", output.get("output"))
        return np.asarray(output, dtype=np.float32)
