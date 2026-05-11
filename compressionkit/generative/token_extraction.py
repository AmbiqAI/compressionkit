"""Extract RVQ token sequences from a trained compression model.

Given a frozen encoder + RVQ module, slice a corpus of signals into
non-overlapping frames, layer-normalise each frame, and dump the
resulting per-frame codebook index sequences. This is the training
corpus for a generative prior.

The output dtype is ``int16`` (sufficient for any sane codebook size up
to 32k) and the layout is deliberately simple: ``tokens`` shaped
``(num_frames, tokens_per_frame, num_levels)``. For the common
single-level (``num_levels == 1``) case, callers usually want
``tokens[..., 0]``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf


def _normalised_frames(signal: np.ndarray, frame_size: int, epsilon: float) -> np.ndarray:
    """Slice *signal* into contiguous layer-normalised frames."""
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    n_frames = sig.size // frame_size
    if n_frames == 0:
        return np.zeros((0, frame_size), dtype=np.float32)
    usable = sig[: n_frames * frame_size].reshape(n_frames, frame_size)
    means = usable.mean(axis=1, keepdims=True)
    stds = usable.std(axis=1, keepdims=True) + epsilon
    return (usable - means) / stds


def extract_rvq_tokens(
    model: keras.Model,
    signals: Iterable[np.ndarray],
    *,
    frame_size: int,
    num_leads: int = 1,
    epsilon: float = 1e-3,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract per-frame codebook tokens from a trained model.

    Args:
        model: Trained :class:`VQAutoencoder` with ``.encoder`` and ``.vq``.
        signals: Iterable of 1-D (or (T, C)) numpy signals.
        frame_size: Model frame size in samples.
        num_leads: Channels per frame (1 for single-lead ECG/PPG).
        epsilon: Layer-norm epsilon (must match training).
        batch_size: Frames per forward pass.

    Returns:
        ``tokens`` array of shape ``(total_frames, tokens_per_frame,
        num_levels)`` with dtype ``int16``.
    """
    encoder = model.encoder
    vq = model.vq

    all_frames: list[np.ndarray] = []
    for sig in signals:
        arr = np.asarray(sig, dtype=np.float32)
        if arr.ndim == 2:
            # Use first lead if multi-channel (keeps this helper simple)
            arr = arr[:, 0]
        frames = _normalised_frames(arr, frame_size, epsilon)
        if frames.size:
            all_frames.append(frames)

    if not all_frames:
        return np.zeros((0, 0, 0), dtype=np.int16)

    stacked = np.concatenate(all_frames, axis=0)  # (N, frame_size)
    # Shape expected by encoder: (N, 1, frame_size, num_leads)
    batched_in = stacked[:, np.newaxis, :, np.newaxis]
    if num_leads > 1:
        batched_in = np.repeat(batched_in, num_leads, axis=-1)

    # Run encoder in mini-batches to keep memory bounded
    z_chunks: list[np.ndarray] = []
    for start in range(0, batched_in.shape[0], batch_size):
        chunk = batched_in[start : start + batch_size]
        with tf.device("/CPU:0"):
            z = encoder(chunk, training=False)
        z_chunks.append(np.asarray(z))
    z_all = np.concatenate(z_chunks, axis=0)

    # Latent layout from the 1-D encoder: (N, 1, T_latent, D)
    latent_shape = z_all.shape
    with tf.device("/CPU:0"):
        indices_list = vq.encode(z_all)
    tokens_per_frame = int(np.prod(latent_shape[:-1])) // latent_shape[0]

    out = np.zeros(
        (latent_shape[0], tokens_per_frame, len(indices_list)),
        dtype=np.int16,
    )
    for level, idx in enumerate(indices_list):
        idx_np = np.asarray(idx).reshape(latent_shape[0], tokens_per_frame).astype(np.int16)
        out[..., level] = idx_np
    return out


def save_tokens(out_path: Path, tokens: np.ndarray, *, meta: dict) -> None:
    """Persist tokens plus a small metadata dict as a single ``.npz``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, tokens=tokens, **meta)


__all__ = ["extract_rvq_tokens", "save_tokens"]
