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


def _encode_frames(
    encoder: keras.Model,
    vq: keras.Model,
    stacked: np.ndarray,
    *,
    num_leads: int,
    batch_size: int,
) -> np.ndarray:
    """Run encoder + VQ on stacked frames and return int16 tokens.

    Args:
        stacked: (N, frame_size) float32 frames.

    Returns:
        (N, tokens_per_frame, num_levels) int16 token array.
    """
    # Shape expected by encoder: (N, 1, frame_size, num_leads)
    batched_in = stacked[:, np.newaxis, :, np.newaxis]
    if num_leads > 1:
        batched_in = np.repeat(batched_in, num_leads, axis=-1)

    token_chunks: list[np.ndarray] = []
    for start in range(0, batched_in.shape[0], batch_size):
        chunk = batched_in[start : start + batch_size]
        with tf.device("/CPU:0"):
            z = encoder(chunk, training=False)
        z_np = np.asarray(z)
        latent_shape = z_np.shape
        with tf.device("/CPU:0"):
            indices_list = vq.encode(z_np)
        tokens_per_frame = int(np.prod(latent_shape[:-1])) // latent_shape[0]

        chunk_tokens = np.zeros(
            (latent_shape[0], tokens_per_frame, len(indices_list)),
            dtype=np.int16,
        )
        for level, idx in enumerate(indices_list):
            idx_np = np.asarray(idx).reshape(latent_shape[0], tokens_per_frame).astype(np.int16)
            chunk_tokens[..., level] = idx_np
        token_chunks.append(chunk_tokens)

    return np.concatenate(token_chunks, axis=0)


def extract_rvq_tokens(
    model: keras.Model,
    signals: Iterable[np.ndarray],
    *,
    frame_size: int,
    num_leads: int = 1,
    per_lead: bool = False,
    epsilon: float = 1e-3,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract per-frame codebook tokens from a trained model.

    Args:
        model: Trained :class:`VQAutoencoder` with ``.encoder`` and ``.vq``.
        signals: Iterable of 1-D (or (T, C)) numpy signals.
        frame_size: Model frame size in samples.
        num_leads: Channels per frame (1 for single-lead ECG/PPG).
        per_lead: If True and signals have multiple columns, encode each
            lead independently and return shape
            ``(num_leads, total_frames, tokens_per_frame, num_levels)``.
            When False (default), only the first column is used.
        epsilon: Layer-norm epsilon (must match training).
        batch_size: Frames per forward pass.

    Returns:
        When ``per_lead=False``: ``(total_frames, tokens_per_frame,
        num_levels)`` int16. When ``per_lead=True``: ``(num_leads,
        total_frames, tokens_per_frame, num_levels)`` int16.
    """
    encoder = model.encoder
    vq = model.vq

    if per_lead:
        return _extract_per_lead(encoder, vq, signals, frame_size=frame_size, epsilon=epsilon, batch_size=batch_size)

    all_frames: list[np.ndarray] = []
    for sig in signals:
        arr = np.asarray(sig, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, 0]
        frames = _normalised_frames(arr, frame_size, epsilon)
        if frames.size:
            all_frames.append(frames)

    if not all_frames:
        return np.zeros((0, 0, 0), dtype=np.int16)

    stacked = np.concatenate(all_frames, axis=0)
    return _encode_frames(encoder, vq, stacked, num_leads=num_leads, batch_size=batch_size)


def _extract_per_lead(
    encoder: keras.Model,
    vq: keras.Model,
    signals: Iterable[np.ndarray],
    *,
    frame_size: int,
    epsilon: float,
    batch_size: int,
) -> np.ndarray:
    """Per-lead extraction: returns (num_leads, N_frames, tpf, levels)."""
    # Group frames by lead
    lead_frames: dict[int, list[np.ndarray]] = {}
    for sig in signals:
        arr = np.asarray(sig, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        n_leads = arr.shape[1]
        for lead_idx in range(n_leads):
            frames = _normalised_frames(arr[:, lead_idx], frame_size, epsilon)
            if frames.size:
                lead_frames.setdefault(lead_idx, []).append(frames)

    if not lead_frames:
        return np.zeros((0, 0, 0, 0), dtype=np.int16)

    # Encode each lead independently (single-lead encoder applied per lead)
    lead_tokens: list[np.ndarray] = []
    for lead_idx in sorted(lead_frames.keys()):
        stacked = np.concatenate(lead_frames[lead_idx], axis=0)
        tokens = _encode_frames(encoder, vq, stacked, num_leads=1, batch_size=batch_size)
        lead_tokens.append(tokens)

    return np.stack(lead_tokens, axis=0)


def save_tokens(out_path: Path, tokens: np.ndarray, *, meta: dict) -> None:
    """Persist tokens plus a small metadata dict as a single ``.npz``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, tokens=tokens, **meta)


__all__ = ["extract_rvq_tokens", "save_tokens"]
