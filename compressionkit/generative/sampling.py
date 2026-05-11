"""Autoregressive sampling and RVQ-token → signal decoding utilities."""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _top_k_logits(logits: np.ndarray, k: int) -> np.ndarray:
    """Mask all but the top-*k* entries along the last axis to ``-inf``."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    # Keep top-k per row
    kth = np.partition(logits, -k, axis=-1)[..., -k, np.newaxis]
    masked = np.where(logits < kth, -np.inf, logits)
    return masked


def sample_tokens(
    prior: keras.Model,
    *,
    num_samples: int,
    context_length: int,
    vocab_size: int,
    seed_tokens: np.ndarray | None = None,
    temperature: float = 1.0,
    top_k: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Autoregressively sample token sequences from *prior*.

    Uses a fixed-length context buffer — at each step the buffer is
    passed through the model and only the logits at the current position
    are used. Positions beyond the current step are ignored. This keeps
    the model call signature static (good for deployment), at the cost
    of recomputing attention each step (fine for small contexts).

    Args:
        prior: Causal LM from :func:`build_prior`.
        num_samples: Batch size of sequences to draw.
        context_length: Sequence length (must match prior input).
        vocab_size: Codebook size.
        seed_tokens: Optional ``(num_samples, L0)`` array of ``L0 <
            context_length`` priming tokens; if ``None``, each sequence
            starts at a uniformly-random token.
        temperature: Softmax temperature.
        top_k: Restrict sampling to top-k logits (0 disables).
        rng: Numpy random generator.

    Returns:
        Sampled token array of shape ``(num_samples, context_length)`` and dtype int32.
    """
    rng = rng if rng is not None else np.random.default_rng()
    buf = np.zeros((num_samples, context_length), dtype=np.int32)
    if seed_tokens is None:
        buf[:, 0] = rng.integers(0, vocab_size, size=num_samples)
        start = 1
    else:
        seed_tokens = np.asarray(seed_tokens, dtype=np.int32)
        if seed_tokens.shape[0] != num_samples:
            raise ValueError("seed_tokens must have leading dim == num_samples")
        L0 = seed_tokens.shape[1]
        if context_length <= L0:
            raise ValueError("seed sequence must be shorter than context_length")
        buf[:, :L0] = seed_tokens
        start = L0

    for t in range(start, context_length):
        logits = np.asarray(prior(buf, training=False))  # (B, L, V)
        step_logits = logits[:, t - 1, :] / max(float(temperature), 1e-6)
        step_logits = _top_k_logits(step_logits, top_k)
        probs = _softmax_np(step_logits)
        for i in range(num_samples):
            buf[i, t] = rng.choice(vocab_size, p=probs[i])
    return buf


def decode_tokens_to_signal(
    model: keras.Model,
    tokens: np.ndarray,
    *,
    frame_size: int,
    tokens_per_frame: int,
    embedding_dim: int,
    num_leads: int = 1,
) -> np.ndarray:
    """Decode a token sequence into a (normalised) waveform.

    The tokens are split into ``ceil(L / tokens_per_frame)`` frames,
    each dequantised via ``model.vq.decode`` and decoded by
    ``model.decoder``. Frames are concatenated to produce a single 1-D
    signal per sample. The output is in layer-normalised amplitude
    space (zero mean, unit std per generated frame).

    Args:
        model: Trained :class:`VQAutoencoder`.
        tokens: ``(num_samples, total_tokens)`` int token ids. ``total_tokens``
            must be a multiple of ``tokens_per_frame``.
        frame_size: Decoder output frame size (samples).
        tokens_per_frame: Latent positions per frame.
        embedding_dim: RVQ embedding dimensionality.
        num_leads: Decoder output channels.

    Returns:
        Signal array of shape ``(num_samples, num_frames * frame_size)``.
    """
    tokens = np.asarray(tokens, dtype=np.int32)
    if tokens.shape[-1] % tokens_per_frame != 0:
        raise ValueError(
            f"tokens length {tokens.shape[-1]} is not a multiple of "
            f"tokens_per_frame={tokens_per_frame}"
        )
    num_samples = tokens.shape[0]
    num_frames = tokens.shape[-1] // tokens_per_frame

    # One frame per row → (num_samples * num_frames, tokens_per_frame)
    frame_tokens = tokens.reshape(num_samples * num_frames, tokens_per_frame)
    flat_idx = frame_tokens.reshape(-1)
    with tf.device("/CPU:0"):
        zq_flat = np.asarray(
            model.vq.decode([flat_idx], (flat_idx.size, embedding_dim)),
            dtype=np.float32,
        )
    # Reshape to decoder's expected latent layout: (N, 1, T_latent, D)
    zq = zq_flat.reshape(num_samples * num_frames, 1, tokens_per_frame, embedding_dim)
    with tf.device("/CPU:0"):
        recon = np.asarray(model.decoder(zq, training=False), dtype=np.float32)
    # Decoder output shape: (N, 1, frame_size, num_leads) — take first lead
    recon = recon[:, 0, :, 0] if num_leads == 1 else recon[:, 0, :, :].mean(axis=-1)
    signals = recon.reshape(num_samples, num_frames * frame_size)
    return signals


def sample_signals(
    *,
    prior: keras.Model,
    compressor: keras.Model,
    num_samples: int,
    num_frames: int,
    tokens_per_frame: int,
    vocab_size: int,
    embedding_dim: int,
    frame_size: int,
    num_leads: int = 1,
    temperature: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
) -> np.ndarray:
    """End-to-end: sample tokens from *prior* and decode through *compressor*."""
    rng = np.random.default_rng(seed)
    context_length = num_frames * tokens_per_frame
    tokens = sample_tokens(
        prior,
        num_samples=num_samples,
        context_length=context_length,
        vocab_size=vocab_size,
        temperature=temperature,
        top_k=top_k,
        rng=rng,
    )
    return decode_tokens_to_signal(
        compressor,
        tokens,
        frame_size=frame_size,
        tokens_per_frame=tokens_per_frame,
        embedding_dim=embedding_dim,
        num_leads=num_leads,
    )


__all__ = ["decode_tokens_to_signal", "sample_signals", "sample_tokens"]
