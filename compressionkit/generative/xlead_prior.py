"""Cross-channel (multi-lead) entropy prior architectures.

Two prior types that exploit inter-lead correlations in 12-lead ECG:

1. ``xlead_concat`` — embed each lead independently with a shared
   codebook + lead-positional embedding, then run a standard causal
   Conv1D stack. Per-lead output heads predict each lead's next token.

2. ``xlead_interleave`` — interleave per-lead token streams into one
   sequence (``[t0_lead0, t0_lead1, ..., t0_lead11, t1_lead0, ...]``)
   with lead-id embedding. A standard causal Conv1D sees cross-lead
   conditioning automatically.

Both are INT8-portable (only Conv1D, Dense, Embedding, LayerNorm, Add,
ReLU) and intended for deployment via LiteRT.
"""

from __future__ import annotations

import keras
import numpy as np


def build_xlead_concat_prior(
    *,
    vocab_size: int,
    context_length: int,
    num_leads: int = 12,
    embed_dim: int = 48,
    num_layers: int = 4,
    kernel_size: int = 5,
    dropout: float = 0.0,
    name: str = "xlead_concat_prior",
) -> keras.Model:
    """Concatenated multi-lead causal Conv1D prior.

    Input: ``(B, context_length, num_leads)`` int32 tokens (each lead's
    token stream at each time step).

    Output: ``(B, context_length, num_leads, vocab_size)`` logits — per-lead
    next-token predictions.

    Architecture:
        - Shared token embedding (vocab_size → embed_dim).
        - Learned lead embedding (num_leads → embed_dim), added per position.
        - Reshape to (B, T, num_leads * embed_dim) for the conv stack.
        - Causal dilated Conv1D stack (same as single-lead CNN prior).
        - Per-lead Dense head splits output back to (B, T, num_leads, V).
    """
    # Input: (B, T, num_leads)
    tokens_in = keras.Input(shape=(context_length, num_leads), dtype="int32", name="tokens")

    # Shared token embedding: (B, T, L) → (B, T, L, E)
    tok_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="token_embedding")(tokens_in)

    # Lead embedding: (L,) → (1, 1, L, E) broadcast over (B, T)
    lead_ids = keras.ops.arange(num_leads)  # (L,)
    lead_emb_layer = keras.layers.Embedding(input_dim=num_leads, output_dim=embed_dim, name="lead_embedding")
    lead_emb = lead_emb_layer(lead_ids)  # (L, E)
    # Broadcast add: (B, T, L, E) + (L, E) → (B, T, L, E)
    x = tok_emb + lead_emb

    # Flatten leads into channel dim: (B, T, L*E)
    x = keras.layers.Reshape((context_length, num_leads * embed_dim), name="flatten_leads")(x)

    # Causal dilated Conv1D stack
    for i in range(num_layers):
        x = keras.layers.Conv1D(
            filters=num_leads * embed_dim,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=2**i,
            activation="relu",
            name=f"causal_conv_d{2**i}",
        )(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"drop_d{2**i}")(x)

    x = keras.layers.LayerNormalization(epsilon=1e-5, name="final_ln")(x)

    # Per-lead output: (B, T, L*E) → (B, T, L, V)
    # Dense per lead — reshape then apply shared head
    x = keras.layers.Reshape((context_length, num_leads, embed_dim), name="unflatten_leads")(x)
    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)  # (B, T, L, V)

    return keras.Model(tokens_in, logits, name=name)


def build_xlead_interleave_prior(
    *,
    vocab_size: int,
    context_length: int,
    num_leads: int = 12,
    embed_dim: int = 48,
    num_layers: int = 4,
    kernel_size: int = 5,
    dropout: float = 0.0,
    name: str = "xlead_interleave_prior",
) -> keras.Model:
    """Interleaved multi-lead causal Conv1D prior.

    Input: ``(B, context_length * num_leads)`` int32 — interleaved as
    ``[t0_L0, t0_L1, ..., t0_L11, t1_L0, ...]``.

    Output: ``(B, context_length * num_leads, vocab_size)`` logits.

    Architecture:
        - Token embedding + lead-id embedding (cyclically assigned).
        - Standard positional offset not needed since dilated convs with
          causal padding implicitly encode position.
        - Causal Conv1D stack (context is 12× longer, so cross-lead
          conditioning is automatic: predicting lead-N at time-t uses
          leads <N at time-t that precede it in the sequence).
    """
    seq_len = context_length * num_leads
    tokens_in = keras.Input(shape=(seq_len,), dtype="int32", name="tokens")

    # Token embedding
    x = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="token_embedding")(
        tokens_in
    )  # (B, S, E)

    # Lead-id embedding: cyclically assigned position % num_leads
    lead_ids = np.tile(np.arange(num_leads, dtype=np.int32), context_length)  # (S,)
    lead_id_const = keras.ops.convert_to_tensor(lead_ids)  # static
    lead_emb_layer = keras.layers.Embedding(input_dim=num_leads, output_dim=embed_dim, name="lead_embedding")
    lead_emb = lead_emb_layer(lead_id_const)  # (S, E)
    x = x + lead_emb  # (B, S, E)

    # Causal Conv1D stack
    for i in range(num_layers):
        x = keras.layers.Conv1D(
            filters=embed_dim,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=2**i,
            activation="relu",
            name=f"causal_conv_d{2**i}",
        )(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"drop_d{2**i}")(x)

    x = keras.layers.LayerNormalization(epsilon=1e-5, name="final_ln")(x)
    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)  # (B, S, V)

    return keras.Model(tokens_in, logits, name=name)


def interleave_leads(tokens: np.ndarray) -> np.ndarray:
    """Interleave per-lead tokens for the xlead_interleave prior.

    Args:
        tokens: (num_leads, N_frames, tpf, levels) from per-lead extraction,
            or (num_leads, seq_len) if already flattened per lead.

    Returns:
        (seq_len * num_leads,) or (N, seq_len * num_leads) interleaved 1-D stream.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim == 4:
        # (L, N, tpf, levels) → (L, N*tpf) taking level 0
        num_leads, n_frames, tpf, _ = tokens.shape
        flat = tokens[:, :, :, 0].reshape(num_leads, n_frames * tpf)
    elif tokens.ndim == 2:
        flat = tokens  # (L, S)
    else:
        raise ValueError(f"Expected 2-D or 4-D tokens, got shape {tokens.shape}")

    num_leads, seq_len = flat.shape
    # Transpose to (S, L) then flatten → interleaved
    return flat.T.reshape(-1)


def deinterleave_leads(interleaved: np.ndarray, num_leads: int) -> np.ndarray:
    """Reverse of interleave_leads.

    Args:
        interleaved: (seq_len * num_leads,) flat interleaved stream.
        num_leads: Number of leads.

    Returns:
        (num_leads, seq_len) array.
    """
    interleaved = np.asarray(interleaved)
    total = interleaved.size
    seq_len = total // num_leads
    # Reshape (S, L) then transpose → (L, S)
    return interleaved.reshape(seq_len, num_leads).T


__all__ = [
    "build_xlead_concat_prior",
    "build_xlead_interleave_prior",
    "deinterleave_leads",
    "interleave_leads",
]
