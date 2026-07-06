"""Causal token-prior architectures for RVQ entropy priors.

These are lightweight, INT8/LiteRT-portable causal sequence models that
predict next-token logits over an RVQ codebook stream. They share a
uniform interface — input ``(B, context_length)`` int32 tokens, output
``(B, context_length, vocab_size)`` logits — so the trainer
(:mod:`compressionkit.trainers.rvq_prior`) and entropy-coding runtime
(:mod:`compressionkit.runtime.entropy_algorithms`) are backbone-agnostic.

``build_wavenet_prior`` (gated-residual dilated causal Conv1D, WaveNet-style)
is the empirically-strongest architecture found in this repo's entropy-prior
sweeps (see repo docs/notes) — prefer it as the default unless a specific
deployment constraint calls for a smaller/cheaper alternative.

``build_wavenet_bit_prior`` is a related but distinct architecture for
SPIHT's bit-level entropy stream (context-class/bitplane/previous-bit side
info rather than a single token vocabulary) — see
:mod:`compressionkit.generative.spiht_bit_prior` for its numpy inference
counterpart and pluggable AC sink/source.
"""

from __future__ import annotations

import keras

__all__ = [
    "SplitHalf",
    "build_cnn_prior",
    "build_cnngru_prior",
    "build_dscnn_prior",
    "build_gru_prior",
    "build_hybrid_prior",
    "build_wavenet_bit_prior",
    "build_wavenet_prior",
]


@keras.saving.register_keras_serializable(package="compressionkit")
class SplitHalf(keras.layers.Layer):
    """Splits a tensor's last axis into two equal halves: ``(..., 2*n) -> (..., n), (..., n)``.

    Used by the gated-residual WaveNet blocks below to split a ``2*embed_dim``
    causal-conv output into its ``(filter, gate)`` halves. Replaces an earlier
    ``keras.layers.Lambda(lambda t: ...)`` — a Python closure that pickles its
    bytecode when serialized and reliably FAILS to deserialize with
    ``model.save()``/``keras.models.load_model()`` (even with
    ``safe_mode=False``). A real layer with ``get_config()`` has none of that
    fragility.
    """

    def __init__(self, split_size: int, **kwargs):
        super().__init__(**kwargs)
        self.split_size = int(split_size)

    def call(self, x):
        return x[..., : self.split_size], x[..., self.split_size :]

    def compute_output_shape(self, input_shape):
        half = (*input_shape[:-1], self.split_size)
        return half, half

    def get_config(self):
        config = super().get_config()
        config.update({"split_size": self.split_size})
        return config


def build_cnn_prior(
    *,
    vocab_size: int,
    context_length: int,
    embed_dim: int = 48,
    num_layers: int = 4,
    kernel_size: int = 5,
    dropout: float = 0.0,
    name: str = "rvq_cnn_prior",
) -> keras.Model:
    """Causal dilated Conv1D prior (LiteRT-portable, INT8-friendly).

    Each conv layer has dilation ``2**i`` for layer ``i``, giving a
    receptive field of ``1 + (k-1) * (2**num_layers - 1)`` tokens.
    """
    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="token_embedding")(tokens_in)
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
    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)
    return keras.Model(tokens_in, logits, name=name)


def build_dscnn_prior(
    *,
    vocab_size: int,
    context_length: int,
    embed_dim: int = 64,
    proj_dim: int | None = None,
    num_layers: int = 5,
    kernel_size: int = 5,
    dropout: float = 0.0,
    norm: str = "layer",
    name: str = "rvq_dscnn_prior",
) -> keras.Model:
    """Lean depthwise-separable causal Conv1D prior for endpoint AI.

    Each dilated Conv1D becomes ``DepthwiseConv1D + pointwise Conv1D`` —
    same receptive field as :func:`build_cnn_prior`, ~80% fewer params
    per layer. Optional factored embedding when ``proj_dim < embed_dim``.
    """
    if proj_dim is None:
        proj_dim = embed_dim
    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(input_dim=vocab_size, output_dim=proj_dim, name="token_embedding")(tokens_in)
    if proj_dim != embed_dim:
        x = keras.layers.Dense(embed_dim, name="embed_proj")(x)
    for i in range(num_layers):
        d = 2**i
        x = keras.layers.ZeroPadding1D(padding=((kernel_size - 1) * d, 0), name=f"causal_pad_d{d}")(x)
        x = keras.layers.DepthwiseConv1D(kernel_size=kernel_size, padding="valid", dilation_rate=d, name=f"dw_d{d}")(x)
        x = keras.layers.Conv1D(filters=embed_dim, kernel_size=1, activation="relu", name=f"pw_d{d}")(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"drop_d{d}")(x)
    if norm == "layer":
        x = keras.layers.LayerNormalization(epsilon=1e-5, name="final_ln")(x)
    elif norm != "none":
        raise ValueError(f"Unknown norm={norm!r}; expected 'layer' or 'none'.")
    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)
    return keras.Model(tokens_in, logits, name=name)


def build_hybrid_prior(
    *,
    vocab_size: int,
    context_length: int,
    embed_dim: int = 64,
    proj_dim: int | None = None,
    stem_layers: int = 2,
    ds_layers: int = 4,
    kernel_size: int = 5,
    dropout: float = 0.0,
    name: str = "rvq_hybrid_prior",
) -> keras.Model:
    """Hybrid causal prior: dense Conv1D stem + residual DSConv body.

    First ``stem_layers`` blocks are full causal Conv1D (free channel
    mixing near the token embedding); remaining ``ds_layers`` are
    residual DepthwiseConv1D + pointwise Conv1D. Every conv is followed
    by ``BatchNormalization`` (foldable into the preceding conv at INT8
    export time).
    """
    if proj_dim is None:
        proj_dim = embed_dim
    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(input_dim=vocab_size, output_dim=proj_dim, name="token_embedding")(tokens_in)
    if proj_dim != embed_dim:
        x = keras.layers.Dense(embed_dim, name="embed_proj")(x)

    layer_idx = 0
    for _ in range(stem_layers):
        d = 2**layer_idx
        x = keras.layers.Conv1D(
            filters=embed_dim,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=d,
            use_bias=False,
            name=f"stem_conv_d{d}",
        )(x)
        x = keras.layers.BatchNormalization(name=f"stem_bn_d{d}")(x)
        x = keras.layers.ReLU(name=f"stem_relu_d{d}")(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"stem_drop_d{d}")(x)
        layer_idx += 1

    for _ in range(ds_layers):
        d = 2**layer_idx
        residual = x
        y = keras.layers.ZeroPadding1D(padding=((kernel_size - 1) * d, 0), name=f"ds_pad_d{d}")(x)
        y = keras.layers.DepthwiseConv1D(
            kernel_size=kernel_size, padding="valid", dilation_rate=d, use_bias=False, name=f"ds_dw_d{d}"
        )(y)
        y = keras.layers.BatchNormalization(name=f"ds_dw_bn_d{d}")(y)
        y = keras.layers.ReLU(name=f"ds_dw_relu_d{d}")(y)
        y = keras.layers.Conv1D(filters=embed_dim, kernel_size=1, use_bias=False, name=f"ds_pw_d{d}")(y)
        y = keras.layers.BatchNormalization(name=f"ds_pw_bn_d{d}")(y)
        x = keras.layers.Add(name=f"ds_add_d{d}")([residual, y])
        x = keras.layers.ReLU(name=f"ds_out_relu_d{d}")(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"ds_drop_d{d}")(x)
        layer_idx += 1

    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)
    return keras.Model(tokens_in, logits, name=name)


def build_gru_prior(
    *,
    vocab_size: int,
    context_length: int,
    embed_dim: int = 48,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    name: str = "rvq_gru_prior",
) -> keras.Model:
    """Small causal GRU prior, naturally streaming with O(hidden_dim) state."""
    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="token_embedding")(tokens_in)
    for i in range(num_layers):
        x = keras.layers.GRU(
            hidden_dim, return_sequences=True, dropout=dropout, recurrent_dropout=0.0, unroll=False, name=f"gru_{i}"
        )(x)
    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)
    return keras.Model(tokens_in, logits, name=name)


def build_wavenet_prior(
    *,
    vocab_size: int,
    context_length: int,
    embed_dim: int = 32,
    num_layers: int = 6,
    kernel_size: int = 5,
    dropout: float = 0.0,
    name: str = "rvq_wavenet_prior",
) -> keras.Model:
    """Gated-residual causal dilated Conv1D prior (WaveNet-style block).

    Each block is::

        h = causal_conv(x, 2*embed_dim)         # gated conv
        a, b = split(h)
        y = tanh(a) * sigmoid(b)
        out = 1x1_conv(y, embed_dim)
        x = x + out                               # residual
        skip += 1x1_conv(y, embed_dim)            # skip-out

    Output head sums skips, applies ReLU, then 1x1 -> 1x1 -> Dense(vocab).
    All ops are INT8-portable. Empirically the strongest architecture in
    this repo's entropy-prior sweeps (default ``num_layers=6``).
    """
    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="token_embedding")(tokens_in)
    skip_total = None
    for i in range(num_layers):
        d = 2**i
        h = keras.layers.Conv1D(
            filters=2 * embed_dim, kernel_size=kernel_size, padding="causal", dilation_rate=d, name=f"gated_conv_d{d}"
        )(x)
        a, b = SplitHalf(embed_dim, name=f"split_d{d}")(h)
        a = keras.layers.Activation("tanh", name=f"tanh_d{d}")(a)
        b = keras.layers.Activation("sigmoid", name=f"sig_d{d}")(b)
        y = keras.layers.Multiply(name=f"gate_d{d}")([a, b])
        res = keras.layers.Conv1D(filters=embed_dim, kernel_size=1, name=f"res_d{d}")(y)
        skip = keras.layers.Conv1D(filters=embed_dim, kernel_size=1, name=f"skip_d{d}")(y)
        x = keras.layers.Add(name=f"res_add_d{d}")([x, res])
        skip_total = skip if skip_total is None else keras.layers.Add(name=f"skip_add_d{d}")([skip_total, skip])
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"drop_d{d}")(x)
    h = keras.layers.ReLU(name="head_relu_1")(skip_total)
    h = keras.layers.Conv1D(filters=embed_dim, kernel_size=1, activation="relu", name="head_1x1_1")(h)
    h = keras.layers.Conv1D(filters=embed_dim, kernel_size=1, activation="relu", name="head_1x1_2")(h)
    logits = keras.layers.Dense(vocab_size, name="lm_head")(h)
    return keras.Model(tokens_in, logits, name=name)


def build_cnngru_prior(
    *,
    vocab_size: int,
    context_length: int,
    embed_dim: int = 32,
    cnn_layers: int = 3,
    kernel_size: int = 5,
    gru_hidden: int = 48,
    gru_layers: int = 1,
    dropout: float = 0.0,
    name: str = "rvq_cnngru_prior",
) -> keras.Model:
    """Hybrid causal CNN frontend + GRU head.

    Cheap dilated-causal-Conv1D stem extracts local features, then a
    small GRU integrates unbounded history. NOTE: GRU-family priors
    require training windows to cover all ``token_position % num_levels``
    phases (use an odd ``stride_tokens`` when ``num_levels`` is even) or
    training will catastrophically fail on out-of-phase windows at eval
    time — see repo entropy-prior sweep notes.
    """
    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="token_embedding")(tokens_in)
    for i in range(cnn_layers):
        x = keras.layers.Conv1D(
            filters=embed_dim,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=2**i,
            activation="relu",
            name=f"causal_conv_d{2**i}",
        )(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"cnn_drop_d{2**i}")(x)
    for i in range(gru_layers):
        x = keras.layers.GRU(
            gru_hidden, return_sequences=True, dropout=dropout, recurrent_dropout=0.0, name=f"gru_{i}"
        )(x)
    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)
    return keras.Model(tokens_in, logits, name=name)


def build_wavenet_bit_prior(
    context_length: int,
    *,
    ctx_vocab: int = 6,
    bp_vocab: int = 16,
    prevbit_vocab: int = 3,
    embed_dim: int = 16,
    num_layers: int = 4,
    kernel_size: int = 3,
    name: str = "spiht_wavenet_bit_prior",
) -> keras.Model:
    """WaveNet-style causal prior for SPIHT's per-bit entropy stream.

    Unlike ``build_wavenet_prior`` (single token vocabulary), each position
    here has THREE side-channel inputs, all causally available before the
    bit at that position is coded/decoded:

        - ``ctx``: SPIHT symbol class (``CTX_*`` in
          :mod:`compressionkit.dsp.spiht` — LIP/LIS-A/LIS-B/child
          significance, sign, refinement). Structurally known in advance
          (driven by SPIHT's deterministic tree-traversal control flow, not
          the coefficient values), identical on encode and decode.
        - ``bitplane``: current SPIHT bit-plane index, clipped to
          ``[0, bp_vocab)``. Also structural.
        - ``prev_bit``: the bit immediately before this position (or a
          START-of-frame sentinel at position 0 — ``prevbit_vocab - 1``).
          The causal history.

    Embeds and sums all three, then runs a gated-residual dilated causal
    Conv1D stack (same block design as ``build_wavenet_prior``), predicting
    a single ``P(bit=1)`` logit per position (``BinaryCrossentropy`` target,
    not a softmax over a token vocabulary).

    See :mod:`compressionkit.generative.spiht_bit_prior` for:
      - a pure-numpy re-implementation of this forward pass (no TF/Keras
        dependency at inference time, verified bit-exact against this model),
      - ``SpihtBitPredictor``, the incremental per-symbol interface used
        during real SPIHT encode/decode,
      - ``NeuralAcSink``/``NeuralAcSource``, which plug into
        ``spiht_encode``/``spiht_decode``'s generic ``sink``/``source``
        injection point (that module has zero knowledge of this model).
    """
    ctx_in = keras.Input(shape=(context_length,), dtype="int32", name="ctx")
    bp_in = keras.Input(shape=(context_length,), dtype="int32", name="bitplane")
    prev_in = keras.Input(shape=(context_length,), dtype="int32", name="prev_bit")

    ctx_emb = keras.layers.Embedding(ctx_vocab, embed_dim, name="ctx_embed")(ctx_in)
    bp_emb = keras.layers.Embedding(bp_vocab, embed_dim, name="bp_embed")(bp_in)
    prev_emb = keras.layers.Embedding(prevbit_vocab, embed_dim, name="prevbit_embed")(prev_in)
    x = keras.layers.Add(name="embed_sum")([ctx_emb, bp_emb, prev_emb])

    for i in range(num_layers):
        d = 2**i
        h = keras.layers.Conv1D(
            filters=2 * embed_dim, kernel_size=kernel_size, padding="causal", dilation_rate=d, name=f"gated_conv_d{d}"
        )(x)
        a, b = SplitHalf(embed_dim, name=f"split_d{d}")(h)
        gated = keras.layers.Multiply(name=f"gate_d{d}")(
            [keras.layers.Activation("tanh")(a), keras.layers.Activation("sigmoid")(b)]
        )
        out = keras.layers.Conv1D(embed_dim, kernel_size=1, name=f"resid_proj_d{d}")(gated)
        x = keras.layers.Add(name=f"resid_add_d{d}")([x, out])

    logits = keras.layers.Dense(1, name="bit_logit")(x)
    logits = keras.layers.Reshape((context_length,), name="squeeze")(logits)
    model = keras.Model([ctx_in, bp_in, prev_in], logits, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    return model
