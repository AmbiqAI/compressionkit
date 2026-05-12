"""Measure the entropy of an RVQ token stream → effective CR uplift.

For a frozen compression model (encoder + RVQ), train a small causal
prior :math:`p(c_t \\mid c_{<t})` on the discrete latent tokens and
evaluate cross-entropy on a held-out validation set. The cross-entropy
in bits/token is a tight upper bound on the entropy of the latent
stream and equals the bitrate achievable with arithmetic coding under
that prior.

Three prior types are supported (``--prior-type``):

* ``unigram`` — static 0-order Huffman / range-coding baseline. No
  context, no compute beyond a 256-entry probability table.  Sets the
  *floor* for what entropy coding can do without modelling structure.
* ``cnn`` — small dilated causal Conv1D prior (~30 k params, INT8 ≈
  30 KB Flash).  Realistic edge-deployment target.
* ``transformer`` — the original causal transformer prior (~480 k
  params).  Upper-bound reference of what a bigger model could buy.

Reported metrics include both **mean** bits/token (average bitrate /
average effective CR) and the **per-frame distribution** (min / median /
p95 / p99 / max), which characterises the dynamic-range behaviour of
arithmetic / range coding.

Example::

    uv run python scripts/measure_rvq_entropy.py \\
        --run-dir results/ecg_rvq_256hz_32x_golden \\
        --prior-type cnn --tag cnn_v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.configs.ppg_h5_rvq import PpgH5RvqConfig
from compressionkit.configs.ppg_rvq import PpgRvqConfig
from compressionkit.datasets.ecg import load_ecg_file_splits, load_ecg_signal
from compressionkit.datasets.ppg import load_ppg_file_splits, load_ppg_signal
from compressionkit.datasets.ppg_h5 import PpgH5Source, SplitConfig, WindowSpec, iter_windows
from compressionkit.generative import build_prior, extract_rvq_tokens
from compressionkit.preprocessing.sanitize import SanitizeConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_modality(config_json: str) -> str:
    cfg_dict = json.loads(config_json)
    data = cfg_dict.get("data", {})
    if "sources" in data and "target_fs" in data:
        return "ppg-h5"
    if "target_label" in data and "sampling_rate" in data:
        return "ppg"
    return "ecg"


def _load_compressor(
    run_dir: Path,
    *,
    modality: str = "auto",
) -> tuple[keras.Model, EcgRvqConfig | PpgRvqConfig | PpgH5RvqConfig, str]:
    config_json = (run_dir / "config.json").read_text()
    detected = _detect_modality(config_json) if modality == "auto" else modality

    if detected == "ppg":
        from compressionkit.trainers.ppg_rvq import build_model

        cfg = PpgRvqConfig.model_validate_json(config_json)
        dummy = np.zeros((1, 1, cfg.data.frame_size, 1), dtype=np.float32)
    elif detected == "ppg-h5":
        from compressionkit.recipes.train_ppg_h5_rvq import build_model

        cfg = PpgH5RvqConfig.model_validate_json(config_json)
        dummy = np.zeros((1, 1, cfg.data.window_samples, 1), dtype=np.float32)
    else:
        from compressionkit.trainers.ecg_rvq import build_model

        cfg = EcgRvqConfig.model_validate_json(config_json)
        dummy = np.zeros(
            (1, 1, cfg.data.frame_size, max(1, cfg.data.num_leads or 1)),
            dtype=np.float32,
        )

    with tf.device("/CPU:0"):
        model = build_model(cfg)
        model(dummy, training=False)
    for name in ("best_model.weights.h5", "model.weights.h5"):
        p = run_dir / name
        if p.exists():
            model.load_weights(p)
            break
    else:
        raise FileNotFoundError(f"No weights in {run_dir}")
    return model, cfg, detected


def _extract_split_tokens(
    files: list[Path],
    compressor: keras.Model,
    *,
    frame_size: int,
    num_leads: int,
    epsilon: float,
    lead_index: int,
) -> np.ndarray:
    signals = (load_ecg_signal(p, lead_index=lead_index) for p in files)
    return extract_rvq_tokens(
        compressor,
        signals,
        frame_size=frame_size,
        num_leads=num_leads,
        epsilon=epsilon,
        batch_size=64,
    )


def _tokens_from_model_inputs(
    compressor: keras.Model,
    frames: np.ndarray,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode already-normalized ``(N, frame_size)`` frames into RVQ tokens."""
    if frames.size == 0:
        return np.zeros((0, 0, 0), dtype=np.int16)
    batched_in = np.asarray(frames, dtype=np.float32)[:, np.newaxis, :, np.newaxis]
    z_chunks: list[np.ndarray] = []
    for start in range(0, batched_in.shape[0], batch_size):
        chunk = batched_in[start : start + batch_size]
        with tf.device("/CPU:0"):
            z = compressor.encoder(chunk, training=False)
        z_chunks.append(np.asarray(z))
    z_all = np.concatenate(z_chunks, axis=0)
    latent_shape = z_all.shape
    with tf.device("/CPU:0"):
        indices_list = compressor.vq.encode(z_all)
    tokens_per_frame = int(np.prod(latent_shape[:-1])) // latent_shape[0]
    out = np.zeros((latent_shape[0], tokens_per_frame, len(indices_list)), dtype=np.int16)
    for level, idx in enumerate(indices_list):
        idx_np = np.asarray(idx).reshape(latent_shape[0], tokens_per_frame).astype(np.int16)
        out[..., level] = idx_np
    return out


def _normalised_ppg_frames(signal: np.ndarray, frame_size: int, epsilon: float) -> np.ndarray:
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    n_frames = sig.size // frame_size
    if n_frames == 0:
        return np.zeros((0, frame_size), dtype=np.float32)
    usable = sig[: n_frames * frame_size].reshape(n_frames, frame_size)
    means = usable.mean(axis=1, keepdims=True)
    stds = usable.std(axis=1, keepdims=True) + float(epsilon)
    return ((usable - means) / stds).astype(np.float32)


def _extract_unified_cache_tokens(
    compressor: keras.Model,
    cfg: PpgRvqConfig,
    *,
    split: str,
    max_frames: int | None,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract RVQ tokens from the unified per-source TFRecord cache."""
    from compressionkit.datasets.ppg_cache import SourceWeight, load_cached_raw_windows

    ucfg = cfg.data.unified_cache
    source_weights = [SourceWeight(slug=s.slug, weight=s.weight) for s in ucfg.sources]
    cache_root = Path(ucfg.cache_root)
    if not cache_root.is_absolute():
        cache_root = cache_root.resolve()

    raw_windows = load_cached_raw_windows(
        source_weights,
        cache_root=cache_root,
        frame_size=cfg.data.frame_size,
        split=split,
        max_windows=max_frames,
        seed=cfg.data.shuffle_seed,
    )
    # Normalize per-frame (same as training pipeline)
    means = raw_windows.mean(axis=1, keepdims=True)
    stds = raw_windows.std(axis=1, keepdims=True) + cfg.data.epsilon
    frames = ((raw_windows - means) / stds).astype(np.float32)
    return _tokens_from_model_inputs(compressor, frames, batch_size=batch_size)


def _extract_ppg_split_tokens(
    files: list[Path],
    compressor: keras.Model,
    cfg: PpgRvqConfig,
    *,
    max_frames: int | None,
    batch_size: int = 64,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    total = 0
    for path in files:
        try:
            signal = load_ppg_signal(
                path,
                target_rate=cfg.data.sampling_rate,
                offset_samples=cfg.data.offset_samples,
                num_samples=cfg.data.segment_samples,
                target_label=cfg.data.target_label,
            )
        except Exception:
            continue
        file_frames = _normalised_ppg_frames(signal, cfg.data.frame_size, cfg.data.epsilon)
        if file_frames.size == 0:
            continue
        if max_frames is not None:
            remaining = max(0, max_frames - total)
            if remaining == 0:
                break
            file_frames = file_frames[:remaining]
        frames.append(file_frames)
        total += int(file_frames.shape[0])
    if not frames:
        return np.zeros((0, 0, 0), dtype=np.int16)
    return _tokens_from_model_inputs(compressor, np.concatenate(frames, axis=0), batch_size=batch_size)


def _h5_sanitize_config(cfg: PpgH5RvqConfig) -> SanitizeConfig | None:
    if not cfg.data.sanitize.enabled:
        return None
    return SanitizeConfig(
        min_std=cfg.data.sanitize.min_std,
        max_saturation_frac=cfg.data.sanitize.max_saturation_frac,
        max_abs_z=cfg.data.sanitize.max_abs_z,
        max_outlier_frac=cfg.data.sanitize.max_outlier_frac,
    )


def _h5_sources(cfg: PpgH5RvqConfig) -> list[PpgH5Source]:
    root = Path(cfg.data.root)
    return [
        PpgH5Source(
            slug=src.slug,
            root=root,
            glob=src.glob,
            butppg_quality_only=src.butppg_quality_only,
        )
        for src in cfg.data.sources
    ]


def _extract_ppg_h5_split_tokens(
    compressor: keras.Model,
    cfg: PpgH5RvqConfig,
    *,
    split: str,
    max_frames: int | None,
    batch_size: int = 64,
) -> np.ndarray:
    spec = WindowSpec(
        target_fs=cfg.data.target_fs,
        window_seconds=cfg.data.window_seconds,
        hop_seconds=cfg.data.hop_seconds,
        sanitize=_h5_sanitize_config(cfg),
        normalize=cfg.data.normalize,
    )
    split_cfg = SplitConfig(
        train_frac=cfg.data.split.train_frac,
        val_frac=cfg.data.split.val_frac,
        seed=cfg.data.split.seed,
        split=split,
    )
    frames: list[np.ndarray] = []
    for sample in iter_windows(_h5_sources(cfg), spec, split_cfg=split_cfg):
        frames.append(np.asarray(sample["data"], dtype=np.float32).reshape(spec.window_samples))
        if max_frames is not None and len(frames) >= max_frames:
            break
    if not frames:
        return np.zeros((0, 0, 0), dtype=np.int16)
    return _tokens_from_model_inputs(compressor, np.stack(frames, axis=0), batch_size=batch_size)


def _get_frame_size(cfg: EcgRvqConfig | PpgRvqConfig | PpgH5RvqConfig, modality: str) -> int:
    if modality == "ppg-h5":
        return int(cfg.data.window_samples)
    return int(cfg.data.frame_size)


def _get_input_bit_depth(cfg: EcgRvqConfig | PpgRvqConfig | PpgH5RvqConfig) -> int:
    return int(cfg.evaluation.input_bit_depth)


def _get_sample_rate(cfg: EcgRvqConfig | PpgRvqConfig | PpgH5RvqConfig, modality: str) -> float:
    if modality == "ppg-h5":
        return float(cfg.data.target_fs)
    return float(cfg.data.sampling_rate)


def _interleave_levels(tokens: np.ndarray) -> np.ndarray:
    """``(F, T, L)`` → flat ``(F * T * L,)`` with level-major intra-step order."""
    f, t, lvl = tokens.shape
    return tokens.reshape(f * t * lvl).astype(np.int32)


def _build_windows(
    stream: np.ndarray,
    context_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = stream.size
    if n < context_length + 1:
        raise ValueError(f"Token stream too short ({n}) for context {context_length}")
    starts = np.arange(0, n - context_length - 1, max(1, stride))
    xs = np.stack([stream[s : s + context_length] for s in starts]).astype(np.int32)
    ys = np.stack([stream[s + 1 : s + 1 + context_length] for s in starts]).astype(np.int32)
    return xs, ys


def _baseline_bits_per_token(vocab_size: int, tokens: np.ndarray) -> float:
    counts = np.bincount(tokens.astype(np.int64), minlength=vocab_size).astype(np.float64)
    probs = counts / counts.sum()
    nz = probs > 0
    return float(-(probs[nz] * np.log2(probs[nz])).sum())


# ---------------------------------------------------------------------------
# Prior builders
# ---------------------------------------------------------------------------


def _build_cnn_prior(
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

    Inputs ``(B, context_length)`` int tokens, outputs
    ``(B, context_length, vocab_size)`` next-token logits — same
    interface as the transformer prior so the rest of the script is
    backbone-agnostic.

    Each conv layer has dilation ``2**i`` for layer ``i``, giving a
    receptive field of ``1 + (k-1) * (2**num_layers - 1)`` tokens.
    """
    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="token_embedding",
    )(tokens_in)
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


def _build_dscnn_prior(
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

    Architectural variants vs ``_build_cnn_prior``:
      * Each dilated Conv1D becomes ``DepthwiseConv1D + pointwise Conv1D``
        — same receptive field, ~80% fewer params per layer.
      * Optional factored embedding: when ``proj_dim < embed_dim``, the
        token table is ``Embedding(vocab, proj_dim) -> Dense(embed_dim)``,
        which is much cheaper than a full ``vocab x embed_dim`` table.
      * ``norm`` selects the head normalisation: ``"layer"`` (default,
        same as ``_build_cnn_prior``) or ``"none"`` (skip entirely).

    All ops are INT8-portable in TFLite/LiteRT.

    Args:
        vocab_size: Codebook size K.
        context_length: Sequence length the prior is *trained* on.
        embed_dim: Channel width through the conv stack.
        proj_dim: Optional embedding bottleneck dim. Defaults to ``embed_dim``
            (i.e. no factoring).
        num_layers: Number of dilated DSConv blocks.
        kernel_size: Depthwise kernel size.
        dropout: Optional spatial dropout after each block.
        norm: ``"layer"`` (default) or ``"none"``.
    """
    if proj_dim is None:
        proj_dim = embed_dim
    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=proj_dim,
        name="token_embedding",
    )(tokens_in)
    if proj_dim != embed_dim:
        x = keras.layers.Dense(embed_dim, name="embed_proj")(x)
    for i in range(num_layers):
        d = 2**i
        # Manual causal left-pad: DepthwiseConv1D doesn't support
        # padding="causal", so we ZeroPad1D by (k-1)*dilation on the
        # left then run a "valid" depthwise conv.
        x = keras.layers.ZeroPadding1D(
            padding=((kernel_size - 1) * d, 0),
            name=f"causal_pad_d{d}",
        )(x)
        x = keras.layers.DepthwiseConv1D(
            kernel_size=kernel_size,
            padding="valid",
            dilation_rate=d,
            name=f"dw_d{d}",
        )(x)
        x = keras.layers.Conv1D(
            filters=embed_dim,
            kernel_size=1,
            activation="relu",
            name=f"pw_d{d}",
        )(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"drop_d{d}")(x)
    if norm == "layer":
        x = keras.layers.LayerNormalization(epsilon=1e-5, name="final_ln")(x)
    elif norm != "none":
        raise ValueError(f"Unknown norm={norm!r}; expected 'layer' or 'none'.")
    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)
    return keras.Model(tokens_in, logits, name=name)


def _build_hybrid_prior(
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

    Designed for INT8 LiteRT deployment:
      * The first ``stem_layers`` blocks are *full* causal Conv1D so all
        channels can mix freely up front (where token embeddings still
        carry codeword identity).
      * The next ``ds_layers`` blocks are residual DepthwiseConv1D +
        pointwise Conv1D — cheap params, but the residual restores the
        gradient/feature path that pure depthwise stacks lose.
      * Every conv is followed by ``BatchNormalization`` (no LayerNorm).
        BN is fold-able into the preceding conv at INT8 export time, so
        runtime cost is zero.

    Dilation grows ``2**i`` across the *combined* stack to keep the
    receptive field comparable to the dense / dscnn priors at the same
    total depth.

    Args:
        vocab_size: Codebook size K.
        context_length: Sequence length the prior is *trained* on.
        embed_dim: Channel width through the stack.
        proj_dim: Optional embedding bottleneck dim. ``None`` = no factoring.
        stem_layers: Number of dense Conv1D blocks at the input.
        ds_layers: Number of residual DSConv blocks after the stem.
        kernel_size: Convolution kernel size.
        dropout: Optional dropout after each block.
    """
    if proj_dim is None:
        proj_dim = embed_dim
    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=proj_dim,
        name="token_embedding",
    )(tokens_in)
    if proj_dim != embed_dim:
        x = keras.layers.Dense(embed_dim, name="embed_proj")(x)

    layer_idx = 0
    # ---- Dense Conv1D stem (causal, BN, ReLU) ------------------------------
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

    # ---- Residual DSConv body (causal depthwise + pointwise, BN, ReLU) ----
    for _ in range(ds_layers):
        d = 2**layer_idx
        residual = x
        # Manual causal pad for depthwise (DepthwiseConv1D rejects "causal").
        y = keras.layers.ZeroPadding1D(
            padding=((kernel_size - 1) * d, 0),
            name=f"ds_pad_d{d}",
        )(x)
        y = keras.layers.DepthwiseConv1D(
            kernel_size=kernel_size,
            padding="valid",
            dilation_rate=d,
            use_bias=False,
            name=f"ds_dw_d{d}",
        )(y)
        y = keras.layers.BatchNormalization(name=f"ds_dw_bn_d{d}")(y)
        y = keras.layers.ReLU(name=f"ds_dw_relu_d{d}")(y)
        y = keras.layers.Conv1D(
            filters=embed_dim,
            kernel_size=1,
            use_bias=False,
            name=f"ds_pw_d{d}",
        )(y)
        y = keras.layers.BatchNormalization(name=f"ds_pw_bn_d{d}")(y)
        x = keras.layers.Add(name=f"ds_add_d{d}")([residual, y])
        x = keras.layers.ReLU(name=f"ds_out_relu_d{d}")(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"ds_drop_d{d}")(x)
        layer_idx += 1

    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)
    return keras.Model(tokens_in, logits, name=name)


def _build_gru_prior(
    *,
    vocab_size: int,
    context_length: int,
    embed_dim: int = 48,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    name: str = "rvq_gru_prior",
) -> keras.Model:
    """Small causal GRU prior, naturally streaming with O(hidden_dim) state.

    Inputs ``(B, context_length)`` int tokens, outputs
    ``(B, context_length, vocab_size)`` next-token logits — same
    interface as the conv priors.

    LiteRT supports unrolled GRU; for streaming deployment the same cell
    can be unrolled to step=1 with the hidden state exported as an
    explicit input/output.
    """
    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="token_embedding",
    )(tokens_in)
    for i in range(num_layers):
        x = keras.layers.GRU(
            hidden_dim,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.0,
            unroll=False,
            name=f"gru_{i}",
        )(x)
    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)
    return keras.Model(tokens_in, logits, name=name)


def _build_wavenet_prior(
    *,
    vocab_size: int,
    context_length: int,
    embed_dim: int = 32,
    num_layers: int = 4,
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

    Output head sums skips, applies ReLU, then 1x1 -> 1x1 -> Dense(vocab),
    matching the canonical WaveNet topology. All ops are INT8-portable.
    """
    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="token_embedding",
    )(tokens_in)
    skip_total = None
    for i in range(num_layers):
        d = 2**i
        h = keras.layers.Conv1D(
            filters=2 * embed_dim,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=d,
            name=f"gated_conv_d{d}",
        )(x)
        a, b = keras.layers.Lambda(
            lambda t: (t[..., :embed_dim], t[..., embed_dim:]),
            output_shape=((context_length, embed_dim), (context_length, embed_dim)),
            name=f"split_d{d}",
        )(h)
        a = keras.layers.Activation("tanh", name=f"tanh_d{d}")(a)
        b = keras.layers.Activation("sigmoid", name=f"sig_d{d}")(b)
        y = keras.layers.Multiply(name=f"gate_d{d}")([a, b])
        res = keras.layers.Conv1D(
            filters=embed_dim,
            kernel_size=1,
            name=f"res_d{d}",
        )(y)
        skip = keras.layers.Conv1D(
            filters=embed_dim,
            kernel_size=1,
            name=f"skip_d{d}",
        )(y)
        x = keras.layers.Add(name=f"res_add_d{d}")([x, res])
        skip_total = (
            skip
            if skip_total is None
            else keras.layers.Add(
                name=f"skip_add_d{d}",
            )([skip_total, skip])
        )
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"drop_d{d}")(x)
    h = keras.layers.ReLU(name="head_relu_1")(skip_total)
    h = keras.layers.Conv1D(
        filters=embed_dim,
        kernel_size=1,
        activation="relu",
        name="head_1x1_1",
    )(h)
    h = keras.layers.Conv1D(
        filters=embed_dim,
        kernel_size=1,
        activation="relu",
        name="head_1x1_2",
    )(h)
    logits = keras.layers.Dense(vocab_size, name="lm_head")(h)
    return keras.Model(tokens_in, logits, name=name)


def _build_cnngru_prior(
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
    small GRU integrates unbounded history — combines fixed receptive
    field (cheap) with infinite-state recurrence (expressive). Streams
    naturally on-device.
    """
    tokens_in = keras.Input(shape=(context_length,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="token_embedding",
    )(tokens_in)
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
            gru_hidden,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.0,
            name=f"gru_{i}",
        )(x)
    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)
    return keras.Model(tokens_in, logits, name=name)


class _UnigramLayer(keras.layers.Layer):
    """Constant-output layer emitting train-set log-probabilities."""

    def __init__(self, log_probs: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self._log_probs_np = np.asarray(log_probs, dtype=np.float32)
        self._log_probs = keras.ops.convert_to_tensor(self._log_probs_np)

    def call(self, x):
        b = keras.ops.shape(x)[0]
        t = keras.ops.shape(x)[1]
        lp = keras.ops.reshape(self._log_probs, (1, 1, -1))
        return keras.ops.broadcast_to(lp, (b, t, self._log_probs_np.shape[0]))


def _build_unigram_prior(
    train_stream: np.ndarray,
    vocab_size: int,
) -> tuple[keras.Model, np.ndarray]:
    """Static unigram prior fitted on train counts (Laplace smoothed)."""
    counts = (
        np.bincount(
            train_stream.astype(np.int64),
            minlength=vocab_size,
        ).astype(np.float64)
        + 1.0
    )
    probs = counts / counts.sum()
    log_probs = np.log(probs).astype(np.float32)

    inp = keras.Input(shape=(None,), dtype="int32", name="tokens")
    out = _UnigramLayer(log_probs, name="unigram")(inp)
    return keras.Model(inp, out, name="unigram_prior"), log_probs


# ---------------------------------------------------------------------------
# Per-token / per-frame bitrate analysis
# ---------------------------------------------------------------------------


def _per_token_nll_bits(
    prior: keras.Model,
    stream: np.ndarray,
    *,
    context_length: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Predict last-position logits for every sliding window over *stream*.

    For each position ``t >= context_length`` the model sees window
    ``stream[t-context_length:t]`` and predicts ``stream[t]``.  Returns a
    bits-per-token array of shape ``(N - context_length,)``.
    """
    n = stream.size
    if n <= context_length:
        return np.zeros((0,), dtype=np.float64)
    starts = np.arange(n - context_length, dtype=np.int64)
    targets = stream[context_length:].astype(np.int32)

    bits = np.zeros(starts.size, dtype=np.float64)
    log2 = math.log(2.0)
    for i in range(0, starts.size, batch_size):
        sl = slice(i, i + batch_size)
        bs = starts[sl]
        x = np.stack([stream[s : s + context_length] for s in bs]).astype(np.int32)
        logits = prior(x, training=False)
        last = keras.ops.convert_to_numpy(logits[:, -1, :]).astype(np.float64)
        last -= last.max(axis=-1, keepdims=True)
        log_norm = np.log(np.exp(last).sum(axis=-1, keepdims=True))
        log_probs = last - log_norm
        idx = np.arange(bs.size)
        bits[sl] = -log_probs[idx, targets[sl]] / log2
    return bits


def _per_frame_summary(
    bits_per_token: np.ndarray,
    *,
    tokens_per_frame: int,
    skipped_tokens: int,
) -> dict:
    """Aggregate per-token bits into per-frame totals (full frames only)."""
    if bits_per_token.size == 0:
        return {}
    base_pos = skipped_tokens
    first_frame = (base_pos + tokens_per_frame - 1) // tokens_per_frame
    first_token_offset = first_frame * tokens_per_frame - base_pos
    trimmed = bits_per_token[first_token_offset:]
    n_full = (trimmed.size // tokens_per_frame) * tokens_per_frame
    if n_full == 0:
        return {}
    frames = trimmed[:n_full].reshape(-1, tokens_per_frame)
    bits_per_frame = frames.sum(axis=1)
    return {
        "num_frames": int(bits_per_frame.size),
        "mean": float(bits_per_frame.mean()),
        "std": float(bits_per_frame.std()),
        "min": float(bits_per_frame.min()),
        "p05": float(np.percentile(bits_per_frame, 5)),
        "p25": float(np.percentile(bits_per_frame, 25)),
        "median": float(np.median(bits_per_frame)),
        "p75": float(np.percentile(bits_per_frame, 75)),
        "p95": float(np.percentile(bits_per_frame, 95)),
        "p99": float(np.percentile(bits_per_frame, 99)),
        "max": float(bits_per_frame.max()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--modality",
        choices=["auto", "ecg", "ppg", "ppg-h5"],
        default="auto",
        help="Codec/data path to use. Default detects from config.json.",
    )
    parser.add_argument(
        "--num-train-files",
        type=int,
        default=400,
        help="Number of train files to encode for ECG/PPG EDF runs (-1 = all).",
    )
    parser.add_argument(
        "--num-val-files", type=int, default=80, help="Number of val files to encode for ECG/PPG EDF runs (-1 = all)."
    )
    parser.add_argument(
        "--max-train-frames", type=int, default=None, help="Optional cap on encoded training frames/windows."
    )
    parser.add_argument(
        "--max-val-frames", type=int, default=None, help="Optional cap on encoded validation frames/windows."
    )
    parser.add_argument(
        "--token-cache-dir",
        type=Path,
        default=None,
        help="Where to cache extracted token streams. Default: <run_dir>/entropy_prior/_token_cache.",
    )
    parser.add_argument("--no-token-cache", action="store_true", help="Disable read/write of cached token streams.")
    parser.add_argument(
        "--prior-type",
        choices=["unigram", "cnn", "dscnn", "hybrid", "gru", "wavenet", "cnngru", "transformer"],
        default="transformer",
    )
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--stride-tokens", type=int, default=8)
    # Transformer hyperparams
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    # CNN hyperparams
    parser.add_argument("--cnn-embed-dim", type=int, default=48)
    parser.add_argument("--cnn-num-layers", type=int, default=4)
    parser.add_argument("--cnn-kernel", type=int, default=5)
    # DSCNN hyperparams (lean depthwise-separable variant)
    parser.add_argument("--dscnn-embed-dim", type=int, default=64, help="Channel width through the DSConv stack.")
    parser.add_argument(
        "--dscnn-proj-dim",
        type=int,
        default=None,
        help="Optional embedding bottleneck dim (factored embed). None = same as --dscnn-embed-dim.",
    )
    parser.add_argument("--dscnn-num-layers", type=int, default=5)
    parser.add_argument("--dscnn-kernel", type=int, default=5)
    parser.add_argument("--dscnn-norm", choices=["layer", "none"], default="layer")
    # Hybrid hyperparams (dense stem + residual DSConv body, BatchNorm)
    parser.add_argument("--hybrid-embed-dim", type=int, default=64)
    parser.add_argument(
        "--hybrid-proj-dim",
        type=int,
        default=None,
        help="Optional embedding bottleneck dim; None = same as --hybrid-embed-dim.",
    )
    parser.add_argument("--hybrid-stem-layers", type=int, default=2)
    parser.add_argument("--hybrid-ds-layers", type=int, default=4)
    parser.add_argument("--hybrid-kernel", type=int, default=5)
    # GRU hyperparams
    parser.add_argument("--gru-embed-dim", type=int, default=48)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--gru-num-layers", type=int, default=1)
    # WaveNet (gated-residual) hyperparams
    parser.add_argument("--wavenet-embed-dim", type=int, default=32)
    parser.add_argument("--wavenet-num-layers", type=int, default=4)
    parser.add_argument("--wavenet-kernel", type=int, default=5)
    # CNN+GRU hybrid hyperparams
    parser.add_argument("--cnngru-embed-dim", type=int, default=32)
    parser.add_argument("--cnngru-cnn-layers", type=int, default=3)
    parser.add_argument("--cnngru-kernel", type=int, default=5)
    parser.add_argument("--cnngru-gru-hidden", type=int, default=48)
    parser.add_argument("--cnngru-gru-layers", type=int, default=1)
    # Training
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--per-frame-batch", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir.resolve()
    out_root = args.output_dir or (run_dir / "entropy_prior")
    out_dir = out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading compressor from {run_dir} ...", file=sys.stderr)
    compressor, cfg, modality = _load_compressor(run_dir, modality=args.modality)
    data = cfg.data
    mcfg = cfg.model
    vocab_size = int(mcfg.latent_width)
    frame_size = _get_frame_size(cfg, modality)
    num_leads = int(getattr(data, "num_leads", 1) or 1)
    lead_index = int(getattr(data, "lead_index", 1) or 1)
    tokens_per_frame_per_level = frame_size // (2**mcfg.num_stages)
    num_levels = int(mcfg.num_levels)
    tokens_per_frame = tokens_per_frame_per_level * num_levels
    bits_per_frame_uniform = math.log2(vocab_size) * tokens_per_frame
    print(
        f"      frame_size={frame_size} tokens/frame={tokens_per_frame} "
        f"levels={num_levels} vocab={vocab_size} modality={modality}",
        file=sys.stderr,
    )

    # --- Token extraction ------------------------------------------------
    print("[2/5] Splitting files & extracting tokens ...", file=sys.stderr)
    train_files: list[Path] = []
    val_files: list[Path] = []
    use_unified_cache = modality == "ppg" and hasattr(data, "unified_cache") and data.unified_cache.enabled
    if use_unified_cache:
        print("      unified_cache mode: loading from TFRecord cache", file=sys.stderr)
    elif modality == "ppg-h5":
        if args.max_train_frames is None or args.max_val_frames is None:
            print(
                "      h5 mode: consider --max-train-frames/--max-val-frames to bound extraction cost",
                file=sys.stderr,
            )
    elif modality == "ppg":
        train_files, val_files, _ = load_ppg_file_splits(
            Path(data.datasets_dir),
            data.dataset_glob,
            seed=data.shuffle_seed,
        )
    else:
        train_files, val_files, _ = load_ecg_file_splits(
            Path(data.datasets_dir),
            data.dataset_glob,
            seed=data.shuffle_seed,
        )
    if not use_unified_cache and modality != "ppg-h5":
        if args.num_train_files is not None and args.num_train_files >= 0:
            train_files = train_files[: args.num_train_files]
        if args.num_val_files is not None and args.num_val_files >= 0:
            val_files = val_files[: args.num_val_files]
        if not val_files and not use_unified_cache:
            raise RuntimeError("No validation files available for entropy measurement.")

    cache_root: Path | None = None
    if not args.no_token_cache:
        cache_root = args.token_cache_dir or (run_dir / "entropy_prior" / "_token_cache")
        cache_root.mkdir(parents=True, exist_ok=True)

    def _cache_path(split: str, n_files: int) -> Path | None:
        if cache_root is None:
            return None
        frame_cap = args.max_train_frames if split == "train" else args.max_val_frames
        key = f"{modality}_{split}_n{n_files}_max{frame_cap}_fs{frame_size}_l{num_leads}_li{lead_index}.npy"
        return cache_root / key

    def _get_or_extract(split: str, files: list[Path]) -> np.ndarray:
        path = _cache_path(split, len(files))
        if path is not None and path.exists():
            print(f"      [cache] reuse {split} tokens \u2190 {path.name}", file=sys.stderr)
            return np.load(path)
        max_frames = args.max_train_frames if split == "train" else args.max_val_frames
        if use_unified_cache:
            tokens = _extract_unified_cache_tokens(
                compressor,
                cfg,
                split=split,
                max_frames=max_frames,
                batch_size=64,
            )
        elif modality == "ppg-h5":
            tokens = _extract_ppg_h5_split_tokens(
                compressor,
                cfg,
                split=split,
                max_frames=max_frames,
                batch_size=64,
            )
        elif modality == "ppg":
            tokens = _extract_ppg_split_tokens(
                files,
                compressor,
                cfg,
                max_frames=max_frames,
                batch_size=64,
            )
        else:
            tokens = _extract_split_tokens(
                files,
                compressor,
                frame_size=frame_size,
                num_leads=num_leads,
                epsilon=data.epsilon,
                lead_index=lead_index,
            )
        if path is not None:
            np.save(path, tokens)
            print(f"      [cache] saved {split} tokens \u2192 {path.name}", file=sys.stderr)
        return tokens

    train_tok = _get_or_extract("train", train_files)
    val_tok = _get_or_extract("val", val_files)
    if train_tok.size == 0 or val_tok.size == 0:
        raise RuntimeError("Token extraction produced an empty train or validation split.")
    print(f"      train tokens {train_tok.shape}  val tokens {val_tok.shape}", file=sys.stderr)

    train_stream = _interleave_levels(train_tok)
    val_stream = _interleave_levels(val_tok)
    train_unigram = _baseline_bits_per_token(vocab_size, train_stream)
    val_unigram = _baseline_bits_per_token(vocab_size, val_stream)
    print(
        f"      unigram entropy: train={train_unigram:.4f} val={val_unigram:.4f} "
        f"(uniform={math.log2(vocab_size):.4f}) bits/token",
        file=sys.stderr,
    )

    context_length = args.context_frames * tokens_per_frame
    print(f"[3/5] Building prior (type={args.prior_type}, ctx={context_length}) ...", file=sys.stderr)

    history_dict: dict = {}
    if args.prior_type == "unigram":
        prior, log_probs = _build_unigram_prior(train_stream, vocab_size)
        prior_meta = {"type": "unigram", "params": 0, "context_length": 1, "fit": "train_counts_with_laplace_smoothing"}
        # Fully-deterministic eval: bits[t] = -log2 p[c_t] for every t.
        bits_full = (-log_probs[val_stream.astype(np.int64)] / math.log(2.0)).astype(np.float64)
        skipped = 0
    else:
        if args.prior_type == "cnn":
            prior = _build_cnn_prior(
                vocab_size=vocab_size,
                context_length=context_length,
                embed_dim=args.cnn_embed_dim,
                num_layers=args.cnn_num_layers,
                kernel_size=args.cnn_kernel,
                dropout=args.dropout,
            )
            prior_meta = {
                "type": "cnn",
                "embed_dim": args.cnn_embed_dim,
                "num_layers": args.cnn_num_layers,
                "kernel": args.cnn_kernel,
                "receptive_field": 1 + (args.cnn_kernel - 1) * (2**args.cnn_num_layers - 1),
            }
        elif args.prior_type == "dscnn":
            prior = _build_dscnn_prior(
                vocab_size=vocab_size,
                context_length=context_length,
                embed_dim=args.dscnn_embed_dim,
                proj_dim=args.dscnn_proj_dim,
                num_layers=args.dscnn_num_layers,
                kernel_size=args.dscnn_kernel,
                dropout=args.dropout,
                norm=args.dscnn_norm,
            )
            prior_meta = {
                "type": "dscnn",
                "embed_dim": args.dscnn_embed_dim,
                "proj_dim": args.dscnn_proj_dim or args.dscnn_embed_dim,
                "num_layers": args.dscnn_num_layers,
                "kernel": args.dscnn_kernel,
                "norm": args.dscnn_norm,
                "receptive_field": 1 + (args.dscnn_kernel - 1) * (2**args.dscnn_num_layers - 1),
            }
        elif args.prior_type == "hybrid":
            prior = _build_hybrid_prior(
                vocab_size=vocab_size,
                context_length=context_length,
                embed_dim=args.hybrid_embed_dim,
                proj_dim=args.hybrid_proj_dim,
                stem_layers=args.hybrid_stem_layers,
                ds_layers=args.hybrid_ds_layers,
                kernel_size=args.hybrid_kernel,
                dropout=args.dropout,
            )
            total_layers = args.hybrid_stem_layers + args.hybrid_ds_layers
            prior_meta = {
                "type": "hybrid",
                "embed_dim": args.hybrid_embed_dim,
                "proj_dim": args.hybrid_proj_dim or args.hybrid_embed_dim,
                "stem_layers": args.hybrid_stem_layers,
                "ds_layers": args.hybrid_ds_layers,
                "kernel": args.hybrid_kernel,
                "receptive_field": 1 + (args.hybrid_kernel - 1) * (2**total_layers - 1),
            }
        elif args.prior_type == "gru":
            prior = _build_gru_prior(
                vocab_size=vocab_size,
                context_length=context_length,
                embed_dim=args.gru_embed_dim,
                hidden_dim=args.gru_hidden,
                num_layers=args.gru_num_layers,
                dropout=args.dropout,
            )
            prior_meta = {
                "type": "gru",
                "embed_dim": args.gru_embed_dim,
                "hidden": args.gru_hidden,
                "num_layers": args.gru_num_layers,
            }
        elif args.prior_type == "wavenet":
            prior = _build_wavenet_prior(
                vocab_size=vocab_size,
                context_length=context_length,
                embed_dim=args.wavenet_embed_dim,
                num_layers=args.wavenet_num_layers,
                kernel_size=args.wavenet_kernel,
                dropout=args.dropout,
            )
            prior_meta = {
                "type": "wavenet",
                "embed_dim": args.wavenet_embed_dim,
                "num_layers": args.wavenet_num_layers,
                "kernel": args.wavenet_kernel,
                "receptive_field": 1 + (args.wavenet_kernel - 1) * (2**args.wavenet_num_layers - 1),
            }
        elif args.prior_type == "cnngru":
            prior = _build_cnngru_prior(
                vocab_size=vocab_size,
                context_length=context_length,
                embed_dim=args.cnngru_embed_dim,
                cnn_layers=args.cnngru_cnn_layers,
                kernel_size=args.cnngru_kernel,
                gru_hidden=args.cnngru_gru_hidden,
                gru_layers=args.cnngru_gru_layers,
                dropout=args.dropout,
            )
            prior_meta = {
                "type": "cnngru",
                "embed_dim": args.cnngru_embed_dim,
                "cnn_layers": args.cnngru_cnn_layers,
                "kernel": args.cnngru_kernel,
                "gru_hidden": args.cnngru_gru_hidden,
                "gru_layers": args.cnngru_gru_layers,
            }
        else:
            prior = build_prior(
                vocab_size=vocab_size,
                context_length=context_length,
                embed_dim=args.embed_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                ffn_dim=args.ffn_dim,
                dropout=args.dropout,
            )
            prior_meta = {
                "type": "transformer",
                "embed_dim": args.embed_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "ffn_dim": args.ffn_dim,
            }
        prior.summary(print_fn=lambda s: print("      " + s, file=sys.stderr))
        prior.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            ),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="token_acc")],
        )
        x_tr, y_tr = _build_windows(train_stream, context_length, args.stride_tokens)
        x_val, y_val = _build_windows(val_stream, context_length, args.stride_tokens)
        print(f"      train windows={x_tr.shape[0]}  val windows={x_val.shape[0]}", file=sys.stderr)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            restore_best_weights=True,
        )
        print(f"[4/5] Training prior ({args.prior_type}) ...", file=sys.stderr)
        history = prior.fit(
            x_tr,
            y_tr,
            validation_data=(x_val, y_val),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=[early_stop],
            verbose=2,
        )
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        prior_meta["params"] = int(prior.count_params())
        prior_meta["context_length"] = context_length
        bits_full = _per_token_nll_bits(
            prior,
            val_stream,
            context_length=context_length,
            batch_size=args.per_frame_batch,
        )
        skipped = context_length

    print("[5/5] Aggregating per-token / per-frame bitrate stats ...", file=sys.stderr)
    val_bits_per_token = float(bits_full.mean())
    val_bits_per_token_std = float(bits_full.std())
    bits_per_frame_learned = val_bits_per_token * tokens_per_frame
    cr_uplift_vs_uniform = (math.log2(vocab_size)) / val_bits_per_token
    cr_uplift_vs_unigram = val_unigram / val_bits_per_token
    raw_input_bits_per_frame = frame_size * _get_input_bit_depth(cfg)
    sample_rate = _get_sample_rate(cfg, modality)
    raw_bitrate_bps = sample_rate * _get_input_bit_depth(cfg)
    uniform_bitrate_bps = bits_per_frame_uniform * sample_rate / frame_size
    learned_bitrate_bps = bits_per_frame_learned * sample_rate / frame_size
    cr_codec_uniform = raw_input_bits_per_frame / bits_per_frame_uniform
    cr_codec_learned = raw_input_bits_per_frame / bits_per_frame_learned

    per_frame = _per_frame_summary(
        bits_full,
        tokens_per_frame=tokens_per_frame,
        skipped_tokens=skipped,
    )
    cr_per_frame = {}
    if per_frame:
        cr_per_frame = {
            "best": raw_input_bits_per_frame / per_frame["min"],
            "p95_cr": raw_input_bits_per_frame / per_frame["p05"],
            "median_cr": raw_input_bits_per_frame / per_frame["median"],
            "mean_cr": raw_input_bits_per_frame / per_frame["mean"],
            "p05_cr": raw_input_bits_per_frame / per_frame["p95"],
            "p01_cr": raw_input_bits_per_frame / per_frame["p99"],
            "worst": raw_input_bits_per_frame / per_frame["max"],
        }

    report = {
        "run_dir": str(run_dir),
        "modality": modality,
        "vocab_size": vocab_size,
        "num_levels": num_levels,
        "tokens_per_frame_per_level": tokens_per_frame_per_level,
        "tokens_per_frame": tokens_per_frame,
        "frame_size": frame_size,
        "input_bit_depth": _get_input_bit_depth(cfg),
        "sample_rate_hz": sample_rate,
        "context_length": context_length,
        "num_train_files": len(train_files),
        "num_val_files": len(val_files),
        "train_tokens": int(train_stream.size),
        "val_tokens": int(val_stream.size),
        "baselines": {
            "uniform_bits_per_token": math.log2(vocab_size),
            "unigram_bits_per_token_train": train_unigram,
            "unigram_bits_per_token_val": val_unigram,
        },
        "prior": prior_meta,
        "metrics": {
            "val_bits_per_token": val_bits_per_token,
            "val_bits_per_token_std": val_bits_per_token_std,
            "bits_per_frame_uniform": bits_per_frame_uniform,
            "bits_per_frame_learned": bits_per_frame_learned,
            "raw_bitrate_bps": raw_bitrate_bps,
            "uniform_codec_bitrate_bps": uniform_bitrate_bps,
            "learned_codec_bitrate_bps": learned_bitrate_bps,
            "cr_uplift_vs_uniform": cr_uplift_vs_uniform,
            "cr_uplift_vs_unigram": cr_uplift_vs_unigram,
            "cr_codec_uniform": cr_codec_uniform,
            "cr_codec_learned": cr_codec_learned,
            "per_frame_bits": per_frame,
            "per_frame_cr": cr_per_frame,
            "history": history_dict,
        },
    }

    if args.prior_type != "unigram":
        prior.save_weights(out_dir / "prior.weights.h5")
    report_path = out_dir / "entropy_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Pretty summary
    print("", file=sys.stderr)
    print(f"===== Entropy Measurement ({args.prior_type}) =====", file=sys.stderr)
    print(f"  vocab K                    : {vocab_size}", file=sys.stderr)
    print(f"  uniform   bits/token       : {math.log2(vocab_size):>8.4f}", file=sys.stderr)
    print(f"  unigram   bits/token (val) : {val_unigram:>8.4f}", file=sys.stderr)
    print(
        f"  prior     bits/token (val) : {val_bits_per_token:>8.4f}  (std {val_bits_per_token_std:.3f})",
        file=sys.stderr,
    )
    print(
        f"  bits/frame uniform / mean  : {bits_per_frame_uniform:>8.2f} / {bits_per_frame_learned:>8.2f}",
        file=sys.stderr,
    )
    print(
        f"  bitrate uniform / learned  : {uniform_bitrate_bps:>8.2f} / {learned_bitrate_bps:>8.2f} bps",
        file=sys.stderr,
    )
    if per_frame:
        print("  bits/frame   min /  p05 /  med /  p95 /  max:", file=sys.stderr)
        print(
            f"    {per_frame['min']:>7.1f} / {per_frame['p05']:>6.1f} / "
            f"{per_frame['median']:>6.1f} / {per_frame['p95']:>6.1f} / "
            f"{per_frame['max']:>7.1f}",
            file=sys.stderr,
        )
        print("  effective CR best / p95 / med / p05 / worst:", file=sys.stderr)
        print(
            f"    x{cr_per_frame['best']:>5.2f} / "
            f"x{cr_per_frame['p95_cr']:>5.2f} / "
            f"x{cr_per_frame['median_cr']:>5.2f} / "
            f"x{cr_per_frame['p05_cr']:>5.2f} / "
            f"x{cr_per_frame['worst']:>5.2f}",
            file=sys.stderr,
        )
    print(f"  CR uplift vs uniform / uni : x{cr_uplift_vs_uniform:.3f} / x{cr_uplift_vs_unigram:.3f}", file=sys.stderr)
    print(f"  Codec CR uniform / learned : x{cr_codec_uniform:.2f} / x{cr_codec_learned:.2f}", file=sys.stderr)
    print(f"  prior params               : {prior_meta.get('params', 0):,}", file=sys.stderr)
    print(f"  report → {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
