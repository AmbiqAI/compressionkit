"""Measure RVQ token entropy with a genuinely 2D (time x level) prior.

``measure_rvq_entropy.py`` flattens ``(T, num_levels)`` RVQ indices into a
single interleaved 1-D stream (``[pos0_L0, pos0_L1, pos1_L0, pos1_L1, ...]``)
so every prior architecture can share one generic "predict next symbol in a
flat sequence" interface. That's simple, but it means a causal conv/RNN with
receptive field ``R`` (measured in flattened tokens) only reaches back
``R / num_levels`` *real time positions* — half its nominal reach is spent
re-observing the other level of a position it has already seen.

This script instead keeps level-0 and level-1 as two parallel per-position
channels over the *real* time axis (no interleaving), so a given
architectural depth reaches back ``2x`` as far in real time as the 1-D
approach for the same receptive-field-in-positions. To keep the comparison
fair (not just an easier task), level-1 at each step is still predicted
*conditioned on the true level-0 code at that same step* (teacher forcing),
mirroring exactly how a real sequential arithmetic decoder would work:
decode level-0 first, then use it to help decode level-1. This preserves the
same intra-step correlation modelling the 1-D interleaved approach gets "for
free", while doubling the real-time reach per layer.

Reuses token extraction / caching / compressor loading from
``measure_rvq_entropy.py`` so results land in the same
``<run_dir>/entropy_prior/<tag>/entropy_report.json`` schema and are directly
comparable via the same summary tooling.

Example::

    uv run python scripts/measure_rvq_entropy_2level.py \\
        --run-dir results/ecg_rvq_256hz_08x_golden \\
        --tag wavenet2d_L4 --context-frames 4 --num-layers 4 --epochs 50 \\
        --num-train-files 1600 --num-val-files 320
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_rvq_entropy as base

# ---------------------------------------------------------------------------
# Token extraction (thin wrapper around base script's extractors + cache)
# ---------------------------------------------------------------------------


def _cache_path(
    cache_root: Path, modality: str, split: str, n_files: int, frame_cap, frame_size, num_leads, lead_index
) -> Path:
    key = f"{modality}_{split}_n{n_files}_max{frame_cap}_fs{frame_size}_l{num_leads}_li{lead_index}.npy"
    return cache_root / key


def _get_or_extract_tokens(
    *,
    split: str,
    compressor: keras.Model,
    cfg,
    modality: str,
    frame_size: int,
    num_leads: int,
    lead_index: int,
    args: argparse.Namespace,
    cache_root: Path,
) -> np.ndarray:
    """Extract (or reuse cached) ``(N, tokens_per_frame_per_level, num_levels)`` tokens."""
    from compressionkit.datasets.ecg import load_ecg_file_splits
    from compressionkit.datasets.ppg import load_ppg_file_splits

    use_unified_cache = modality == "ppg" and hasattr(cfg.data, "unified_cache") and cfg.data.unified_cache.enabled
    max_frames = args.max_train_frames if split == "train" else args.max_val_frames

    files: list[Path] = []
    n_files_for_key = 0
    if use_unified_cache or modality == "ppg-h5":
        pass
    elif modality == "ppg":
        train_files, val_files, _ = load_ppg_file_splits(
            Path(cfg.data.datasets_dir), cfg.data.dataset_glob, seed=cfg.data.shuffle_seed
        )
        files = train_files if split == "train" else val_files
        n = args.num_train_files if split == "train" else args.num_val_files
        if n is not None and n >= 0:
            files = files[:n]
        n_files_for_key = len(files)
    else:
        train_files, val_files, _ = load_ecg_file_splits(
            Path(cfg.data.datasets_dir), cfg.data.dataset_glob, seed=cfg.data.shuffle_seed
        )
        files = train_files if split == "train" else val_files
        n = args.num_train_files if split == "train" else args.num_val_files
        if n is not None and n >= 0:
            files = files[:n]
        n_files_for_key = len(files)

    cache_root.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_root, modality, split, n_files_for_key, max_frames, frame_size, num_leads, lead_index)
    if path.exists():
        print(f"      [cache] reuse {split} tokens \u2190 {path.name}", file=sys.stderr)
        return np.load(path)

    if use_unified_cache:
        tokens = base._extract_unified_cache_tokens(compressor, cfg, split=split, max_frames=max_frames, batch_size=64)
    elif modality == "ppg-h5":
        tokens = base._extract_ppg_h5_split_tokens(compressor, cfg, split=split, max_frames=max_frames, batch_size=64)
    elif modality == "ppg":
        tokens = base._extract_ppg_split_tokens(files, compressor, cfg, max_frames=max_frames, batch_size=64)
    else:
        tokens = base._extract_split_tokens(
            files,
            compressor,
            frame_size=frame_size,
            num_leads=num_leads,
            epsilon=cfg.data.epsilon,
            lead_index=lead_index,
        )
    np.save(path, tokens)
    print(f"      [cache] saved {split} tokens \u2192 {path.name}", file=sys.stderr)
    return tokens


# ---------------------------------------------------------------------------
# 2D (time x level) wavenet-style prior: two parallel channels, causal, with
# level-1 conditioned on the same-step level-0 code (teacher forcing).
# ---------------------------------------------------------------------------


def _causal_wavenet_stack(x, *, embed_dim: int, num_layers: int, kernel_size: int, name_prefix: str):
    """Gated-residual dilated causal Conv1D stack (WaveNet-style block)."""
    skip_sum = None
    for i in range(num_layers):
        d = 2**i
        h = keras.layers.Conv1D(
            2 * embed_dim, kernel_size, padding="causal", dilation_rate=d, name=f"{name_prefix}_gc_d{d}"
        )(x)
        a, b = keras.ops.split(h, 2, axis=-1)
        gated = keras.ops.tanh(a) * keras.ops.sigmoid(b)
        out = keras.layers.Conv1D(embed_dim, 1, name=f"{name_prefix}_out_d{d}")(gated)
        skip = keras.layers.Conv1D(embed_dim, 1, name=f"{name_prefix}_skip_d{d}")(gated)
        x = keras.layers.Add(name=f"{name_prefix}_res_d{d}")([x, out])
        skip_sum = skip if skip_sum is None else keras.layers.Add(name=f"{name_prefix}_skipsum_d{d}")([skip_sum, skip])
    return keras.layers.ReLU(name=f"{name_prefix}_final_relu")(skip_sum)


def build_wavenet2d_prior(
    *,
    vocab_size: int,
    context_length: int,
    embed_dim: int = 64,
    num_layers: int = 4,
    kernel_size: int = 5,
    name: str = "rvq_wavenet2d_prior",
) -> keras.Model:
    """Two-channel (level0, level1) causal prior over the real time axis.

    Inputs ``level0_in``, ``level1_in``: ``(B, context_length)`` int tokens.
    Outputs ``level0_logits``, ``level1_logits``: ``(B, context_length, vocab_size)``
    next-*position* logits. ``level1_logits`` at position ``t`` is conditioned
    on the true ``level0`` code at position ``t+1`` (teacher forcing) via a
    shifted auxiliary input — matching a real sequential arithmetic decoder
    that decodes level0 before level1 at each step.
    """
    level0_in = keras.Input(shape=(context_length,), dtype="int32", name="level0_in")
    level1_in = keras.Input(shape=(context_length,), dtype="int32", name="level1_in")

    embed0 = keras.layers.Embedding(vocab_size, embed_dim, name="level0_embedding")
    embed1 = keras.layers.Embedding(vocab_size, embed_dim, name="level1_embedding")

    e0 = embed0(level0_in)
    e1 = embed1(level1_in)
    x = keras.layers.Add(name="level_sum")([e0, e1])

    h = _causal_wavenet_stack(
        x, embed_dim=embed_dim, num_layers=num_layers, kernel_size=kernel_size, name_prefix="wn2d"
    )

    level0_logits = keras.layers.Dense(vocab_size, name="level0_logits")(h)

    # Level-1 head sees h_t (causal state) AND the *next* step's true level0
    # embedding (teacher forcing) — same trick a coarse-to-fine / residual
    # token model uses. Pad the shift with a learned "unknown" embedding so
    # shapes stay static (last position has no "next" ground truth).
    #
    # Implemented with built-in Cropping1D + ZeroPadding1D rather than a
    # closure-based `keras.layers.Lambda` — the same class of Lambda this
    # repo's own `SplitHalf` layer (see `causal_priors.py`) was introduced to
    # replace, since a Python-closure Lambda reliably fails to deserialize
    # via `keras.models.load_model()` (even with `safe_mode=False`).
    next_level0_embed = embed0(level0_in)  # reuse table; shift below
    cropped = keras.layers.Cropping1D(cropping=(1, 0), name="drop_first_step")(next_level0_embed)
    shifted = keras.layers.ZeroPadding1D(padding=(0, 1), name="shift_level0_embed")(cropped)
    level1_input = keras.layers.Concatenate(name="level1_head_input")([h, shifted])
    level1_logits = keras.layers.Dense(vocab_size, name="level1_logits")(level1_input)

    model = keras.Model([level0_in, level1_in], [level0_logits, level1_logits], name=name)
    return model


# ---------------------------------------------------------------------------
# Windowing / evaluation for the two-stream model
# ---------------------------------------------------------------------------


def _build_windows_2level(level0: np.ndarray, level1: np.ndarray, context_length: int, stride: int):
    n = level0.size
    if n < context_length + 1:
        raise ValueError(f"Token stream too short ({n}) for context {context_length}")
    starts = np.arange(0, n - context_length - 1, max(1, stride))
    x0 = np.stack([level0[s : s + context_length] for s in starts]).astype(np.int32)
    x1 = np.stack([level1[s : s + context_length] for s in starts]).astype(np.int32)
    y0 = np.stack([level0[s + 1 : s + 1 + context_length] for s in starts]).astype(np.int32)
    y1 = np.stack([level1[s + 1 : s + 1 + context_length] for s in starts]).astype(np.int32)
    return x0, x1, y0, y1


def _per_position_nll_bits_2level(
    model: keras.Model,
    level0: np.ndarray,
    level1: np.ndarray,
    *,
    context_length: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Combined (level0+level1) bits/token, one value per real time position.

    Returns an array of shape ``(N - context_length,) * 2`` flattened so the
    mean directly matches the "bits per token" convention used elsewhere
    (each real time position contributes 2 tokens: level0 and level1).

    Level-0 and level-1 are scored at *different* offsets within each window
    — this is intentional, not an inconsistency. Level-0 is scored at the
    last window position (``logits0[:, -1, :]``), predicting the token one
    step beyond the window (``level0[s + context_length]``), matching the
    standard sliding-window "predict next" convention used elsewhere in this
    codebase (e.g. ``measure_rvq_entropy.py``'s ``_per_token_nll_bits``).

    Level-1 CANNOT be scored the same way: `build_wavenet2d_prior`'s level-1
    head is teacher-forced on the true level-0 code at the *same* step, taken
    from `level0_in`'s own window — but the window never contains
    ``level0[s + context_length]`` (that's one token beyond it), so the
    shifted input is necessarily the zero/"unknown" padding at exactly the
    last window position. Scoring level-1 there — as an earlier version of
    this function did — always evaluates the model at the one position per
    window where its designed side information is unavailable, silently
    understating how well the model performs when it *does* have that
    information (as it does at every other position during training). Level-1
    is instead scored at the second-to-last window position
    (``logits1[:, -2, :]``), predicting ``level1[s + context_length - 1]``
    — the last token actually inside the window, where the shifted level-0
    embedding is the real (non-padded) code, matching how this head is meant
    to be used.
    """
    n = level0.size
    if n <= context_length:
        return np.zeros((0,), dtype=np.float64)
    starts = np.arange(n - context_length, dtype=np.int64)
    t0 = level0[context_length:].astype(np.int32)
    t1 = level1[context_length - 1 : -1].astype(np.int32)

    bits0 = np.zeros(starts.size, dtype=np.float64)
    bits1 = np.zeros(starts.size, dtype=np.float64)
    log2 = math.log(2.0)
    for i in range(0, starts.size, batch_size):
        sl = slice(i, i + batch_size)
        bs = starts[sl]
        x0 = np.stack([level0[s : s + context_length] for s in bs]).astype(np.int32)
        x1 = np.stack([level1[s : s + context_length] for s in bs]).astype(np.int32)
        logits0, logits1 = model([x0, x1], training=False)
        last0 = keras.ops.convert_to_numpy(logits0[:, -1, :]).astype(np.float64)
        last1 = keras.ops.convert_to_numpy(logits1[:, -2, :]).astype(np.float64)
        for last, targets, bits in ((last0, t0, bits0), (last1, t1, bits1)):
            last_c = last - last.max(axis=-1, keepdims=True)
            log_norm = np.log(np.exp(last_c).sum(axis=-1, keepdims=True))
            log_probs = last_c - log_norm
            idx = np.arange(bs.size)
            bits[sl] = -log_probs[idx, targets[sl]] / log2
    # Interleave so downstream per-frame aggregation (which assumes one flat
    # stream) sees the same total token count as the 1-D approach.
    combined = np.empty(bits0.size * 2, dtype=np.float64)
    combined[0::2] = bits0
    combined[1::2] = bits1
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=["auto", "ecg", "ppg", "ppg-h5"], default="auto")
    parser.add_argument("--tag", type=str, default="wavenet2d")
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--stride-tokens", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-train-files", type=int, default=400)
    parser.add_argument("--num-val-files", type=int, default=80)
    parser.add_argument("--max-train-frames", type=int, default=None)
    parser.add_argument("--max-val-frames", type=int, default=None)
    parser.add_argument("--per-frame-batch", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir.resolve()
    out_root = args.output_dir or (run_dir / "entropy_prior")
    out_dir = out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = run_dir / "entropy_prior" / "_token_cache"

    print(f"[1/5] Loading compressor from {run_dir} ...", file=sys.stderr)
    compressor, cfg, modality = base._load_compressor(run_dir, modality=args.modality)
    mcfg = cfg.model
    vocab_size = int(mcfg.latent_width)
    if int(mcfg.num_levels) != 2:
        raise ValueError(f"2-level prior requires num_levels==2, got {mcfg.num_levels}")
    frame_size = base._get_frame_size(cfg, modality)
    num_leads = int(getattr(cfg.data, "num_leads", 1) or 1)
    lead_index = int(getattr(cfg.data, "lead_index", 1) or 1)
    tokens_per_frame_per_level = frame_size // (2**mcfg.num_stages)
    tokens_per_frame = tokens_per_frame_per_level * 2

    print(
        f"      frame_size={frame_size} positions/frame={tokens_per_frame_per_level} vocab={vocab_size} modality={modality}",
        file=sys.stderr,
    )

    print("[2/5] Extracting tokens (per-level, real time order) ...", file=sys.stderr)
    train_tok = _get_or_extract_tokens(
        split="train",
        compressor=compressor,
        cfg=cfg,
        modality=modality,
        frame_size=frame_size,
        num_leads=num_leads,
        lead_index=lead_index,
        args=args,
        cache_root=cache_root,
    )
    val_tok = _get_or_extract_tokens(
        split="val",
        compressor=compressor,
        cfg=cfg,
        modality=modality,
        frame_size=frame_size,
        num_leads=num_leads,
        lead_index=lead_index,
        args=args,
        cache_root=cache_root,
    )
    if train_tok.size == 0 or val_tok.size == 0:
        raise RuntimeError("Token extraction produced an empty train or validation split.")

    train_level0 = train_tok[..., 0].reshape(-1).astype(np.int32)
    train_level1 = train_tok[..., 1].reshape(-1).astype(np.int32)
    val_level0 = val_tok[..., 0].reshape(-1).astype(np.int32)
    val_level1 = val_tok[..., 1].reshape(-1).astype(np.int32)
    print(f"      train positions={train_level0.size}  val positions={val_level0.size}", file=sys.stderr)

    context_length = args.context_frames * tokens_per_frame_per_level
    print(f"[3/5] Building 2D wavenet prior (L={args.num_layers}, ctx={context_length} positions) ...", file=sys.stderr)

    with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
        model = build_wavenet2d_prior(
            vocab_size=vocab_size,
            context_length=context_length,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            kernel_size=args.kernel,
        )
    model.summary(print_fn=lambda s: print("      " + s, file=sys.stderr))
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay),
        loss={
            "level0_logits": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "level1_logits": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        },
    )

    x0_tr, x1_tr, y0_tr, y1_tr = _build_windows_2level(train_level0, train_level1, context_length, args.stride_tokens)
    x0_val, x1_val, y0_val, y1_val = _build_windows_2level(val_level0, val_level1, context_length, args.stride_tokens)
    print(f"      train windows={x0_tr.shape[0]}  val windows={x0_val.shape[0]}", file=sys.stderr)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience, restore_best_weights=True
    )
    print("[4/5] Training 2D prior ...", file=sys.stderr)
    history = model.fit(
        {"level0_in": x0_tr, "level1_in": x1_tr},
        {"level0_logits": y0_tr, "level1_logits": y1_tr},
        validation_data=(
            {"level0_in": x0_val, "level1_in": x1_val},
            {"level0_logits": y0_val, "level1_logits": y1_val},
        ),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[early_stop],
        verbose=2,
    )
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    print("[5/5] Aggregating per-token / per-frame bitrate stats ...", file=sys.stderr)
    bits_full = _per_position_nll_bits_2level(
        model, val_level0, val_level1, context_length=context_length, batch_size=args.per_frame_batch
    )
    val_bits_per_token = float(bits_full.mean())
    val_bits_per_token_std = float(bits_full.std())

    train_unigram0 = base._baseline_bits_per_token(vocab_size, train_level0)
    train_unigram1 = base._baseline_bits_per_token(vocab_size, train_level1)
    val_unigram0 = base._baseline_bits_per_token(vocab_size, val_level0)
    val_unigram1 = base._baseline_bits_per_token(vocab_size, val_level1)
    val_unigram = (val_unigram0 + val_unigram1) / 2.0

    bits_per_frame_uniform = math.log2(vocab_size) * tokens_per_frame
    bits_per_frame_learned = val_bits_per_token * tokens_per_frame
    cr_uplift_vs_uniform = math.log2(vocab_size) / val_bits_per_token
    cr_uplift_vs_unigram = val_unigram / val_bits_per_token
    raw_input_bits_per_frame = frame_size * base._get_input_bit_depth(cfg)
    sample_rate = base._get_sample_rate(cfg, modality)
    raw_bitrate_bps = sample_rate * base._get_input_bit_depth(cfg)
    uniform_bitrate_bps = bits_per_frame_uniform * sample_rate / frame_size
    learned_bitrate_bps = bits_per_frame_learned * sample_rate / frame_size
    cr_codec_uniform = raw_input_bits_per_frame / bits_per_frame_uniform
    cr_codec_learned = raw_input_bits_per_frame / bits_per_frame_learned

    per_frame = base._per_frame_summary(bits_full, tokens_per_frame=tokens_per_frame, skipped_tokens=context_length * 2)
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

    receptive_field = 1 + (args.kernel - 1) * (2**args.num_layers - 1)
    prior_meta = {
        "type": "wavenet2d",
        "structure": "2d_per_level_channels",
        "embed_dim": args.embed_dim,
        "num_layers": args.num_layers,
        "kernel": args.kernel,
        "receptive_field": receptive_field,
        "receptive_field_note": "in real time POSITIONS (not flattened tokens) — directly comparable to 1D rf/2",
        "params": int(model.count_params()),
        "context_length": context_length,
    }

    report = {
        "run_dir": str(run_dir),
        "modality": modality,
        "vocab_size": vocab_size,
        "num_levels": 2,
        "tokens_per_frame_per_level": tokens_per_frame_per_level,
        "tokens_per_frame": tokens_per_frame,
        "frame_size": frame_size,
        "input_bit_depth": base._get_input_bit_depth(cfg),
        "sample_rate_hz": sample_rate,
        "context_length": context_length,
        "num_train_files": args.num_train_files,
        "num_val_files": args.num_val_files,
        "train_tokens": int(train_level0.size * 2),
        "val_tokens": int(val_level0.size * 2),
        "baselines": {
            "uniform_bits_per_token": math.log2(vocab_size),
            "unigram_bits_per_token_train": (train_unigram0 + train_unigram1) / 2.0,
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

    model.save_weights(out_dir / "prior.weights.h5")
    report_path = out_dir / "entropy_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("", file=sys.stderr)
    print("===== Entropy Measurement (wavenet2d) =====", file=sys.stderr)
    print(f"  vocab K                    : {vocab_size}", file=sys.stderr)
    print(f"  uniform   bits/token       : {math.log2(vocab_size):>8.4f}", file=sys.stderr)
    print(f"  unigram   bits/token (val) : {val_unigram:>8.4f}", file=sys.stderr)
    print(
        f"  prior     bits/token (val) : {val_bits_per_token:>8.4f}  (std {val_bits_per_token_std:.3f})",
        file=sys.stderr,
    )
    print(f"  CR uplift vs uniform / uni : x{cr_uplift_vs_uniform:.3f} / x{cr_uplift_vs_unigram:.3f}", file=sys.stderr)
    print(f"  Codec CR uniform / learned : x{cr_codec_uniform:.2f} / x{cr_codec_learned:.2f}", file=sys.stderr)
    print(f"  receptive field (positions): {receptive_field}", file=sys.stderr)
    print(f"  prior params               : {prior_meta['params']:,}", file=sys.stderr)
    print(f"  report \u2192 {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
