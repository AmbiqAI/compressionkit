"""Train a small RVQ-token prior on a frozen ECG compression model.

End-to-end prototype:

1. Reload a trained ECG compression model from ``--run-dir``.
2. Extract per-frame codebook tokens from a configurable number of
   PTB-XL validation recordings.
3. Train a small causal transformer prior on the token stream.
4. Sample novel token sequences and decode them to ECG waveforms,
   saving per-sample PNG plots.

This is a prototype: it trains for a small number of epochs on a
modest corpus to demonstrate that the RVQ decoder acts as a generator
when fed prior-sampled tokens. Productionising would expand the
training corpus, add conditional tokens (rhythm class, HR), and add
EMA/weight-averaging.

Example::

    uv run python scripts/train_rvq_prior.py \\
        --run-dir results/ecg_rvq_256hz_32x_golden \\
        --num-train-files 200 --epochs 10 \\
        --num-samples 6 --frames-per-sample 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.datasets.ecg import load_ecg_file_splits, load_ecg_signal
from compressionkit.generative import (
    build_prior,
    extract_rvq_tokens,
    sample_signals,
)


def _load_compressor(run_dir: Path) -> tuple[keras.Model, EcgRvqConfig]:
    from compressionkit.trainers.ecg_rvq import build_model

    cfg = EcgRvqConfig.model_validate_json((run_dir / "config.json").read_text())
    with tf.device("/CPU:0"):
        model = build_model(cfg)
        dummy = np.zeros(
            (1, 1, cfg.data.frame_size, max(1, cfg.data.num_leads or 1)),
            dtype=np.float32,
        )
        model(dummy, training=False)
    for name in ("best_model.weights.h5", "model.weights.h5"):
        p = run_dir / name
        if p.exists():
            model.load_weights(p)
            break
    else:
        raise FileNotFoundError(f"No weights in {run_dir}")
    return model, cfg


def _build_token_windows(
    tokens_flat: np.ndarray,
    context_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Chop a long token stream into fixed-length (inputs, targets) windows."""
    n = tokens_flat.size
    if n < context_length + 1:
        raise ValueError(f"Not enough tokens ({n}) for context {context_length}")
    starts = np.arange(0, n - context_length - 1, max(1, stride))
    xs = np.stack([tokens_flat[s : s + context_length] for s in starts]).astype(np.int32)
    ys = np.stack([tokens_flat[s + 1 : s + 1 + context_length] for s in starts]).astype(np.int32)
    return xs, ys


def _plot_samples(signals: np.ndarray, sample_rate: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = signals.shape[0]
    for i in range(n):
        sig = signals[i]
        t = np.arange(sig.size) / sample_rate
        fig, ax = plt.subplots(1, 1, figsize=(10, 2.5), constrained_layout=True)
        ax.plot(t, sig, lw=0.9, color="#d1495b")
        ax.set_title(f"Synthetic ECG sample {i} — {sig.size / sample_rate:.1f}s")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("amplitude (norm.)")
        ax.margins(x=0)
        fig.savefig(out_dir / f"synthetic_{i:02d}.png", dpi=130)
        plt.close(fig)

    # Grid overview
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.6 * n), sharex=True, sharey=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        t = np.arange(signals.shape[1]) / sample_rate
        ax.plot(t, signals[i], lw=0.8, color="#d1495b")
        ax.set_ylabel(f"s{i}", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    axes[0].set_title("Synthetic ECG samples")
    fig.savefig(out_dir / "grid.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-train-files", type=int, default=200)
    parser.add_argument(
        "--context-frames",
        type=int,
        default=4,
        help="Token context length expressed in frames (context_length = context_frames * tokens_per_frame).",
    )
    parser.add_argument("--stride-tokens", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)

    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--frames-per-sample", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)

    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <run-dir>/generative/")
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir.resolve()
    out_dir = args.output_dir or (run_dir / "generative")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading compressor from {run_dir}...", file=sys.stderr)
    compressor, cfg = _load_compressor(run_dir)
    data = cfg.data
    mcfg = cfg.model
    vocab_size = int(mcfg.latent_width)
    embed_dim_rvq = int(mcfg.embedding_dim)
    frame_size = int(data.frame_size)
    tokens_per_frame = frame_size // (2**mcfg.num_stages)
    print(
        f"  frame_size={frame_size} tokens_per_frame={tokens_per_frame} "
        f"vocab_size={vocab_size} num_levels={mcfg.num_levels}",
        file=sys.stderr,
    )
    if mcfg.num_levels != 1:
        print(
            "NOTE: prototype assumes num_levels==1; multi-level RVQ would need "
            "an interleaved token layout. Treating level-0 tokens only.",
            file=sys.stderr,
        )

    # --- 1. Extract tokens ------------------------------------------------
    train_files, _val_files, _ = load_ecg_file_splits(
        Path(data.datasets_dir),
        data.dataset_glob,
        seed=data.shuffle_seed,
    )
    train_files = train_files[: args.num_train_files]
    lead_index = getattr(data, "lead_index", 1) or 1
    print(f"Extracting tokens from {len(train_files)} files...", file=sys.stderr)
    signals = (load_ecg_signal(p, lead_index=lead_index) for p in train_files)
    tokens = extract_rvq_tokens(
        compressor,
        signals,
        frame_size=frame_size,
        num_leads=data.num_leads or 1,
        epsilon=data.epsilon,
        batch_size=64,
    )
    print(f"  extracted tokens shape: {tokens.shape}", file=sys.stderr)

    # Flatten frame tokens across frames (intra-frame contiguous, then inter-frame)
    level0 = tokens[..., 0].reshape(-1)  # (num_frames * tokens_per_frame,)
    print(f"  flat token stream length: {level0.size}", file=sys.stderr)

    # --- 2. Windows -------------------------------------------------------
    context_length = args.context_frames * tokens_per_frame
    xs, ys = _build_token_windows(level0, context_length, args.stride_tokens)
    n_val = max(1, int(xs.shape[0] * args.val_fraction))
    rng = np.random.default_rng(0)
    perm = rng.permutation(xs.shape[0])
    xs, ys = xs[perm], ys[perm]
    x_val, y_val = xs[:n_val], ys[:n_val]
    x_tr, y_tr = xs[n_val:], ys[n_val:]
    print(f"  windows: train={x_tr.shape[0]} val={x_val.shape[0]} ctx={context_length}", file=sys.stderr)

    # --- 3. Build & train prior ------------------------------------------
    prior = build_prior(
        vocab_size=vocab_size,
        context_length=context_length,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
    )
    prior.summary(print_fn=lambda s: print(s, file=sys.stderr))
    prior.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="token_acc")],
    )
    prior.fit(
        x_tr,
        y_tr,
        validation_data=(x_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=2,
    )

    weights_path = out_dir / "prior.weights.h5"
    prior.save_weights(weights_path)
    print(f"Saved prior weights → {weights_path}", file=sys.stderr)

    # --- 4. Sample + decode + plot ---------------------------------------
    sample_ctx = args.frames_per_sample * tokens_per_frame
    if sample_ctx != context_length:
        # Rebuild prior at sample context length and copy weights
        gen_prior = build_prior(
            vocab_size=vocab_size,
            context_length=sample_ctx,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
        )
        gen_prior(np.zeros((1, sample_ctx), dtype=np.int32), training=False)
        # Layer-by-layer weight copy where shapes match
        for src, dst in zip(prior.layers, gen_prior.layers):
            if src.name == "position_embedding":
                src_w = src.get_weights()[0]  # shape (1, L_src, D)
                L_src = src_w.shape[1]
                if sample_ctx <= L_src:
                    dst.set_weights([src_w[:, :sample_ctx, :]])
                else:
                    pad = np.zeros(
                        (1, sample_ctx - L_src, src_w.shape[2]),
                        dtype=src_w.dtype,
                    )
                    dst.set_weights([np.concatenate([src_w, pad], axis=1)])
                continue
            src_w = src.get_weights()
            if src_w:
                dst.set_weights(src_w)
    else:
        gen_prior = prior

    print(
        f"Sampling {args.num_samples} signals of {args.frames_per_sample} frames "
        f"(T={args.temperature}, top_k={args.top_k})...",
        file=sys.stderr,
    )
    signals = sample_signals(
        prior=gen_prior,
        compressor=compressor,
        num_samples=args.num_samples,
        num_frames=args.frames_per_sample,
        tokens_per_frame=tokens_per_frame,
        vocab_size=vocab_size,
        embedding_dim=embed_dim_rvq,
        frame_size=frame_size,
        num_leads=data.num_leads or 1,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.sample_seed,
    )
    plot_dir = out_dir / "samples"
    _plot_samples(signals, sample_rate=data.effective_sample_rate, out_dir=plot_dir)
    print(f"Wrote plots to {plot_dir}", file=sys.stderr)
    np.save(out_dir / "samples.npy", signals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
