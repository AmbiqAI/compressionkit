"""Sweep CNN-prior context windows on a frozen RVQ compressor.

For each context length we (a) train a small dilated causal CNN prior
on the encoded token stream, (b) measure validation bits/token and
effective compression ratio, and (c) collect everything into a single
JSON + Markdown summary.  Each individual run is delegated to
``scripts/measure_rvq_entropy.py`` so the token cache, compressor load,
and per-frame statistics paths are all shared with the existing
single-run tool.

The CNN depth for each window is chosen so that the receptive field is
at least the context length::

    rf = 1 + (kernel - 1) * (2**num_layers - 1) >= ctx

Usage::

    uv run python scripts/sweep_rvq_entropy_window.py \\
        --run-dir results/ecg_rvq_256hz_32x_golden \\
        --context-frames 4 8 16 32 64 128 256 \\
        --epochs 20

The script is idempotent: tags include the window length, so reruns
with the same ``--tag-prefix`` reuse existing reports.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTROPY_SCRIPT = REPO_ROOT / "scripts" / "measure_rvq_entropy.py"
# Prefer the project venv if present; falling back to sys.executable supports
# both `uv run python ...` and direct invocations from inside the venv.
_VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
PYTHON_BIN = str(_VENV_PY) if _VENV_PY.exists() else sys.executable


def _depth_for_context(ctx: int, kernel: int) -> int:
    """Smallest ``num_layers`` whose dilated receptive field covers ``ctx``."""
    if ctx <= 1:
        return 1
    # rf = 1 + (kernel - 1) * (2**L - 1)  =>  2**L >= (ctx-1)/(kernel-1) + 1
    needed = math.ceil(math.log2((ctx - 1) / (kernel - 1) + 1))
    return max(2, int(needed))


def _frame_seconds(frame_size: int, sample_rate: int) -> float:
    return frame_size / float(sample_rate)


def _run_one(
    *,
    run_dir: Path,
    context_frames: int,
    tag: str,
    epochs: int,
    cnn_embed_dim: int,
    cnn_num_layers: int,
    cnn_kernel: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    stride_tokens: int,
    num_train_files: int,
    num_val_files: int,
    extra_args: list[str],
) -> dict:
    cmd = [
        PYTHON_BIN,
        str(ENTROPY_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--prior-type",
        "cnn",
        "--context-frames",
        str(context_frames),
        "--cnn-embed-dim",
        str(cnn_embed_dim),
        "--cnn-num-layers",
        str(cnn_num_layers),
        "--cnn-kernel",
        str(cnn_kernel),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
        "--stride-tokens",
        str(stride_tokens),
        "--num-train-files",
        str(num_train_files),
        "--num-val-files",
        str(num_val_files),
        "--tag",
        tag,
        *extra_args,
    ]
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    report_path = run_dir / "entropy_prior" / tag / "entropy_report.json"
    return json.loads(report_path.read_text())


def _summarize(reports: list[tuple[int, dict]], frame_sec: float) -> str:
    lines = [
        "| frames | seconds | ctx tokens | params | bpt | RVQ-stage CR | end-to-end CR | uplift× |",
        "|-------:|--------:|-----------:|-------:|----:|------------:|--------------:|--------:|",
    ]
    for ctxf, r in reports:
        m = r["metrics"]
        pr = r["prior"]
        lines.append(
            f"| {ctxf} | {ctxf * frame_sec:.1f} | {r['context_length']} | "
            f"{pr.get('params', 0):,} | {m['val_bits_per_token']:.3f} | "
            f"{m['cr_codec_uniform']:.2f}× | {m['cr_codec_learned']:.2f}× | "
            f"{m['cr_uplift_vs_uniform']:.3f}× |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Frozen RVQ compressor results dir.")
    parser.add_argument(
        "--context-frames",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256],
        help="Window sizes to sweep (in frames).",
    )
    parser.add_argument("--tag-prefix", default="winsweep", help="Per-run tag becomes <prefix>_f<context_frames>.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stride-tokens", type=int, default=8)
    parser.add_argument("--cnn-embed-dim", type=int, default=64, help="Channel width — held fixed across the sweep.")
    parser.add_argument("--cnn-kernel", type=int, default=5)
    parser.add_argument("--cnn-num-layers", type=int, default=0, help="Override depth (0 = auto from context).")
    parser.add_argument("--num-train-files", type=int, default=-1)
    parser.add_argument("--num-val-files", type=int, default=-1)
    parser.add_argument("--summary-name", default="window_sweep_summary")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Forwarded verbatim to measure_rvq_entropy.py after a literal ``--`` separator.",
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir.resolve()
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    frame_size = int(cfg["data"]["frame_size"])
    sample_rate = int(cfg["data"].get("target_sample_rate") or cfg["data"].get("sampling_rate"))
    frame_sec = _frame_seconds(frame_size, sample_rate)
    print(f"Compressor: {run_dir.name}  frame={frame_size}@{sample_rate}Hz ({frame_sec:.2f}s/frame)", flush=True)

    extra: list[str] = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    reports: list[tuple[int, dict]] = []
    for ctxf in args.context_frames:
        depth = args.cnn_num_layers if args.cnn_num_layers > 0 else _depth_for_context(ctxf * 32, args.cnn_kernel)
        # NB. ctxf*32 assumes tokens_per_frame ≈ 32 for the golden 32× config.
        # The actual context_length printed by the inner script is authoritative.
        tag = f"{args.tag_prefix}_f{ctxf}"
        report = _run_one(
            run_dir=run_dir,
            context_frames=ctxf,
            tag=tag,
            epochs=args.epochs,
            cnn_embed_dim=args.cnn_embed_dim,
            cnn_num_layers=depth,
            cnn_kernel=args.cnn_kernel,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            stride_tokens=args.stride_tokens,
            num_train_files=args.num_train_files,
            num_val_files=args.num_val_files,
            extra_args=extra,
        )
        reports.append((ctxf, report))

    out_dir = run_dir / "entropy_prior" / args.summary_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_dir": str(run_dir),
        "frame_size": frame_size,
        "sample_rate": sample_rate,
        "frame_seconds": frame_sec,
        "cnn_embed_dim": args.cnn_embed_dim,
        "cnn_kernel": args.cnn_kernel,
        "epochs": args.epochs,
        "results": [
            {
                "context_frames": ctxf,
                "context_seconds": ctxf * frame_sec,
                "context_tokens": r["context_length"],
                "prior": r["prior"],
                "metrics": {k: v for k, v in r["metrics"].items() if k != "history"},
                "tag": f"{args.tag_prefix}_f{ctxf}",
            }
            for ctxf, r in reports
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    md = (
        f"# Window sweep — {run_dir.name}\n\n"
        f"Frame {frame_size} samples @ {sample_rate} Hz = {frame_sec:.2f} s. "
        f"CNN: embed={args.cnn_embed_dim}, kernel={args.cnn_kernel}, "
        f"depth=auto, epochs={args.epochs}.\n\n" + _summarize(reports, frame_sec) + "\n"
    )
    (out_dir / "summary.md").write_text(md)
    print("\n" + md, flush=True)
    print(f"\nWrote: {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
