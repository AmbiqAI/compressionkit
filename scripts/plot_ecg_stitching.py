"""Generate diagnostic plots of ECG stitching quality.

Produces, for a single validation recording:

1. One figure per stitching method, showing the full 1-D reconstruction
   overlaid on the original signal on top, and a grid of seam-centred
   zoom subplots below (different x-axis, shared y-axis) so the visual
   scale is preserved across zooms.
2. A comparison figure that stacks all methods at one representative
   seam so the different windowing strategies can be contrasted directly.

Example::

    uv run python scripts/plot_ecg_stitching.py \\
        --run-dir results/ecg_rvq_256hz_32x_golden
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.datasets.ecg import load_ecg_file_splits, load_ecg_signal
from compressionkit.evaluation.stitching import STITCH_METHODS, stitch

DEFAULT_METHODS: list[str] = ["hard_concat", "overlap_add", "linear_crossfade", "tukey_overlap_add"]
ZOOM_WIDTH_SAMPLES = 80  # window shown around each seam


def _load_model(run_dir: Path):
    from compressionkit.trainers.ecg_rvq import build_model

    cfg = EcgRvqConfig.model_validate_json((run_dir / "config.json").read_text())
    model = build_model(cfg)
    dummy = np.zeros((1, 1, cfg.data.frame_size, max(1, cfg.data.num_leads or 1)), dtype=np.float32)
    model(dummy, training=False)
    weights = run_dir / "best_model.weights.h5"
    if not weights.exists():
        weights = run_dir / "model.weights.h5"
    model.load_weights(weights)
    return model, cfg


def _make_predict_fn(model, batch_size: int = 32):
    def _predict(batch: np.ndarray) -> np.ndarray:
        return model.predict(batch, batch_size=batch_size, verbose=0)

    return _predict


def _seam_positions(signal_len: int, frame_size: int, hop_ratio: float, method: str) -> list[int]:
    """Return representative seam indices for plotting."""
    hop = frame_size if method == "hard_concat" else max(1, int(frame_size * hop_ratio))
    first = frame_size  # first interior seam
    return list(range(first, signal_len - frame_size // 2, hop))


def _plot_method_figure(
    *,
    method: str,
    original: np.ndarray,
    recon: np.ndarray,
    sample_rate: int,
    frame_size: int,
    hop_ratio: float,
    num_zooms: int,
    out_path: Path,
) -> None:
    seams = _seam_positions(len(original), frame_size, hop_ratio, method)
    if not seams:
        return
    # Pick evenly spaced seams for zooms
    idx = np.linspace(0, len(seams) - 1, min(num_zooms, len(seams))).astype(int)
    chosen_seams = [seams[i] for i in idx]

    t = np.arange(len(original)) / sample_rate

    # Layout: 2 rows of axes — top full-width, bottom row of N zooms
    fig = plt.figure(figsize=(12, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, len(chosen_seams), height_ratios=[1.0, 1.1])
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.plot(t, original, color="#6e6e6e", lw=0.8, label="original", alpha=0.8)
    ax_full.plot(t, recon, color="#d1495b", lw=0.8, label="reconstructed")
    for s in seams:
        ax_full.axvline(s / sample_rate, color="#2e86ab", lw=0.4, ls=":", alpha=0.5)
    ax_full.set_title(f"{method} — full {len(original) / sample_rate:.1f}s reconstruction")
    ax_full.set_xlabel("time [s]")
    ax_full.set_ylabel("amplitude")
    ax_full.legend(loc="upper right", fontsize=8)
    ax_full.margins(x=0)

    # Shared y-range for all zoom plots — derived from the full signal so
    # amplitude comparisons across subplots are visually faithful.
    y_lo = float(min(original.min(), recon.min()))
    y_hi = float(max(original.max(), recon.max()))
    pad = 0.05 * (y_hi - y_lo + 1e-9)
    y_range = (y_lo - pad, y_hi + pad)

    for col, seam in enumerate(chosen_seams):
        ax = fig.add_subplot(gs[1, col])
        lo = max(0, seam - ZOOM_WIDTH_SAMPLES // 2)
        hi = min(len(original), seam + ZOOM_WIDTH_SAMPLES // 2)
        ax.plot(t[lo:hi], original[lo:hi], color="#6e6e6e", lw=1.0, alpha=0.8)
        ax.plot(t[lo:hi], recon[lo:hi], color="#d1495b", lw=1.2)
        ax.axvline(seam / sample_rate, color="#2e86ab", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylim(y_range)
        ax.set_title(f"seam @ {seam / sample_rate:.2f}s", fontsize=9)
        ax.set_xlabel("time [s]", fontsize=8)
        if col == 0:
            ax.set_ylabel("amplitude", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_comparison_figure(
    *,
    methods: list[str],
    original: np.ndarray,
    recons: dict[str, np.ndarray],
    sample_rate: int,
    frame_size: int,
    hop_ratio: float,
    out_path: Path,
) -> None:
    """One seam zoom per method, stacked, sharing both x and y axes."""
    # Use the first interior seam of the hop grid for fairness
    seam = frame_size
    lo = max(0, seam - ZOOM_WIDTH_SAMPLES // 2)
    hi = min(len(original), seam + ZOOM_WIDTH_SAMPLES // 2)

    y_lo = float(min([original[lo:hi].min()] + [r[lo:hi].min() for r in recons.values()]))
    y_hi = float(max([original[lo:hi].max()] + [r[lo:hi].max() for r in recons.values()]))
    pad = 0.05 * (y_hi - y_lo + 1e-9)
    y_range = (y_lo - pad, y_hi + pad)

    t = np.arange(len(original)) / sample_rate

    fig, axes = plt.subplots(
        len(methods),
        1,
        figsize=(8, 1.6 * len(methods) + 0.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        ax.plot(t[lo:hi], original[lo:hi], color="#6e6e6e", lw=1.0, alpha=0.8, label="original")
        ax.plot(t[lo:hi], recons[method][lo:hi], color="#d1495b", lw=1.2, label=method)
        ax.axvline(seam / sample_rate, color="#2e86ab", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylim(y_range)
        ax.set_ylabel(method, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    axes[0].set_title(f"seam zoom @ {seam / sample_rate:.2f}s — method comparison")
    axes[-1].set_xlabel("time [s]")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--hop-ratio", type=float, default=0.5)
    parser.add_argument("--num-zooms", type=int, default=4)
    parser.add_argument(
        "--record-index", type=int, default=0, help="Index into the sorted val split (for reproducibility)."
    )
    parser.add_argument("--lead-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=sorted(STITCH_METHODS))
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <run-dir>/stitching_plots/")
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir.resolve()
    model, cfg = _load_model(run_dir)
    data = cfg.data
    sample_rate = data.effective_sample_rate
    frame_size = data.frame_size

    _, val_files, _ = load_ecg_file_splits(
        Path(data.datasets_dir),
        data.dataset_glob,
        seed=data.shuffle_seed,
    )
    if not val_files:
        parser.error("No validation files available.")

    fpath = val_files[args.record_index % len(val_files)]
    print(f"Using {fpath.name}", file=sys.stderr)
    signal = np.asarray(
        load_ecg_signal(fpath, lead_index=args.lead_index),
        dtype=np.float32,
    ).reshape(-1)
    target = int(args.duration_sec * sample_rate)
    if signal.size > target:
        signal = signal[:target]
    if signal.size < 2 * frame_size:
        parser.error("Selected record too short for the configured frame size.")

    predict_fn = _make_predict_fn(model, args.batch_size)

    out_dir = args.output_dir or (run_dir / "stitching_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    recons: dict[str, np.ndarray] = {}
    for method in args.methods:
        kwargs = {"epsilon": data.epsilon}
        if method != "hard_concat":
            kwargs["hop_ratio"] = args.hop_ratio
        print(f"  [{method}] running stitch...", file=sys.stderr)
        recons[method] = stitch(method, predict_fn, signal, frame_size, **kwargs)

        out_path = out_dir / f"stitching_{method}.png"
        _plot_method_figure(
            method=method,
            original=signal,
            recon=recons[method],
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_ratio=args.hop_ratio,
            num_zooms=args.num_zooms,
            out_path=out_path,
        )
        print(f"    wrote {out_path}", file=sys.stderr)

    compare_path = out_dir / "stitching_comparison.png"
    _plot_comparison_figure(
        methods=args.methods,
        original=signal,
        recons=recons,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_ratio=args.hop_ratio,
        out_path=compare_path,
    )
    print(f"wrote {compare_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
