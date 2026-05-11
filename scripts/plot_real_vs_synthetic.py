"""Plot real PTB-XL ECG recordings alongside synthetic ECG from the RVQ prior.

Generates a side-by-side comparison figure with clear "REAL" vs "SYNTHETIC"
labels so the quality of the generative model is immediately apparent.

Example::

    uv run python scripts/plot_real_vs_synthetic.py \
        --run-dir results/ecg_rvq_256hz_32x_golden \
        --synth-dir results/ecg_rvq_256hz_32x_golden/generative_scaled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.datasets.ecg import load_ecg_file_splits, load_ecg_signal


def _norm(sig: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation."""
    s = sig - sig.mean()
    std = s.std()
    if std > 1e-8:
        s /= std
    return s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Trained compressor run directory (contains config.json).")
    parser.add_argument("--synth-dir", type=Path, default=None,
                        help="Directory with samples.npy. Default: <run-dir>/generative_scaled/")
    parser.add_argument("--num-real", type=int, default=8,
                        help="Number of real recordings to show.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path. Default: <synth-dir>/real_vs_synthetic.png")
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir.resolve()
    synth_dir = (args.synth_dir or run_dir / "generative_scaled").resolve()
    out_path = args.output or synth_dir / "real_vs_synthetic.png"

    # Load config
    cfg = EcgRvqConfig.model_validate_json((run_dir / "config.json").read_text())
    data = cfg.data
    sr = data.effective_sample_rate

    # Load synthetic samples
    samples_path = synth_dir / "samples.npy"
    if not samples_path.exists():
        print(f"ERROR: {samples_path} not found", file=sys.stderr)
        return 1
    synth = np.load(samples_path)
    n_synth = synth.shape[0]

    # Load real recordings (from validation set for fairness)
    _, val_files, _ = load_ecg_file_splits(
        Path(data.datasets_dir), data.dataset_glob, seed=data.shuffle_seed,
    )
    lead_index = getattr(data, "lead_index", 1) or 1
    n_real = min(args.num_real, len(val_files))

    real_signals: list[np.ndarray] = []
    idx = 0
    while len(real_signals) < n_real and idx < len(val_files):
        try:
            sig = load_ecg_signal(val_files[idx], lead_index=lead_index)
            real_signals.append(sig)
        except Exception:
            pass
        idx += 1

    # Trim all signals to common length (real may be shorter than synthetic)
    min_len = min(
        min(s.size for s in real_signals) if real_signals else synth.shape[1],
        synth.shape[1],
    )
    real_signals = [s[:min_len] for s in real_signals]
    synth = synth[:, :min_len]
    plot_len = min_len
    n_real = len(real_signals)
    n_rows = max(n_real, n_synth)

    # --- Plot ---
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(16, 1.8 * n_rows + 1.2),
        sharex=True,
        constrained_layout=True,
    )
    print(f"Plotting {n_real} real + {n_synth} synthetic, length={plot_len}", file=sys.stderr)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column titles
    axes[0, 0].set_title("REAL  (PTB-XL validation)", fontsize=13, fontweight="bold",
                          color="#1a659e", loc="center")
    axes[0, 1].set_title("SYNTHETIC  (RVQ prior, T=0.9)", fontsize=13, fontweight="bold",
                          color="#d1495b", loc="center")

    t = np.arange(plot_len) / sr

    for i in range(n_rows):
        # Real
        ax_r = axes[i, 0]
        if i < n_real:
            sig_r = _norm(real_signals[i])
            ax_r.plot(t, sig_r, lw=0.7, color="#1a659e")
            ax_r.set_ylabel(f"R{i}", fontsize=8, rotation=0, labelpad=16)
        else:
            ax_r.set_visible(False)
        ax_r.margins(x=0)
        ax_r.tick_params(labelsize=7)

        # Synthetic
        ax_s = axes[i, 1]
        if i < n_synth:
            sig_s = _norm(synth[i])
            ax_s.plot(t, sig_s, lw=0.7, color="#d1495b")
            ax_s.set_ylabel(f"S{i}", fontsize=8, rotation=0, labelpad=16)
        else:
            ax_s.set_visible(False)
        ax_s.margins(x=0)
        ax_s.tick_params(labelsize=7)

    axes[-1, 0].set_xlabel("time [s]", fontsize=9)
    axes[-1, 1].set_xlabel("time [s]", fontsize=9)

    fig.suptitle(
        "Real vs Synthetic ECG — RVQ Token Prior (108K params, 50 epochs, 17K files)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot → {out_path}", file=sys.stderr)
    plt.close(fig)

    # --- Also make a zoomed 2-second comparison ---
    zoom_len = int(2.0 * sr)
    if plot_len >= zoom_len:
        fig2, axes2 = plt.subplots(
            n_rows, 2,
            figsize=(12, 1.8 * n_rows + 1.2),
            sharex=True,
            constrained_layout=True,
        )
        if n_rows == 1:
            axes2 = axes2[np.newaxis, :]

        axes2[0, 0].set_title("REAL  (2s zoom)", fontsize=13, fontweight="bold",
                               color="#1a659e")
        axes2[0, 1].set_title("SYNTHETIC  (2s zoom)", fontsize=13, fontweight="bold",
                               color="#d1495b")

        t_z = np.arange(zoom_len) / sr
        for i in range(n_rows):
            ax_r = axes2[i, 0]
            if i < n_real:
                ax_r.plot(t_z, _norm(real_signals[i][:zoom_len]), lw=0.9, color="#1a659e")
                ax_r.set_ylabel(f"R{i}", fontsize=8, rotation=0, labelpad=16)
            else:
                ax_r.set_visible(False)
            ax_r.margins(x=0)
            ax_r.tick_params(labelsize=7)

            ax_s = axes2[i, 1]
            if i < n_synth:
                ax_s.plot(t_z, _norm(synth[i][:zoom_len]), lw=0.9, color="#d1495b")
                ax_s.set_ylabel(f"S{i}", fontsize=8, rotation=0, labelpad=16)
            else:
                ax_s.set_visible(False)
            ax_s.margins(x=0)
            ax_s.tick_params(labelsize=7)

        axes2[-1, 0].set_xlabel("time [s]", fontsize=9)
        axes2[-1, 1].set_xlabel("time [s]", fontsize=9)
        fig2.suptitle("Real vs Synthetic ECG — 2-second zoom", fontsize=14,
                       fontweight="bold", y=1.01)

        zoom_path = out_path.with_stem(out_path.stem + "_zoom")
        fig2.savefig(zoom_path, dpi=150, bbox_inches="tight")
        print(f"Saved zoom plot → {zoom_path}", file=sys.stderr)
        plt.close(fig2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
