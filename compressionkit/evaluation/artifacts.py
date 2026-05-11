"""Save evaluation artifacts: sample plots, CSVs, and summary JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compressionkit.evaluation.metrics import compute_signal_metrics


def save_sample_artifacts(
    sample_id: int,
    original: np.ndarray,
    reconstructed: np.ndarray,
    sampling_rate: int,
    run_dir: Path,
    band_original: np.ndarray | None = None,
    band_reconstructed: np.ndarray | None = None,
    physiokit_metrics: dict[str, Any] | None = None,
    save_plot: bool = True,
) -> dict[str, Any]:
    """Save per-sample CSV, plot, and compute metrics for one evaluation sample.

    Args:
        sample_id: Index of the evaluation sample.
        original: Original signal (1D or squeezable).
        reconstructed: Reconstructed signal (1D or squeezable).
        sampling_rate: Signal sampling rate in Hz.
        run_dir: Directory to write artifacts into.
        band_original: Optional band-filtered original for band metrics.
        band_reconstructed: Optional band-filtered reconstruction.
        physiokit_metrics: Optional physiokit metrics dict for this sample.
        save_plot: If False, skip matplotlib plot generation (CSV + metrics
            are still written). Useful when you want many samples for metrics
            but only a handful of visual artifacts.

    Returns:
        Dictionary with paths and metric values. The ``plot`` key is omitted
        when ``save_plot=False``.
    """
    orig_flat = original.reshape(-1)
    recon_flat = reconstructed.reshape(-1)

    csv_rows = [
        {"time_index": idx, "original": float(o), "reconstructed": float(r)}
        for idx, (o, r) in enumerate(zip(orig_flat, recon_flat))
    ]
    csv_path = run_dir / f"sample_{sample_id:03d}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    metrics = compute_signal_metrics(orig_flat, recon_flat)

    plot_path: Path | None = None
    if save_plot:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        ts = np.arange(len(orig_flat)) / sampling_rate
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(ts, orig_flat, label="Original")
        ax.plot(ts, recon_flat, label="Reconstruction", alpha=0.8)
        ax.set_title(f"Sample {sample_id}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = plots_dir / f"sample_{sample_id:03d}.png"
        fig.savefig(plot_path)
        plt.close(fig)

    results: dict[str, Any] = {
        "csv": str(csv_path.relative_to(run_dir)),
        "metrics": {
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "cosine_similarity": metrics["cosine_similarity"],
        },
    }
    if plot_path is not None:
        results["plot"] = str(plot_path.relative_to(run_dir))
    if band_original is not None and band_reconstructed is not None:
        band_metrics = compute_signal_metrics(band_original, band_reconstructed)
        results["band_metrics"] = {
            "mse": band_metrics["mse"],
            "mae": band_metrics["mae"],
            "cosine_similarity": band_metrics["cosine_similarity"],
            "prd_percent": band_metrics["prd_percent"],
        }
    if physiokit_metrics is not None:
        results["physiokit_metrics"] = physiokit_metrics
    return results


__all__ = ["save_sample_artifacts"]
