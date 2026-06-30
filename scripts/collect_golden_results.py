"""Collect golden results from all trained golden configs into a single CSV/JSON.

Usage:
    python scripts/collect_golden_results.py [--results-dir results] [--output results/golden_summary]

Outputs:
    golden_summary.csv  — flat table of all metrics
    golden_summary.json — same data as JSON (list of dicts)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Golden run name patterns
GOLDEN_PATTERNS: dict[str, list[str]] = {
    "ppg": [f"ppg_rvq_64hz_{cr}_golden" for cr in ("02x", "04x", "08x", "16x", "32x")],
    "ecg": [
        "ecg_rvq_256hz_02x_golden",
        "ecg_rvq_256hz_04x_golden",
        "ecg_rvq_256hz_08x_golden_empirical_midpoint",
        "ecg_rvq_256hz_16x_golden",
        "ecg_rvq_256hz_32x_golden",
        "ecg_rvq_256hz_64x_golden",
    ],
}

# Operating sample rates (after resampling)
# PTB-XL is 500 Hz native but we resample to 256 Hz for training
SAMPLE_RATES = {"ppg": 64, "ecg": 256}


def _parse_summary(path: Path, signal: str) -> dict | None:
    """Parse a single summary.json into a flat dict of key metrics."""
    if not path.exists():
        logger.warning("Missing: %s", path)
        return None

    with open(path) as f:
        raw = json.load(f)

    metrics = raw.get("metrics", {})
    comp = raw.get("compression") or raw.get("compression_stats", {})
    best = metrics.get("best", {})
    cfg_model = raw.get("config", {}).get("model", {})

    # Use the operating sample rate (post-resampling), not the native dataset rate
    sample_rate = SAMPLE_RATES[signal]
    ds_factor = comp.get("effective_downsample_factor", 1)
    effective_rate = sample_rate / ds_factor

    row: dict = {
        "signal": signal.upper(),
        "run_name": path.parent.name,
        "cr_label": path.parent.name.split("_")[3],  # e.g. "02x"
        "compression_ratio": comp.get("compression_ratio"),
        "frame_size": comp.get("frame_size"),
        "latent_positions": comp.get("latent_positions"),
        "bits_per_index": comp.get("bits_per_index"),
        "bits_per_frame": comp.get("compressed_bits_per_window"),
        "raw_bits_per_frame": comp.get("raw_bits_per_window"),
        "downsample_factor": ds_factor,
        "sample_rate": sample_rate,
        "effective_sample_rate": effective_rate,
        "encoder_stages": cfg_model.get("encoder_stages", cfg_model.get("num_stages")),
        "num_levels": cfg_model.get("num_levels"),
        "base_filters": cfg_model.get("base_filters"),
        "embedding_dim": cfg_model.get("embedding_dim"),
        "best_epoch": metrics.get("best_epoch"),
        # Core quality metrics (validation)
        "val_mse": best.get("val_mse"),
        "val_prd": best.get("val_prd"),
        "val_cos": best.get("val_cos"),
        "val_loss": best.get("val_loss"),
        # Training metrics
        "train_mse": best.get("mse"),
        "train_prd": best.get("prd"),
        "train_cos": best.get("cos"),
        # RVQ health
        "val_rvq_usage": best.get("val_rvq_usage_mean"),
        "val_rvq_perplexity": best.get("val_rvq_perplexity_mean"),
    }

    # Long-recording physiokit metrics (PPG only currently)
    h5_eval = raw.get("h5_eval_metrics", {})
    lr_pk = metrics.get("long_recording_physiokit") or h5_eval.get("long_recording") or {}
    if lr_pk:
        row.update(
            {
                "hr_mae_bpm": lr_pk.get("hr_mae_bpm"),
                "hr_median_ae_bpm": lr_pk.get("hr_median_ae_bpm"),
                "hr_bias_bpm": lr_pk.get("hr_bias_bpm"),
                "sdnn_mae_ms": lr_pk.get("sdnn_mae_ms"),
                "rmssd_mae_ms": lr_pk.get("rmssd_mae_ms"),
            }
        )

    # Short-window physiokit (best_physiokit)
    bp = metrics.get("best_physiokit") or h5_eval.get("short_window", {}).get("physiokit") or {}
    if bp:
        row.update(
            {
                "sw_hr_mae_bpm": bp.get("hr_mae_bpm"),
                "sw_sdnn_mae_ms": bp.get("sdnn_mae_ms"),
                "sw_rmssd_mae_ms": bp.get("rmssd_mae_ms"),
            }
        )

    return row


def collect(results_dir: Path) -> pd.DataFrame:
    """Collect all golden results into a DataFrame."""
    rows = []
    for signal, run_names in GOLDEN_PATTERNS.items():
        for run_name in run_names:
            summary_path = results_dir / run_name / "summary.json"
            row = _parse_summary(summary_path, signal)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect golden results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/golden_summary"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = collect(args.results_dir)
    if df.empty:
        logger.error("No golden results found in %s", args.results_dir)
        return

    csv_path = args.output.with_suffix(".csv")
    json_path = args.output.with_suffix(".json")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.6f")
    df.to_json(json_path, orient="records", indent=2)

    logger.info("Wrote %d rows to %s and %s", len(df), csv_path, json_path)
    print(df[["signal", "cr_label", "compression_ratio", "val_mse", "val_prd", "val_cos"]].to_string(index=False))


if __name__ == "__main__":
    main()
