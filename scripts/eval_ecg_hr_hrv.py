"""Evaluate ECG compression models on HR/HRV preservation using physiokit.

Loads saved sample CSVs from experiment results directories and compares
R-peak detection + HR/HRV metrics between original and reconstructed signals.

Usage:
    uv run python scripts/eval_ecg_hr_hrv.py <results_dir> [results_dir2 ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from compressionkit.evaluation.metrics import compute_ecg_hr_hrv

SAMPLE_RATE = 256
MIN_PEAKS_FOR_HR = 2
MIN_PEAKS_FOR_HRV = 3


def evaluate_results_dir(results_dir: Path) -> dict:
    """Evaluate all sample CSVs in a results directory."""
    csv_files = sorted(results_dir.glob("sample_*.csv"))
    if not csv_files:
        print(f"  No sample CSVs found in {results_dir}")
        return {}

    hr_errors: list[float] = []
    hr_abs_errors: list[float] = []
    peak_count_diffs: list[int] = []
    peak_timing_errors: list[float] = []
    sdnn_errors: list[float] = []
    rmssd_errors: list[float] = []
    valid_hr = 0
    valid_hrv = 0
    total = 0
    missed_peaks = 0
    extra_peaks = 0

    for csv_path in csv_files:
        total += 1
        df = pd.read_csv(csv_path)
        orig = df["original"].values.astype(np.float32)
        recon = df["reconstructed"].values.astype(np.float32)

        orig_m = compute_ecg_hr_hrv(orig, sample_rate=SAMPLE_RATE)
        recon_m = compute_ecg_hr_hrv(recon, sample_rate=SAMPLE_RATE)

        if orig_m is None or recon_m is None:
            continue

        # Peak count comparison
        count_diff = recon_m["num_peaks"] - orig_m["num_peaks"]
        peak_count_diffs.append(count_diff)
        if count_diff < 0:
            missed_peaks += abs(count_diff)
        elif count_diff > 0:
            extra_peaks += count_diff

        # Peak timing error (match closest peaks)
        orig_locs = np.array(orig_m["peak_locations"])
        recon_locs = np.array(recon_m["peak_locations"])
        for op in orig_locs:
            if recon_locs.size > 0:
                closest_idx = np.argmin(np.abs(recon_locs - op))
                timing_err_ms = abs(recon_locs[closest_idx] - op) / SAMPLE_RATE * 1000
                peak_timing_errors.append(timing_err_ms)

        # HR comparison
        hr_diff = recon_m["hr_bpm"] - orig_m["hr_bpm"]
        hr_errors.append(hr_diff)
        hr_abs_errors.append(abs(hr_diff))
        valid_hr += 1

        # HRV comparison (only if both have enough peaks)
        if "sdnn_ms" in orig_m and "sdnn_ms" in recon_m:
            sdnn_errors.append(abs(recon_m["sdnn_ms"] - orig_m["sdnn_ms"]))
            rmssd_errors.append(abs(recon_m["rmssd_ms"] - orig_m["rmssd_ms"]))
            valid_hrv += 1

    summary = {
        "total_samples": total,
        "valid_hr_pairs": valid_hr,
        "valid_hrv_pairs": valid_hrv,
    }

    if valid_hr > 0:
        summary["hr_mae_bpm"] = float(np.mean(hr_abs_errors))
        summary["hr_bias_bpm"] = float(np.mean(hr_errors))
        summary["hr_max_err_bpm"] = float(np.max(hr_abs_errors))
        summary["hr_median_err_bpm"] = float(np.median(hr_abs_errors))

    if peak_count_diffs:
        summary["peak_count_exact_match"] = sum(1 for d in peak_count_diffs if d == 0)
        summary["peak_count_exact_match_pct"] = summary["peak_count_exact_match"] / len(peak_count_diffs) * 100
        summary["total_missed_peaks"] = missed_peaks
        summary["total_extra_peaks"] = extra_peaks

    if peak_timing_errors:
        summary["peak_timing_mae_ms"] = float(np.mean(peak_timing_errors))
        summary["peak_timing_median_ms"] = float(np.median(peak_timing_errors))
        summary["peak_timing_max_ms"] = float(np.max(peak_timing_errors))
        summary["peak_timing_within_10ms_pct"] = sum(1 for t in peak_timing_errors if t <= 10) / len(peak_timing_errors) * 100

    if valid_hrv > 0:
        summary["sdnn_mae_ms"] = float(np.mean(sdnn_errors))
        summary["rmssd_mae_ms"] = float(np.mean(rmssd_errors))

    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/eval_ecg_hr_hrv.py <results_dir> [...]")
        sys.exit(1)

    results = {}
    for path_str in sys.argv[1:]:
        results_dir = Path(path_str)
        name = results_dir.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        summary = evaluate_results_dir(results_dir)
        results[name] = summary

        if not summary:
            continue

        print(f"  Samples: {summary['total_samples']} total, {summary['valid_hr_pairs']} valid HR, {summary['valid_hrv_pairs']} valid HRV")
        if "hr_mae_bpm" in summary:
            print(f"  HR:  MAE={summary['hr_mae_bpm']:.2f} BPM, bias={summary['hr_bias_bpm']:.2f}, max={summary['hr_max_err_bpm']:.2f}, median={summary['hr_median_err_bpm']:.2f}")
        if "peak_count_exact_match_pct" in summary:
            print(f"  Peaks: {summary['peak_count_exact_match_pct']:.0f}% exact match, {summary['total_missed_peaks']} missed, {summary['total_extra_peaks']} extra")
        if "peak_timing_mae_ms" in summary:
            print(f"  Peak timing: MAE={summary['peak_timing_mae_ms']:.2f}ms, median={summary['peak_timing_median_ms']:.2f}ms, max={summary['peak_timing_max_ms']:.2f}ms, ≤10ms={summary['peak_timing_within_10ms_pct']:.0f}%")
        if "sdnn_mae_ms" in summary:
            print(f"  HRV: SDNN MAE={summary['sdnn_mae_ms']:.2f}ms, RMSSD MAE={summary['rmssd_mae_ms']:.2f}ms")

    # Comparison table
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON TABLE")
        print(f"{'='*60}")
        header = f"{'Model':<45} {'HR MAE':>8} {'Pk Match':>9} {'Pk Time':>8} {'≤10ms':>6}"
        print(header)
        print("-" * len(header))
        for name, s in results.items():
            hr = f"{s.get('hr_mae_bpm', float('nan')):.2f}" if s else "N/A"
            pk_match = f"{s.get('peak_count_exact_match_pct', float('nan')):.0f}%" if s else "N/A"
            pk_time = f"{s.get('peak_timing_mae_ms', float('nan')):.2f}" if s else "N/A"
            within10 = f"{s.get('peak_timing_within_10ms_pct', float('nan')):.0f}%" if s else "N/A"
            print(f"{name:<45} {hr:>8} {pk_match:>9} {pk_time:>8} {within10:>6}")


if __name__ == "__main__":
    main()
