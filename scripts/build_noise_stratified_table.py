"""Build a noise-stratified comparison table across golden codec runs.

Reads ``quality_scorecard.json`` from each golden run directory and produces
a flat table with one row per (CR, noise-tertile) combination, making it
easy to see how time-domain, spectral, and physiology metrics vary with
both compression ratio and input noise level.

Usage:
    python scripts/build_noise_stratified_table.py --modality ecg
    python scripts/build_noise_stratified_table.py --modality ppg
    python scripts/build_noise_stratified_table.py --modality ecg --output results/ecg_noise_table
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ECG_GOLDEN_RUNS = [f"ecg_rvq_256hz_{cr}_golden" for cr in ("02x", "04x", "08x", "16x", "32x", "64x")]
PPG_GOLDEN_RUNS = [f"ppg_rvq_64hz_{cr}_golden" for cr in ("02x", "04x", "08x", "16x", "32x")]
PPG_H5_GOLDEN_RUNS = [
    "ppg_h5_rvq_2x_mixed_golden_sched",
    "ppg_h5_rvq_4x_mixed_golden_sched",
    "ppg_h5_rvq_8x_mixed_golden_sched",
    "ppg_h5_rvq_16x_mixed_golden_sched_rep2",
]

TERTILE_ORDER = ["clean", "median", "noisy"]


def _safe_get(d: dict, *keys: str, default: float | None = None) -> float | None:
    """Nested dict lookup, returns default if any key is missing."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)  # type: ignore[assignment]
        if d is None:
            return default
    return d  # type: ignore[return-value]


def _extract_rows(
    scorecard: dict,
    run_name: str,
) -> list[dict]:
    """Extract per-tertile rows from a single scorecard."""
    bnt = scorecard.get("by_noise_tertile", {})
    if not bnt:
        return []

    cr_raw = scorecard.get("bitrate", {}).get("codec_compression_ratio")
    cr_label = run_name.split("_")
    # Try to find the CR portion (e.g. "02x", "2x", "4x", etc.)
    cr_str = ""
    for part in cr_label:
        if part.endswith("x") and part[:-1].replace("0", "").isdigit():
            cr_str = part
            break
    if not cr_str and cr_raw is not None:
        cr_str = f"{int(cr_raw)}x"

    thresholds = bnt.get("thresholds_bp_noise_rms", {})
    buckets = bnt.get("buckets", {})

    # Also extract physiology tertile data
    phys_bnt = scorecard.get("physiology", {}).get("by_noise_tertile", {})
    phys_buckets = phys_bnt.get("buckets", {})

    rows = []
    for tertile in TERTILE_ORDER:
        b = buckets.get(tertile, {})
        td = b.get("time_domain", {})
        sp = b.get("spectral", {})
        pb = phys_buckets.get(tertile, {})

        row = {
            "run_name": run_name,
            "cr_label": cr_str,
            "compression_ratio": cr_raw,
            "noise_tertile": tertile,
            "n_samples": b.get("n", 0),
            "noise_threshold_clean_max": thresholds.get("clean_max"),
            "noise_threshold_median_max": thresholds.get("median_max"),
            # Time-domain
            "prd_mean": _safe_get(td, "prd_percent", "mean"),
            "prd_p90": _safe_get(td, "prd_percent", "p90"),
            "prdn_mean": _safe_get(td, "prdn_noise_percent", "mean"),
            "prdn_p90": _safe_get(td, "prdn_noise_percent", "p90"),
            "rmse_mean": _safe_get(td, "rmse", "mean"),
            "cosine_mean": _safe_get(td, "cosine_similarity", "mean"),
            # Spectral
            "band_err_mean": _safe_get(sp, "band_total_rel_error", "mean"),
            "wfprd_mean": _safe_get(sp, "weighted_freq_prd_percent", "mean"),
            "coherence_mean": _safe_get(sp, "coherence", "mean"),
            # Physiology (from physiology.by_noise_tertile)
            "hr_mae_mean": _safe_get(pb, "hr_mae_bpm", "mean"),
            "hr_mae_p90": _safe_get(pb, "hr_mae_bpm", "p90"),
            "sdnn_mae_mean": _safe_get(pb, "sdnn_mae_ms", "mean"),
            "rmssd_mae_mean": _safe_get(pb, "rmssd_mae_ms", "mean"),
        }
        rows.append(row)

    # Also add an "all" row from the global metrics
    td_all = scorecard.get("time_domain", {})
    sp_all = scorecard.get("spectral", {})
    phys_all = scorecard.get("physiology", {})
    # For physiology "all", use the top-level vs_raw_original or the flat dict
    phys_top = phys_all.get("vs_raw_original", phys_all)

    rows.append(
        {
            "run_name": run_name,
            "cr_label": cr_str,
            "compression_ratio": cr_raw,
            "noise_tertile": "all",
            "n_samples": scorecard.get("num_samples", 0),
            "noise_threshold_clean_max": thresholds.get("clean_max"),
            "noise_threshold_median_max": thresholds.get("median_max"),
            "prd_mean": _safe_get(td_all, "prd_percent", "mean"),
            "prd_p90": _safe_get(td_all, "prd_percent", "p90"),
            "prdn_mean": _safe_get(td_all, "prdn_noise_percent", "mean"),
            "prdn_p90": _safe_get(td_all, "prdn_noise_percent", "p90"),
            "rmse_mean": _safe_get(td_all, "rmse", "mean"),
            "cosine_mean": _safe_get(td_all, "cosine_similarity", "mean"),
            "band_err_mean": _safe_get(sp_all, "band_total_rel_error", "mean"),
            "wfprd_mean": _safe_get(sp_all, "weighted_freq_prd_percent", "mean"),
            "coherence_mean": _safe_get(sp_all, "coherence", "mean"),
            "hr_mae_mean": phys_top.get("hr_mae_bpm") if isinstance(phys_top, dict) else None,
            "hr_mae_p90": phys_top.get("hr_p90_ae_bpm") if isinstance(phys_top, dict) else None,
            "sdnn_mae_mean": phys_top.get("sdnn_mae_ms") if isinstance(phys_top, dict) else None,
            "rmssd_mae_mean": phys_top.get("rmssd_mae_ms") if isinstance(phys_top, dict) else None,
        }
    )

    return rows


def _fmt(v: float | None, prec: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{prec}f}"


def build_table(
    results_dir: Path,
    run_names: list[str],
) -> pd.DataFrame:
    """Build a noise-stratified DataFrame from multiple golden runs."""
    all_rows: list[dict] = []
    for run_name in run_names:
        sc_path = results_dir / run_name / "quality_scorecard.json"
        if not sc_path.exists():
            logger.warning("No scorecard at %s — skipping", sc_path)
            continue
        scorecard = json.loads(sc_path.read_text())
        rows = _extract_rows(scorecard, run_name)
        if not rows:
            logger.warning("No noise-tertile data in %s — skipping", sc_path)
            continue
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def print_markdown_table(df: pd.DataFrame) -> str:
    """Format the DataFrame as a compact markdown table."""
    cols = [
        ("cr_label", "CR", "s"),
        ("noise_tertile", "Tertile", "s"),
        ("n_samples", "N", "d"),
        ("prd_mean", "PRD%", ".2f"),
        ("prdn_mean", "PRDN%", ".2f"),
        ("cosine_mean", "Cos", ".4f"),
        ("wfprd_mean", "WF-PRD%", ".2f"),
        ("coherence_mean", "Coh", ".4f"),
        ("hr_mae_mean", "HR MAE", ".2f"),
        ("sdnn_mae_mean", "SDNN MAE", ".2f"),
    ]

    header = "| " + " | ".join(label for _, label, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]

    for _, row in df.iterrows():
        cells = []
        for col, _, fmt in cols:
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("—")
            elif fmt == "s":
                cells.append(str(v))
            elif fmt == "d":
                cells.append(str(int(v)))
            else:
                cells.append(f"{v:{fmt}}")
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    print(table)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=("ecg", "ppg"), required=True)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output prefix (writes .csv and .md). Default: results/<modality>_noise_stratified",
    )
    ap.add_argument(
        "--include-h5", action="store_true", help="Include PPG H5 mixed golden runs alongside standard PPG goldens."
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.modality == "ecg":
        run_names = ECG_GOLDEN_RUNS
    else:
        run_names = list(PPG_GOLDEN_RUNS)
        if args.include_h5:
            run_names.extend(PPG_H5_GOLDEN_RUNS)

    df = build_table(args.results_dir, run_names)
    if df.empty:
        logger.error("No data collected — check that scorecard JSONs exist.")
        return

    output = args.output or args.results_dir / f"{args.modality}_noise_stratified"
    csv_path = output.with_suffix(".csv")
    md_path = output.with_suffix(".md")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info("Wrote %d rows to %s", len(df), csv_path)

    md_table = print_markdown_table(df)
    md_path.write_text(f"# {args.modality.upper()} Noise-Stratified Scorecard\n\n{md_table}\n")
    logger.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
