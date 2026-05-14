"""12-lead Holter reporting: per-lead bpt, total bits/frame, effective CR.

Reads per-lead entropy report JSONs (one per lead from measure_rvq_entropy.py)
and/or a cross-channel prior entropy report, then produces a summary table.

Usage:
    python scripts/report_12lead_holter.py \
        --run-dir results/ecg_rvq_256hz_32x_golden \
        --xlead-tag xlead_concat_v1 \
        --independent-tag cnn_xl \
        --output results/12lead_holter_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def compute_per_lead_stats(
    run_dir: Path,
    tag: str,
    *,
    num_leads: int = 12,
    tokens_per_frame: int | None = None,
    vocab_size: int = 256,
) -> dict[str, Any]:
    """Load per-lead entropy reports and compute aggregate stats.

    Expects reports at ``run_dir/entropy_prior/<tag>/entropy_report.json``
    with a ``metrics.val_bits_per_token`` field. For independent per-lead
    priors, one report is shared across all leads (same encoder applied
    independently — the bpt is the same for each lead).
    """
    report_path = run_dir / "entropy_prior" / tag / "entropy_report.json"
    report = _load_report(report_path)
    if report is None:
        return {"error": f"Report not found: {report_path}"}

    bpt = report.get("metrics", {}).get("val_bits_per_token")
    if bpt is None:
        return {"error": "val_bits_per_token not found in report"}

    uniform_bpt = math.log2(vocab_size)
    tpf = tokens_per_frame or report.get("codec", {}).get("tokens_per_frame", 1)
    codec_cr = report.get("codec", {}).get("cr_codec_uniform", 1.0)

    bits_per_frame_per_lead = bpt * tpf
    total_bits_per_frame = bits_per_frame_per_lead * num_leads
    uniform_bits_per_frame = uniform_bpt * tpf * num_leads

    # Effective CR for the full 12-lead bundle
    cr_uplift = uniform_bpt / bpt if bpt > 0 else 1.0
    effective_cr = codec_cr * cr_uplift

    return {
        "tag": tag,
        "strategy": "per_lead_independent",
        "num_leads": num_leads,
        "bpt_per_lead": bpt,
        "uniform_bpt": uniform_bpt,
        "tokens_per_frame": tpf,
        "bits_per_frame_per_lead": bits_per_frame_per_lead,
        "total_bits_per_frame_12lead": total_bits_per_frame,
        "uniform_bits_per_frame_12lead": uniform_bits_per_frame,
        "cr_codec_uniform": codec_cr,
        "cr_uplift_vs_uniform": cr_uplift,
        "effective_cr_12lead": effective_cr,
    }


def compute_xlead_stats(
    run_dir: Path,
    tag: str,
    *,
    num_leads: int = 12,
    tokens_per_frame: int | None = None,
    vocab_size: int = 256,
) -> dict[str, Any]:
    """Load cross-channel prior entropy report and compute stats."""
    report_path = run_dir / "entropy_prior" / tag / "entropy_report.json"
    report = _load_report(report_path)
    if report is None:
        return {"error": f"Report not found: {report_path}"}

    bpt = report.get("metrics", {}).get("val_bits_per_token")
    if bpt is None:
        return {"error": "val_bits_per_token not found in report"}

    prior_meta = report.get("prior", {})
    strategy = prior_meta.get("type", "xlead_unknown")
    uniform_bpt = math.log2(vocab_size)
    tpf = tokens_per_frame or report.get("codec", {}).get("tokens_per_frame", 1)
    codec_cr = report.get("codec", {}).get("cr_codec_uniform", 1.0)

    # For xlead priors, bpt already accounts for cross-lead correlation
    # Total tokens per frame across all leads
    total_tokens_per_frame = tpf * num_leads
    total_bits_per_frame = bpt * total_tokens_per_frame
    uniform_bits_per_frame = uniform_bpt * total_tokens_per_frame

    cr_uplift = uniform_bpt / bpt if bpt > 0 else 1.0
    effective_cr = codec_cr * cr_uplift

    return {
        "tag": tag,
        "strategy": strategy,
        "num_leads": num_leads,
        "bpt_cross_channel": bpt,
        "uniform_bpt": uniform_bpt,
        "tokens_per_frame": tpf,
        "total_tokens_per_frame_12lead": total_tokens_per_frame,
        "total_bits_per_frame_12lead": total_bits_per_frame,
        "uniform_bits_per_frame_12lead": uniform_bits_per_frame,
        "cr_codec_uniform": codec_cr,
        "cr_uplift_vs_uniform": cr_uplift,
        "effective_cr_12lead": effective_cr,
    }


def build_comparison(
    independent_stats: dict[str, Any],
    xlead_stats: dict[str, Any],
) -> dict[str, Any]:
    """Compare independent per-lead vs cross-channel prior."""
    if "error" in independent_stats or "error" in xlead_stats:
        return {
            "independent": independent_stats,
            "xlead": xlead_stats,
            "comparison": None,
        }

    ind_total = independent_stats["total_bits_per_frame_12lead"]
    xl_total = xlead_stats["total_bits_per_frame_12lead"]
    bpt_reduction_pct = (1.0 - xl_total / ind_total) * 100 if ind_total > 0 else 0.0

    return {
        "independent": independent_stats,
        "xlead": xlead_stats,
        "comparison": {
            "bpt_reduction_percent": bpt_reduction_pct,
            "bits_saved_per_frame": ind_total - xl_total,
            "independent_effective_cr": independent_stats["effective_cr_12lead"],
            "xlead_effective_cr": xlead_stats["effective_cr_12lead"],
            "meets_10pct_threshold": bpt_reduction_pct >= 10.0,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render comparison report as a markdown table."""
    lines = ["# 12-Lead Holter Prior Comparison", ""]

    ind = report.get("independent", {})
    xl = report.get("xlead", {})
    comp = report.get("comparison")

    lines.append("| Metric | Per-lead Independent | Cross-channel |")
    lines.append("|--------|---------------------|---------------|")

    def _row(label: str, v1: Any, v2: Any, fmt: str = ".2f") -> str:
        s1 = f"{v1:{fmt}}" if isinstance(v1, (int, float)) else str(v1)
        s2 = f"{v2:{fmt}}" if isinstance(v2, (int, float)) else str(v2)
        return f"| {label} | {s1} | {s2} |"

    if "error" not in ind and "error" not in xl:
        lines.append(_row("Strategy", ind.get("strategy", "?"), xl.get("strategy", "?"), "s"))
        lines.append(_row("bits/token", ind.get("bpt_per_lead", 0), xl.get("bpt_cross_channel", 0)))
        lines.append(
            _row("Total bits/frame (12-lead)", ind["total_bits_per_frame_12lead"], xl["total_bits_per_frame_12lead"])
        )
        lines.append(_row("Effective CR (12-lead)", ind["effective_cr_12lead"], xl["effective_cr_12lead"]))
        lines.append(_row("CR uplift vs uniform", ind["cr_uplift_vs_uniform"], xl["cr_uplift_vs_uniform"]))

    if comp:
        lines.append("")
        lines.append(f"**bpt reduction**: {comp['bpt_reduction_percent']:.1f}%")
        lines.append(f"**Meets ≥10% threshold**: {'Yes' if comp['meets_10pct_threshold'] else 'No'}")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--independent-tag", type=str, default="cnn_xl")
    ap.add_argument("--xlead-tag", type=str, default=None)
    ap.add_argument("--num-leads", type=int, default=12)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    ind_stats = compute_per_lead_stats(
        args.run_dir, args.independent_tag, num_leads=args.num_leads, vocab_size=args.vocab_size
    )

    xl_stats: dict[str, Any] = {"error": "No xlead tag specified"}
    if args.xlead_tag:
        xl_stats = compute_xlead_stats(
            args.run_dir, args.xlead_tag, num_leads=args.num_leads, vocab_size=args.vocab_size
        )

    report = build_comparison(ind_stats, xl_stats)
    report["run_dir"] = str(args.run_dir)

    out = args.output or args.run_dir / "12lead_holter_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    logger.info("Wrote %s", out)

    md = render_markdown(report)
    md_path = out.with_suffix(".md")
    md_path.write_text(md)
    logger.info("Wrote %s", md_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
