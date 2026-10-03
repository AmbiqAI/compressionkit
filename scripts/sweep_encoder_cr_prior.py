"""B3 sweep: encoder-CR × prior-size effective compression sweep (issue #1).

For each (encoder run_dir, prior preset) pair, drive
``scripts/measure_rvq_entropy.py`` to fit a small entropy prior on the
already-frozen RVQ token stream and gather:

* ``val_bits_per_token``, ``cr_uplift_vs_uniform`` (from
  ``entropy_report.json``)
* signal-quality scorecard metrics (from each encoder run's
  ``quality_scorecard.json`` written by the noise-aware scorecard
  builder in :mod:`compressionkit.evaluation.scorecard`)

Outputs:

* ``<out_dir>/sweep_summary.json`` — machine-readable per-combination
  rows.
* ``<out_dir>/sweep_summary.md`` — customer-ready table for inclusion
  in docs (the input artefact #2 will polish further).

Example::

    uv run python scripts/sweep_encoder_cr_prior.py \\
        --runs results/ecg_rvq_256hz_08x_golden \\
               results/ecg_rvq_256hz_16x_golden \\
               results/ecg_rvq_256hz_32x_golden \\
        --presets cnn_s cnn_xl \\
        --out-dir results/b3_sweep_ecg

The script is intentionally idempotent: it skips a (run, preset) cell
when ``entropy_prior/<preset>/entropy_report.json`` already exists,
unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTROPY_SCRIPT = REPO_ROOT / "scripts" / "measure_rvq_entropy.py"

logger = logging.getLogger("sweep-encoder-cr-prior")


# ---------------------------------------------------------------------------
# Prior presets — two well-separated CNN sizes anchor the B3 sweep.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorPreset:
    """Hyperparameters for a single entropy-prior point."""

    name: str
    prior_type: str
    cli_args: tuple[str, ...]


PRESETS: dict[str, PriorPreset] = {
    "cnn_s": PriorPreset(
        name="cnn_s",
        prior_type="cnn",
        cli_args=("--cnn-embed-dim", "32", "--cnn-num-layers", "3", "--cnn-kernel", "5"),
    ),
    "cnn_xl": PriorPreset(
        name="cnn_xl",
        prior_type="cnn",
        cli_args=("--cnn-embed-dim", "96", "--cnn-num-layers", "6", "--cnn-kernel", "7"),
    ),
    "unigram": PriorPreset(name="unigram", prior_type="unigram", cli_args=()),
}


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------


def _load_entropy_report(run_dir: Path, preset: str) -> dict | None:
    p = run_dir / "entropy_prior" / preset / "entropy_report.json"
    return json.loads(p.read_text()) if p.exists() else None


def _load_scorecard(run_dir: Path) -> dict | None:
    p = run_dir / "quality_scorecard.json"
    return json.loads(p.read_text()) if p.exists() else None


def _row_from_artifacts(
    run_dir: Path,
    preset: str,
    entropy: dict,
    scorecard: dict | None,
) -> dict:
    """Build one summary row from the two source JSON files.

    The scorecard schema is intentionally loose: we walk a handful of
    common locations and report ``None`` when a metric is unavailable so
    the row is still serialisable.
    """
    metrics = entropy.get("metrics", {})
    codec = entropy.get("codec", {})
    bpt = metrics.get("val_bits_per_token")
    cr_codec = codec.get("cr_codec_uniform") or codec.get("encoder_cr")
    cr_uplift = metrics.get("cr_uplift_vs_uniform")
    effective_cr = (cr_codec * cr_uplift) if (cr_codec and cr_uplift) else None

    quality: dict[str, float | None] = {
        "prd_percent": None,
        "prdn_noise_percent": None,
        "hr_mae_bpm": None,
        "spectral_coherence": None,
    }
    if scorecard is not None:
        # Scorecards may store under "overall" or top-level depending on modality.
        overall = scorecard.get("overall", scorecard)
        quality["prd_percent"] = overall.get("prd_percent") or overall.get("mean_prd_percent")
        quality["prdn_noise_percent"] = (
            overall.get("prdn_noise_percent") or overall.get("prdn_percent") or overall.get("mean_prdn_percent")
        )
        # HR MAE — prefer long-recording when available, fall back to per-window.
        hr_block = scorecard.get("long_recording") or scorecard.get("ecg_physiology") or {}
        quality["hr_mae_bpm"] = hr_block.get("hr_mae_bpm")
        quality["spectral_coherence"] = overall.get("spectral_coherence") or overall.get("mean_spectral_coherence")

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "preset": preset,
        "prior_type": entropy.get("prior_type"),
        "encoder_cr": cr_codec,
        "bits_per_token": bpt,
        "cr_uplift": cr_uplift,
        "effective_total_cr": effective_cr,
        **quality,
    }


def build_summary(rows: list[dict]) -> dict:
    """Aggregate per-cell rows into the final sweep summary payload."""
    finite = [r for r in rows if r.get("effective_total_cr") is not None]
    recommendation: dict | None = None
    if finite:
        # Pick the highest effective CR whose PRD is within +25% of the best
        # PRD across the sweep — a simple heuristic that #2 will refine.
        best_prd_row = min(
            (r for r in finite if r.get("prd_percent") is not None),
            key=lambda r: r["prd_percent"],
            default=None,
        )
        if best_prd_row is not None:
            prd_ceiling = best_prd_row["prd_percent"] * 1.25
            eligible = [r for r in finite if r.get("prd_percent") is not None and r["prd_percent"] <= prd_ceiling]
            if eligible:
                pick = max(eligible, key=lambda r: r["effective_total_cr"])
                recommendation = {
                    "criterion": "max effective_total_cr s.t. PRD ≤ 1.25 × min(PRD)",
                    "prd_ceiling": prd_ceiling,
                    "winner": {"run_name": pick["run_name"], "preset": pick["preset"]},
                    "effective_total_cr": pick["effective_total_cr"],
                    "prd_percent": pick["prd_percent"],
                }
    return {"rows": rows, "recommendation": recommendation}


def render_markdown(summary: dict) -> str:
    """Render the customer-facing comparison table."""
    rows = summary["rows"]
    lines = [
        "| Run | Preset | Encoder CR | bits/token | CR uplift | Effective total CR | PRD (%) | PRDN-noise (%) | HR MAE (bpm) | Spectral coherence |",
        "|-----|--------|-----------|------------|-----------|--------------------|---------|----------------|--------------|--------------------|",
    ]

    def _fmt(v, spec=".3f") -> str:
        return f"{v:{spec}}" if isinstance(v, int | float) else "—"

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["run_name"],
                    r["preset"],
                    _fmt(r.get("encoder_cr"), ".2f"),
                    _fmt(r.get("bits_per_token"), ".3f"),
                    _fmt(r.get("cr_uplift"), ".2f"),
                    _fmt(r.get("effective_total_cr"), ".2f"),
                    _fmt(r.get("prd_percent"), ".2f"),
                    _fmt(r.get("prdn_noise_percent"), ".2f"),
                    _fmt(r.get("hr_mae_bpm"), ".2f"),
                    _fmt(r.get("spectral_coherence"), ".3f"),
                ]
            )
            + " |"
        )
    rec = summary.get("recommendation")
    if rec:
        lines.append("")
        lines.append(
            f"**Recommendation**: `{rec['winner']['run_name']}` × `{rec['winner']['preset']}` — "
            f"effective CR = {rec['effective_total_cr']:.2f}, PRD = {rec['prd_percent']:.2f}%. "
            f"({rec['criterion']}.)"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run_one(run_dir: Path, preset: PriorPreset, *, force: bool, extra_args: list[str]) -> None:
    out_dir = run_dir / "entropy_prior" / preset.name
    report = out_dir / "entropy_report.json"
    if report.exists() and not force:
        logger.info("Skipping %s × %s — report already at %s", run_dir.name, preset.name, report)
        return
    cmd = [
        sys.executable,
        str(ENTROPY_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--prior-type",
        preset.prior_type,
        "--tag",
        preset.name,
        *preset.cli_args,
        *extra_args,
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Encoder run dirs.")
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["cnn_s", "cnn_xl"],
        choices=list(PRESETS),
        help="Prior presets to sweep over.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Re-train priors even if a report exists.")
    parser.add_argument("--skip-train", action="store_true", help="Only collect existing reports.")
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to measure_rvq_entropy.py after `--`.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    extra: list[str] = list(args.extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]

    rows: list[dict] = []
    for run_dir in args.runs:
        run_dir = run_dir.resolve()
        for preset_name in args.presets:
            preset = PRESETS[preset_name]
            if not args.skip_train:
                _run_one(run_dir, preset, force=args.force, extra_args=extra)
            entropy = _load_entropy_report(run_dir, preset.name)
            if entropy is None:
                logger.warning("No entropy report for %s × %s — skipping row.", run_dir.name, preset.name)
                continue
            scorecard = _load_scorecard(run_dir)
            rows.append(_row_from_artifacts(run_dir, preset.name, entropy, scorecard))

    summary = build_summary(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "sweep_summary.md").write_text(render_markdown(summary))
    logger.info("Wrote %d rows to %s/sweep_summary.{json,md}", len(rows), args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
