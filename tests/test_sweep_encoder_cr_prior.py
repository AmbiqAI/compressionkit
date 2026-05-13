"""Tests for the B3 encoder-CR × prior-size sweep driver (issue #1)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "sweep_encoder_cr_prior",
    Path(__file__).resolve().parent.parent / "scripts" / "sweep_encoder_cr_prior.py",
)
assert SPEC is not None and SPEC.loader is not None
sweep = importlib.util.module_from_spec(SPEC)
sys.modules["sweep_encoder_cr_prior"] = sweep
SPEC.loader.exec_module(sweep)


def _write_entropy(run_dir: Path, preset: str, *, bpt: float, uplift: float, codec_cr: float) -> None:
    p = run_dir / "entropy_prior" / preset
    p.mkdir(parents=True, exist_ok=True)
    (p / "entropy_report.json").write_text(
        json.dumps(
            {
                "prior_type": "cnn",
                "codec": {"cr_codec_uniform": codec_cr},
                "metrics": {
                    "val_bits_per_token": bpt,
                    "cr_uplift_vs_uniform": uplift,
                },
            }
        )
    )


def _write_scorecard(run_dir: Path, prd: float, hr_mae: float) -> None:
    (run_dir / "quality_scorecard.json").write_text(
        json.dumps(
            {
                "overall": {
                    "prd_percent": prd,
                    "prdn_noise_percent": prd * 0.5,
                    "spectral_coherence": 0.95,
                },
                "long_recording": {"hr_mae_bpm": hr_mae},
            }
        )
    )


def test_row_and_summary_recommendation(tmp_path: Path):
    runs = []
    for cr, prd, hr in [(8.0, 5.0, 0.5), (16.0, 8.0, 0.7), (32.0, 12.0, 1.5)]:
        run = tmp_path / f"ecg_{int(cr):02d}x"
        run.mkdir()
        _write_scorecard(run, prd=prd, hr_mae=hr)
        _write_entropy(run, "cnn_s", bpt=4.0, uplift=2.0, codec_cr=cr)
        _write_entropy(run, "cnn_xl", bpt=3.0, uplift=2.7, codec_cr=cr)
        runs.append(run)

    rows = []
    for run in runs:
        for preset in ("cnn_s", "cnn_xl"):
            entropy = sweep._load_entropy_report(run, preset)
            sc = sweep._load_scorecard(run)
            rows.append(sweep._row_from_artifacts(run, preset, entropy, sc))

    assert len(rows) == 6
    first = rows[0]
    assert first["encoder_cr"] == 8.0
    assert first["effective_total_cr"] == 16.0
    assert first["prd_percent"] == 5.0
    assert first["hr_mae_bpm"] == 0.5

    summary = sweep.build_summary(rows)
    rec = summary["recommendation"]
    # PRD ceiling = 1.25 × 5 = 6.25; only 8x cells qualify.
    # Max effective CR among them is cnn_xl × 8x = 21.6.
    assert rec is not None
    assert rec["winner"]["run_name"] == "ecg_08x"
    assert rec["winner"]["preset"] == "cnn_xl"
    assert abs(rec["effective_total_cr"] - 21.6) < 1e-6


def test_render_markdown_columns(tmp_path: Path):
    run = tmp_path / "ecg_16x"
    run.mkdir()
    _write_scorecard(run, prd=7.0, hr_mae=0.8)
    _write_entropy(run, "cnn_s", bpt=4.0, uplift=2.0, codec_cr=16.0)
    rows = [
        sweep._row_from_artifacts(run, "cnn_s", sweep._load_entropy_report(run, "cnn_s"), sweep._load_scorecard(run))
    ]
    md = sweep.render_markdown(sweep.build_summary(rows))
    assert "Effective total CR" in md
    assert "ecg_16x" in md
    assert "32.00" in md  # 16 × 2.0
