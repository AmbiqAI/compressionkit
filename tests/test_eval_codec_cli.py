"""Smoke tests for ``scripts/eval_codec.py`` CLI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "eval_codec.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("eval_codec", CLI_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_codec"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cli_spiht_ppg_smoke(tmp_path: Path) -> None:
    cli = _load_cli_module()
    args = cli.build_parser().parse_args(
        [
            "--codec", "spiht_ac",
            "--modality", "ppg",
            "--cr", "4",
            "--tiers", "fidelity", "adversarial", "stitching",
            "--n-frames", "4",
            "--out", str(tmp_path),
        ]
    )
    report = cli.run(args)
    assert report["codec"]["name"].startswith("spiht_ac_ppg")
    assert "fidelity" in report["tiers"]
    assert "adversarial" in report["tiers"]
    assert "stitching" in report["tiers"]
    # Identity-of-output check: fidelity must produce a numeric mean PRD.
    prd_mean = report["tiers"]["fidelity"]["metrics"]["prd_percent"]["mean"]
    assert prd_mean > 0.0
    # Files are written.
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()
    on_disk = json.loads((tmp_path / "report.json").read_text())
    assert on_disk["codec"]["frame_size"] == 320


def test_cli_qos_requires_rvq(tmp_path: Path) -> None:
    # QoS tier should be a no-op (with a reason) when used with a non-RVQ codec.
    cli = _load_cli_module()
    args = cli.build_parser().parse_args(
        [
            "--codec", "spiht_ac",
            "--modality", "ppg",
            "--cr", "4",
            "--tiers", "fidelity", "qos",
            "--n-frames", "4",
            "--out", str(tmp_path),
        ]
    )
    report = cli.run(args)
    assert report["tiers"]["qos"]["skipped"] is True


def test_cli_rejects_rvq_without_run(tmp_path: Path) -> None:
    cli = _load_cli_module()
    args = cli.build_parser().parse_args(
        [
            "--codec", "rvq",
            "--modality", "ppg",
            "--tiers", "fidelity",
            "--n-frames", "4",
            "--out", str(tmp_path),
        ]
    )
    with pytest.raises(SystemExit):
        cli.run(args)
