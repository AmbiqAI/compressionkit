"""Regression tests to verify golden baselines are intact.

These tests ensure that golden config files, weight files, and recorded metrics
remain consistent through the v1 refactoring. They do NOT require GPU or model
inference — they validate the frozen artifacts on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "goldens" / "manifest.json"


def _goldens_available() -> bool:
    """Check if golden results directories are present on disk."""
    if not MANIFEST_PATH.exists():
        return False
    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    return any((REPO_ROOT / g["results_dir"]).is_dir() for g in data["goldens"])


# Skip entire module when golden results are not available (e.g. CI).
# The results/ directory is git-ignored and only present in dev containers.
pytestmark = pytest.mark.skipif(not _goldens_available(), reason="Golden results not available")


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    return data["goldens"]


def _golden_ids(manifest_path: Path) -> list[str]:
    """Extract golden names for parametrize IDs without loading at import time."""
    if not manifest_path.exists():
        return []
    with open(manifest_path) as f:
        data = json.load(f)
    return [g["name"] for g in data["goldens"]]


GOLDEN_IDS = _golden_ids(MANIFEST_PATH)


@pytest.fixture(params=range(len(GOLDEN_IDS)), ids=GOLDEN_IDS)
def golden_entry(request: pytest.FixtureRequest, manifest: list[dict]) -> dict:
    return manifest[request.param]


class TestGoldenArtifactsExist:
    """Verify that all golden artifacts are present on disk."""

    def test_config_yaml_exists(self, golden_entry: dict) -> None:
        path = REPO_ROOT / golden_entry["config_yaml"]
        assert path.exists(), f"Missing config: {path}"

    def test_results_dir_exists(self, golden_entry: dict) -> None:
        path = REPO_ROOT / golden_entry["results_dir"]
        assert path.is_dir(), f"Missing results dir: {path}"

    def test_weights_file_exists(self, golden_entry: dict) -> None:
        path = REPO_ROOT / golden_entry["weights"]
        assert path.exists(), f"Missing weights: {path}"

    def test_encoder_decoder_keras_exist(self, golden_entry: dict) -> None:
        results = REPO_ROOT / golden_entry["results_dir"]
        assert (results / "encoder.keras").exists(), f"Missing encoder.keras in {results}"
        assert (results / "decoder.keras").exists(), f"Missing decoder.keras in {results}"

    def test_rvq_weights_exist(self, golden_entry: dict) -> None:
        results = REPO_ROOT / golden_entry["results_dir"]
        assert (results / "rvq_weights.npz").exists(), f"Missing rvq_weights.npz in {results}"


class TestGoldenMetricsConsistent:
    """Verify golden summary matches manifest metrics."""

    def test_summary_json_exists(self) -> None:
        path = REPO_ROOT / "results" / "golden_summary.json"
        assert path.exists(), "results/golden_summary.json is missing"

    def test_1ch_metrics_match_summary(self, manifest: list[dict]) -> None:
        """For 1-channel goldens, cross-check manifest vs golden_summary.json."""
        summary_path = REPO_ROOT / "results" / "golden_summary.json"
        if not summary_path.exists():
            pytest.skip("golden_summary.json not found")

        with open(summary_path) as f:
            summary_entries = json.load(f)

        summary_by_name = {e["run_name"]: e for e in summary_entries}

        for golden in manifest:
            if golden["task"] != "1ch_rvq":
                continue
            name = golden["name"]
            assert name in summary_by_name, f"{name} not in golden_summary.json"

            s = summary_by_name[name]
            m = golden["metrics"]

            assert abs(s["val_mse"] - m["val_mse"]) < 1e-6, (
                f"{name}: val_mse mismatch: summary={s['val_mse']}, manifest={m['val_mse']}"
            )
            assert abs(s["val_prd"] - m["val_prd"]) < 1e-4, (
                f"{name}: val_prd mismatch: summary={s['val_prd']}, manifest={m['val_prd']}"
            )
            assert abs(s["val_cos"] - m["val_cos"]) < 1e-6, (
                f"{name}: val_cos mismatch: summary={s['val_cos']}, manifest={m['val_cos']}"
            )


class TestGoldenConfigsParseable:
    """Verify golden YAML configs can be parsed by current Pydantic models."""

    def test_ecg_configs_parse(self, manifest: list[dict]) -> None:
        from compressionkit.configs.ecg_rvq import EcgRvqConfig

        for golden in manifest:
            if golden["signal"] != "ECG":
                continue
            path = REPO_ROOT / golden["config_yaml"]
            cfg = EcgRvqConfig.from_yaml(str(path))
            assert cfg.run_name == golden["name"], f"run_name mismatch: {cfg.run_name} != {golden['name']}"

    def test_ppg_configs_parse(self, manifest: list[dict]) -> None:
        from compressionkit.configs.ppg_rvq import PpgRvqConfig

        for golden in manifest:
            if golden["signal"] != "PPG":
                continue
            path = REPO_ROOT / golden["config_yaml"]
            cfg = PpgRvqConfig.from_yaml(str(path))
            assert cfg.run_name == golden["name"], f"run_name mismatch: {cfg.run_name} != {golden['name']}"
