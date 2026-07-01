"""Tests for the golden release packager helpers."""

from __future__ import annotations

import json

from compressionkit.experiments.repackage import (
    finalize_release_metadata,
    load_scorecard_payload,
    resolve_scorecard_path,
)
from compressionkit.export.release import write_checksums, write_json


def test_resolve_scorecard_path_defaults_to_run_scorecard(tmp_path) -> None:
    scorecard_path = tmp_path / "quality_scorecard.json"
    scorecard_path.write_text("{}")

    resolved = resolve_scorecard_path(tmp_path, None)

    assert resolved == scorecard_path


def test_load_scorecard_payload_returns_full_payload(tmp_path) -> None:
    scorecard_path = tmp_path / "quality_scorecard.json"
    payload = {
        "time_domain": {"prd_percent": {"mean": 2.0}},
        "spectral": {"log_spectral_distance_db": {"mean": 0.1}},
        "safety": {"zero_input": {"passed": True}},
    }
    scorecard_path.write_text(json.dumps(payload))

    loaded = load_scorecard_payload(scorecard_path)

    assert loaded == payload


def test_finalize_release_metadata_syncs_scorecard_and_checksums(tmp_path) -> None:
    write_json(
        tmp_path / "deploy_manifest.json",
        {
            "manifest_version": 1,
            "package_version": "1.0",
            "family": "rvq",
            "method": "ai",
            "modality": "ppg",
            "spec": "codec_spec.json",
            "scorecard": None,
            "checksums": "checksums.json",
            "artifacts": {},
        },
    )
    write_json(tmp_path / "codec_spec.json", {"family": "rvq", "modality": "ppg"})
    write_json(tmp_path / "model_card.json", {"model_name": "demo", "model_version": "1.0", "scorecard_summary": {}})
    (tmp_path / "sample_stimulus.npz").write_bytes(b"stimulus")
    write_checksums(tmp_path)

    scorecard = {"safety": {"zero_input": {"passed": True}}}
    finalize_release_metadata(tmp_path, scorecard)

    manifest = json.loads((tmp_path / "deploy_manifest.json").read_text())
    checksums = json.loads((tmp_path / "checksums.json").read_text())
    model_card = json.loads((tmp_path / "model_card.json").read_text())

    assert manifest["scorecard"] == "scorecard.json"
    assert manifest["artifacts"]["scorecard"] == "scorecard.json"
    assert json.loads((tmp_path / "scorecard.json").read_text()) == scorecard
    assert model_card["scorecard_summary"] == scorecard
    assert "sample_stimulus.npz" in checksums
    assert "scorecard.json" in checksums


def test_finalize_release_metadata_refreshes_checksums_without_scorecard(tmp_path) -> None:
    (tmp_path / "sample_stimulus.npz").write_bytes(b"stimulus")
    write_checksums(tmp_path)

    finalize_release_metadata(tmp_path, None)

    checksums = json.loads((tmp_path / "checksums.json").read_text())
    assert "sample_stimulus.npz" in checksums
