"""Tests for shared deploy-package release helpers."""

from __future__ import annotations

import json

from compressionkit.export.release import (
    build_model_card,
    build_release_metadata,
    sha256_file,
    write_checksums,
    write_json,
    write_model_card,
    write_scorecard_artifact,
)


def test_write_json_and_checksums(tmp_path) -> None:
    payload_path = write_json(tmp_path / "payload.json", {"ok": True})
    assert json.loads(payload_path.read_text()) == {"ok": True}

    checksums_path = write_checksums(tmp_path)
    checksums = json.loads(checksums_path.read_text())
    assert checksums["payload.json"]["sha256"] == sha256_file(payload_path)
    assert checksums["payload.json"]["bytes"] == payload_path.stat().st_size
    assert "checksums.json" not in checksums


def test_build_release_metadata_allows_experiment_extras() -> None:
    metadata = build_release_metadata(
        run_name="ppg_rvq_64hz_04x_golden",
        modality="ppg",
        sample_rate=64,
        compression_ratio=4,
        experiment_id="issue-42",
        extra={"license": "other"},
    )
    assert metadata == {
        "run_name": "ppg_rvq_64hz_04x_golden",
        "modality": "ppg",
        "sample_rate": 64,
        "compression_ratio": 4,
        "experiment_id": "issue-42",
        "license": "other",
    }


def test_model_card_helpers(tmp_path) -> None:
    model_card_info = {
        "modality": "ecg",
        "sample_rate": 256,
        "compression_ratio": 8,
        "scorecard_summary": {"prd_mean": 2.5},
    }
    model_card = build_model_card(
        model_name="ecg_rvq_256hz_08x",
        model_version="1.0",
        model_card_info=model_card_info,
    )
    assert model_card["model_name"] == "ecg_rvq_256hz_08x"
    assert model_card["license"] == "other"
    assert model_card["scorecard_summary"] == {"prd_mean": 2.5}

    model_card_path = write_model_card(
        tmp_path,
        model_name="ecg_rvq_256hz_08x",
        model_version="1.0",
        model_card_info=model_card_info,
    )
    assert json.loads(model_card_path.read_text()) == model_card


def test_write_scorecard_artifact(tmp_path) -> None:
    scorecard_path = write_scorecard_artifact(tmp_path, {"quality": "ok"})
    assert scorecard_path.name == "scorecard.json"
    assert json.loads(scorecard_path.read_text()) == {"quality": "ok"}
