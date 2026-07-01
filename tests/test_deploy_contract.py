"""Tests for the v1 deploy-package artifact contract."""

from __future__ import annotations

import json

from compressionkit.export.deploy import sync_scorecard_to_deploy
from compressionkit.export.release import write_checksums, write_json
from compressionkit.export.validate import validate_deploy_package


def _write_minimal_manifest(root, *, family: str, scorecard: str | None = "scorecard.json") -> None:
    write_json(
        root / "deploy_manifest.json",
        {
            "manifest_version": 1,
            "package_version": "1.0",
            "family": family,
            "method": "ai" if family == "rvq" else "dsp",
            "modality": "ppg",
            "compression_ratio": 4,
            "spec": "codec_spec.json",
            "scorecard": scorecard,
            "checksums": "checksums.json",
        },
    )
    write_json(root / "codec_spec.json", {"family": family, "modality": "ppg"})


def test_strict_rvq_contract_requires_release_artifacts(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="rvq", scorecard=None)
    (tmp_path / "encoder.tflite").write_bytes(b"encoder")
    (tmp_path / "codebook.npz").write_bytes(b"codebook")
    (tmp_path / "codebook.h").write_text("/* codebook */")
    write_checksums(tmp_path)

    result = validate_deploy_package(
        tmp_path,
        check_runtime=False,
        check_reference_vectors=False,
        strict_release=True,
    )

    assert not result.ok
    assert "missing release artifact: model_card.json" in result.errors
    assert "missing release artifact: scorecard.json" in result.errors
    assert "missing release artifact: reference_vectors.npz" in result.errors
    assert "missing release artifact: sample_data.npz" in result.errors


def test_strict_rvq_contract_accepts_complete_file_set(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="rvq")
    for rel in [
        "encoder.tflite",
        "codebook.npz",
        "codebook.h",
        "model_card.json",
        "scorecard.json",
        "reference_vectors.npz",
        "sample_data.npz",
        "README.md",
    ]:
        (tmp_path / rel).write_bytes(b"artifact")
    write_checksums(tmp_path)

    result = validate_deploy_package(
        tmp_path,
        check_runtime=False,
        check_reference_vectors=False,
        strict_release=True,
    )

    assert result.ok
    assert result.warnings == []
    assert "scorecard.json" in result.checked_files
    assert "sample_data.npz" in result.checked_files


def test_strict_spiht_contract_accepts_complete_file_set(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="spiht")
    for rel in [
        "sample_stimulus.npz",
        "reference_vectors.npz",
        "spiht_app_config.h",
        "model_card.json",
        "scorecard.json",
        "README.md",
    ]:
        (tmp_path / rel).write_bytes(b"artifact")
    write_checksums(tmp_path)

    result = validate_deploy_package(
        tmp_path,
        check_runtime=False,
        check_reference_vectors=False,
        strict_release=True,
    )

    assert result.ok
    assert "sample_stimulus.npz" in result.checked_files


def test_strict_hybrid_contract_accepts_complete_file_set(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="hybrid")
    for rel in [
        "sample_stimulus.npz",
        "reference_vectors.npz",
        "spiht_app_config.h",
        "denoiser_gain_model.keras",
        "denoiser_train_config.json",
        "hybrid_manifest.json",
        "model_card.json",
        "scorecard.json",
        "README.md",
    ]:
        (tmp_path / rel).write_bytes(b"artifact")
    write_checksums(tmp_path)

    result = validate_deploy_package(
        tmp_path,
        check_runtime=False,
        check_reference_vectors=False,
        strict_release=True,
    )

    assert result.ok
    assert "hybrid_manifest.json" in result.checked_files
    assert "denoiser_gain_model.keras" in result.checked_files


def test_checksum_mismatch_is_error(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="rvq")
    for rel in ["encoder.tflite", "codebook.npz", "codebook.h"]:
        (tmp_path / rel).write_bytes(b"artifact")
    write_checksums(tmp_path)
    (tmp_path / "encoder.tflite").write_bytes(b"changed")

    result = validate_deploy_package(
        tmp_path,
        check_runtime=False,
        check_reference_vectors=False,
    )

    assert "checksum mismatch for encoder.tflite" in result.errors


def test_manifest_scorecard_points_to_written_scorecard(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="rvq")
    manifest = json.loads((tmp_path / "deploy_manifest.json").read_text())
    assert manifest["scorecard"] == "scorecard.json"


def test_sync_scorecard_to_deploy_updates_manifest_model_card_and_checksums(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="rvq", scorecard=None)
    write_json(
        tmp_path / "model_card.json",
        {
            "model_name": "ppg_rvq_64hz_04x_golden",
            "model_version": "1.0",
            "scorecard_summary": {},
        },
    )
    write_checksums(tmp_path)

    scorecard = {"time_domain": {"prd_percent": {"mean": 2.5}}}
    scorecard_path = sync_scorecard_to_deploy(tmp_path, scorecard)

    assert scorecard_path.name == "scorecard.json"
    assert json.loads(scorecard_path.read_text()) == scorecard

    manifest = json.loads((tmp_path / "deploy_manifest.json").read_text())
    assert manifest["scorecard"] == "scorecard.json"
    assert manifest["artifacts"]["scorecard"] == "scorecard.json"

    model_card = json.loads((tmp_path / "model_card.json").read_text())
    assert model_card["scorecard_summary"] == scorecard

    checksums = json.loads((tmp_path / "checksums.json").read_text())
    assert "scorecard.json" in checksums
