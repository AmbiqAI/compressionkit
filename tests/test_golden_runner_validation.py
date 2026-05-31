"""Tests for deploy-package validation in the golden runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from compressionkit.experiments import runner
from compressionkit.experiments.registry import GoldenExperiment
from compressionkit.export.validate import DeployValidationResult


def _experiment(config_path: Path) -> GoldenExperiment:
    return GoldenExperiment(
        experiment_id="ppg-rvq-4x",
        modality="ppg",
        family="codec",
        method="rvq",
        recipe="train-ppg-rvq",
        config_path=config_path,
        run_name="ppg_rvq_64hz_04x_golden",
        sample_rate=64,
        compression_ratio=4,
        hf_repo_id="Ambiq/compressionkit-ppg-4x",
        dataset_id="mesa",
    )


def test_run_golden_validates_before_publish(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_name: ppg_rvq_64hz_04x_golden\n")
    experiment = _experiment(config_path)
    deploy_dir = tmp_path / experiment.run_name / "deploy"
    deploy_dir.mkdir(parents=True)

    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(runner, "get_golden", lambda experiment_id: experiment)
    monkeypatch.setattr(
        runner,
        "get_recipe",
        lambda name: SimpleNamespace(
            config_cls=SimpleNamespace(from_yaml=lambda path: SimpleNamespace()),
            train_fn=lambda cfg: calls.append(("train", cfg)),
        ),
    )
    monkeypatch.setattr(
        runner,
        "validate_deploy_package",
        lambda path, *, strict_release: calls.append(("validate", (path, strict_release)))
        or DeployValidationResult(path, "rvq", [], ["missing release artifact: scorecard.json"], ["deploy_manifest.json"]),
    )
    monkeypatch.setattr(runner, "_publish", lambda exp, run_dir, dry_run: calls.append(("publish", run_dir)) or 0)

    summary = runner.run_golden(
        experiment.experiment_id,
        results_root=tmp_path,
        skip_dataset_check=True,
        publish=True,
    )

    assert [name for name, _ in calls] == ["train", "validate", "publish"]
    assert summary["published"] is True
    assert summary["validation"] == {
        "ok": True,
        "family": "rvq",
        "checked_files": ["deploy_manifest.json"],
        "warnings": ["missing release artifact: scorecard.json"],
        "errors": [],
    }


def test_run_golden_blocks_publish_on_validation_errors(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_name: ppg_rvq_64hz_04x_golden\n")
    experiment = _experiment(config_path)
    deploy_dir = tmp_path / experiment.run_name / "deploy"
    deploy_dir.mkdir(parents=True)

    monkeypatch.setattr(runner, "get_golden", lambda experiment_id: experiment)
    monkeypatch.setattr(
        runner,
        "validate_deploy_package",
        lambda path, *, strict_release: DeployValidationResult(path, "rvq", ["checksum mismatch"], [], []),
    )
    monkeypatch.setattr(runner, "_publish", lambda exp, run_dir, dry_run: pytest.fail("publish should not run"))

    with pytest.raises(RuntimeError, match="deploy validation failed"):
        runner.run_golden(
            experiment.experiment_id,
            results_root=tmp_path,
            skip_train=True,
            publish=True,
        )


def test_run_golden_can_skip_validation(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_name: ppg_rvq_64hz_04x_golden\n")
    experiment = _experiment(config_path)
    (tmp_path / experiment.run_name / "deploy").mkdir(parents=True)

    monkeypatch.setattr(runner, "get_golden", lambda experiment_id: experiment)
    monkeypatch.setattr(
        runner,
        "validate_deploy_package",
        lambda path, *, strict_release: pytest.fail("validation should not run"),
    )
    monkeypatch.setattr(runner, "_publish", lambda exp, run_dir, dry_run: 0)

    summary = runner.run_golden(
        experiment.experiment_id,
        results_root=tmp_path,
        skip_train=True,
        publish=True,
        validate=False,
    )

    assert summary["published"] is True
    assert summary["validation"] is None
