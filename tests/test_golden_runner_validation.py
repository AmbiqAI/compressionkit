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
        dataset_id="ppg-unified-strict-sanitize-v1",
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


def test_run_golden_spiht_uses_evaluator_script(monkeypatch, tmp_path) -> None:
    experiment = GoldenExperiment(
        experiment_id="ppg-spiht-4x",
        modality="ppg",
        family="codec",
        method="spiht",
        recipe=None,
        config_path=None,
        run_name="ppg_spiht_64hz_04x_golden",
        sample_rate=64,
        compression_ratio=4,
        hf_repo_id="Ambiq/compressionkit-ppg-spiht-4x",
        dataset_id="ppg-unified-strict-sanitize-v1",
    )
    run_dir = tmp_path / experiment.run_name
    deploy_dir = run_dir / "deploy"

    calls: list[list[str]] = []

    monkeypatch.setattr(runner, "get_golden", lambda experiment_id: experiment)

    def _fake_run(cmd, check=False):
        calls.append(list(cmd))
        deploy_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        runner,
        "validate_deploy_package",
        lambda path, *, strict_release: DeployValidationResult(path, "spiht", [], [], ["deploy_manifest.json"]),
    )

    summary = runner.run_golden(
        experiment.experiment_id,
        results_root=tmp_path,
    )

    assert len(calls) == 1
    assert calls[0][0].endswith("python") or calls[0][0] == runner.sys.executable
    assert calls[0][1] == "scripts/run_spiht_golden_ppg.py"
    assert calls[0][2:] == [
        "--experiment-id",
        experiment.experiment_id,
        "--results-root",
        str(tmp_path),
    ]
    assert summary["trained"] is True
    assert summary["validation"] == {
        "ok": True,
        "family": "spiht",
        "checked_files": ["deploy_manifest.json"],
        "warnings": [],
        "errors": [],
    }
    assert summary["spiht_build"] == {
        "command": calls[0],
        "deploy_dir": str(deploy_dir),
    }


def test_repackage_golden_validates_repackaged_deploy(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_name: ppg_rvq_64hz_04x_golden\n")
    experiment = _experiment(config_path)
    run_dir = tmp_path / experiment.run_name
    run_dir.mkdir(parents=True)

    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(runner, "get_golden", lambda experiment_id: experiment)
    monkeypatch.setattr(
        runner,
        "repackage_rvq_golden",
        lambda exp, **kwargs: calls.append(("repackage", kwargs))
        or {"experiment_id": exp.experiment_id, "run_dir": str(run_dir), "deploy_dir": str(run_dir / "deploy"), "artifacts": {}},
    )
    monkeypatch.setattr(
        runner,
        "validate_deploy_package",
        lambda path, *, strict_release: calls.append(("validate", (path, strict_release)))
        or DeployValidationResult(path, "rvq", [], [], ["deploy_manifest.json"]),
    )

    summary = runner.repackage_golden(
        experiment.experiment_id,
        results_root=tmp_path,
        strict_release_validation=True,
    )

    assert [name for name, _ in calls] == ["repackage", "validate"]
    assert summary["validation"] == {
        "ok": True,
        "family": "rvq",
        "checked_files": ["deploy_manifest.json"],
        "warnings": [],
        "errors": [],
    }


def test_repackage_golden_rejects_non_rvq(monkeypatch, tmp_path) -> None:
    experiment = GoldenExperiment(
        experiment_id="ppg-spiht-4x",
        modality="ppg",
        family="codec",
        method="spiht",
        recipe=None,
        config_path=None,
        run_name="ppg_spiht_64hz_04x_golden",
        sample_rate=64,
        compression_ratio=4,
        hf_repo_id="Ambiq/compressionkit-ppg-spiht-4x",
        dataset_id="ppg-unified-strict-sanitize-v1",
    )

    monkeypatch.setattr(runner, "get_golden", lambda experiment_id: experiment)

    with pytest.raises(ValueError, match="supports only RVQ"):
        runner.repackage_golden(experiment.experiment_id, results_root=tmp_path)
