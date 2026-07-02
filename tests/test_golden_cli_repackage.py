"""Tests for the golden CLI repackage subcommand."""

from __future__ import annotations

from pathlib import Path

from compressionkit.experiments import cli


def test_parser_accepts_repackage_subcommand() -> None:
    args = cli.build_parser().parse_args(
        [
            "repackage",
            "ppg-rvq-4x",
            "--results-root",
            "results",
            "--strict-release-validation",
        ]
    )

    assert args.action == "repackage"
    assert args.experiment_id == "ppg-rvq-4x"
    assert args.results_root == Path("results")
    assert args.strict_release_validation is True


def test_main_dispatches_repackage(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "get_golden", lambda experiment_id: object())
    monkeypatch.setattr(
        cli,
        "repackage_golden",
        lambda experiment_id, **kwargs: calls.append((experiment_id, kwargs)) or {"validation": None},
    )

    rc = cli.main(["repackage", "ppg-rvq-4x", "--skip-validation"])

    assert rc == 0
    assert calls == [
        (
            "ppg-rvq-4x",
            {
                "results_root": Path("results"),
                "output_dir": None,
                "num_stimulus": 10,
                "export_decoder_int8": True,
                "scorecard_path": None,
                "validate": False,
                "strict_release_validation": False,
            },
        )
    ]


def test_validate_all_reports_pass_fail_and_skip(monkeypatch, tmp_path, capsys) -> None:
    from compressionkit.experiments.registry import GoldenExperiment
    from compressionkit.export.validate import DeployValidationResult

    def _exp(experiment_id: str, cr: int) -> GoldenExperiment:
        return GoldenExperiment(
            experiment_id=experiment_id,
            modality="ppg",
            structure="codec",
            method="rvq",
            run_name=f"ppg_rvq_64hz_{cr:02d}x_golden",
            sample_rate=64,
            compression_ratio=cr,
            hf_repo_id=f"Ambiq/compressionkit-ppg-{cr}x",
            dataset_id="ppg-unified-strict-sanitize-v1",
        )

    built = tmp_path / "ppg_rvq_64hz_04x_golden" / "deploy"
    built.mkdir(parents=True)
    experiments = [
        _exp("ppg-ok", 4),
        _exp("ppg-missing", 8),
    ]

    monkeypatch.setattr(cli, "list_goldens", lambda modality, method: experiments)
    monkeypatch.setattr(
        cli,
        "validate_deploy_package",
        lambda path, **kwargs: DeployValidationResult(path, "rvq", [], [], ["deploy_manifest.json"]),
    )

    rc = cli.main(["validate-all", "--results-root", str(tmp_path), "--strict-release"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "ppg-ok" in out and "OK" in out
    assert "ppg-missing" in out and "SKIP" in out
    assert "1/1 passed, 0 failed, 1 skipped" in out


def test_validate_all_fails_when_any_package_fails(monkeypatch, tmp_path) -> None:
    from compressionkit.experiments.registry import GoldenExperiment
    from compressionkit.export.validate import DeployValidationResult

    deploy_dir = tmp_path / "ppg_rvq_64hz_04x_golden" / "deploy"
    deploy_dir.mkdir(parents=True)
    experiment = GoldenExperiment(
        experiment_id="ppg-bad",
        modality="ppg",
        structure="codec",
        method="rvq",
        run_name="ppg_rvq_64hz_04x_golden",
        sample_rate=64,
        compression_ratio=4,
        hf_repo_id="Ambiq/compressionkit-ppg-4x",
        dataset_id="ppg-unified-strict-sanitize-v1",
    )

    monkeypatch.setattr(cli, "list_goldens", lambda modality, method: [experiment])
    monkeypatch.setattr(
        cli,
        "validate_deploy_package",
        lambda path, **kwargs: DeployValidationResult(path, "rvq", ["checksum mismatch"], [], []),
    )

    rc = cli.main(["validate-all", "--results-root", str(tmp_path)])

    assert rc == 1
