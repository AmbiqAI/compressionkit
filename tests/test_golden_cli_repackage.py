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
