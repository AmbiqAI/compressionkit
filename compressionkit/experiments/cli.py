"""CLI for the golden experiment lifecycle.

Wired into the top-level ``compressionkit`` multiplexer as the
``golden`` subcommand (see :func:`compressionkit.recipes._registry.dispatch`).

Subcommands::

    compressionkit golden list [--modality {ppg,ecg}]
    compressionkit golden run <experiment_id> [--results-root PATH]
                                              [--skip-train]
                                              [--publish] [--dry-run]
    compressionkit golden run-all --modality {ppg,ecg}
                                  [--results-root PATH]
                                  [--publish] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from compressionkit.experiments.registry import (
    GoldenModality,
    get_golden,
    list_goldens,
)
from compressionkit.experiments.runner import run_golden

logger = logging.getLogger(__name__)


def _print_table(modality: GoldenModality | None) -> None:
    rows = list_goldens(modality)
    if not rows:
        print("(no golden experiments registered)")
        return
    header = f"{'EXPERIMENT_ID':16s}  {'MODALITY':8s}  {'FAMILY':9s}  {'CR':>3s}  {'CONFIG':45s}  HF_REPO_ID"
    print(header)
    print("-" * len(header))
    for exp in rows:
        config_path_str = str(exp.config_path)
        print(
            f"{exp.experiment_id:16s}  {exp.modality:8s}  {exp.family:9s}  "
            f"{exp.compression_ratio:>3d}  {config_path_str:45s}  {exp.hf_repo_id}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compressionkit golden", description="Run v1 golden experiments.")
    sub = parser.add_subparsers(dest="action", metavar="ACTION", required=True)

    p_list = sub.add_parser("list", help="List registered golden experiments.")
    p_list.add_argument("--modality", choices=["ppg", "ecg"], default=None)

    p_run = sub.add_parser("run", help="Run a single golden experiment end-to-end.")
    p_run.add_argument("experiment_id")
    p_run.add_argument("--results-root", type=Path, default=Path("results"))
    p_run.add_argument(
        "--datasets-root", type=Path, default=None, help="Override dataset root for the pre-flight check."
    )
    p_run.add_argument("--skip-train", action="store_true", help="Skip training; assume run_dir already exists.")
    p_run.add_argument("--skip-dataset-check", action="store_true", help="Skip the dataset availability pre-flight.")
    p_run.add_argument("--publish", action="store_true", help="Publish to HuggingFace after training.")
    p_run.add_argument("--dry-run", action="store_true", help="Stage publish files without uploading.")

    p_all = sub.add_parser("run-all", help="Run every golden in a modality, sequentially.")
    p_all.add_argument("--modality", choices=["ppg", "ecg"], required=True)
    p_all.add_argument("--results-root", type=Path, default=Path("results"))
    p_all.add_argument("--datasets-root", type=Path, default=None)
    p_all.add_argument("--skip-dataset-check", action="store_true")
    p_all.add_argument("--publish", action="store_true")
    p_all.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.action == "list":
        _print_table(args.modality)
        return 0

    if args.action == "run":
        get_golden(args.experiment_id)  # validate early
        summary = run_golden(
            args.experiment_id,
            results_root=args.results_root,
            datasets_root=args.datasets_root,
            skip_train=args.skip_train,
            skip_dataset_check=args.skip_dataset_check,
            publish=args.publish,
            dry_run=args.dry_run,
        )
        logger.info("golden run summary: %s", summary)
        return 0 if (not args.publish or summary["published"] or args.dry_run) else 1

    # run-all
    failures: list[str] = []
    for exp in list_goldens(args.modality):
        try:
            run_golden(
                exp.experiment_id,
                results_root=args.results_root,
                datasets_root=args.datasets_root,
                skip_dataset_check=args.skip_dataset_check,
                publish=args.publish,
                dry_run=args.dry_run,
            )
        except Exception:  # keep running other goldens
            logger.exception("golden %s failed", exp.experiment_id)
            failures.append(exp.experiment_id)
    if failures:
        logger.error("golden run-all completed with failures: %s", failures)
        return 1
    return 0


__all__ = ["build_parser", "main"]
