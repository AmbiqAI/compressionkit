"""Golden experiment lifecycle runner.

Single entry point that drives a registered golden experiment through:

1. **Config load** — read the recipe's Pydantic config from the registered YAML.
2. **Train + evaluate + export** — invoke the registered recipe's
   training function. The recipe (via :class:`compressionkit.recipes.base_rvq.BaseRVQTrainer`)
   already runs evaluation, writes the deploy package, and emits the
   model card.
3. **HuggingFace publish (optional)** — shell out to
   :mod:`scripts.publish_to_huggingface` to upload the deploy package
   to ``hf_repo_id``.

Dataset acquisition is currently a manual prerequisite. The contract
that makes ingestion fully runner-driven is tracked in #26.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from compressionkit.experiments.registry import GoldenExperiment, get_golden
from compressionkit.recipes import get_recipe

logger = logging.getLogger(__name__)


def _resolve_run_dir(experiment: GoldenExperiment, results_root: Path) -> Path:
    return results_root / experiment.run_name


def _publish(experiment: GoldenExperiment, run_dir: Path, dry_run: bool) -> int:
    deploy_dir = run_dir / "deploy"
    if not deploy_dir.is_dir():
        raise FileNotFoundError(f"deploy directory missing for {experiment.experiment_id!r}: {deploy_dir}")
    scorecard = run_dir / "quality_scorecard.json"
    cmd: list[str] = [
        sys.executable,
        "scripts/publish_to_huggingface.py",
        "--deploy-dir",
        str(deploy_dir),
        "--repo-id",
        experiment.hf_repo_id,
    ]
    if scorecard.is_file():
        cmd.extend(["--scorecard", str(scorecard)])
    if dry_run:
        cmd.append("--dry-run")
    logger.info(
        "Publishing %s → %s%s", experiment.experiment_id, experiment.hf_repo_id, " (dry-run)" if dry_run else ""
    )
    return subprocess.run(cmd, check=False).returncode


def run_golden(
    experiment_id: str,
    *,
    results_root: Path | str = Path("results"),
    datasets_root: Path | str | None = None,
    skip_train: bool = False,
    skip_parent: bool = False,
    skip_dataset_check: bool = False,
    publish: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run a single golden experiment end-to-end.

    For ``two_stage`` experiments the parent codec is trained first
    (unless already on disk or ``skip_parent`` is set), then the prior
    is trained against the parent's run directory. The bundled deploy
    artifacts (encoder/decoder/codebook + ``prior_int8.tflite``) live in
    the parent's ``run_dir/deploy/``.

    Args:
        experiment_id: Registered golden experiment id.
        results_root: Root directory for run outputs (defaults to ``results/``).
        datasets_root: Optional override for the dataset root used by the
            pre-flight check.
        skip_train: If True, only publish an already-trained run.
        skip_parent: For two_stage entries: assume the parent codec is
            already trained. Skips the parent training step.
        skip_dataset_check: If True, do not pre-flight dataset availability.
        publish: If True, upload the deploy package to HuggingFace.
        dry_run: If True (and ``publish`` is set), stage without upload.

    Returns:
        Run summary with ``experiment_id``, ``run_dir``, ``trained``,
        ``published``, ``publish_returncode``, and for two_stage runs
        ``parent_trained`` and ``prior_summary``.
    """
    experiment = get_golden(experiment_id)
    results_root = Path(results_root)
    run_dir = _resolve_run_dir(experiment, results_root)

    parent_trained = False
    prior_summary: dict[str, object] | None = None
    trained = False

    if not skip_train:
        if not experiment.config_path.is_file():
            raise FileNotFoundError(f"config for {experiment.experiment_id!r} not found: {experiment.config_path}")
        if not skip_dataset_check:
            from compressionkit.datasets.contract import ensure_dataset_available

            root = Path(datasets_root) if datasets_root is not None else None
            ensure_dataset_available(experiment.dataset_id, root=root)

        if experiment.family == "two_stage":
            assert experiment.parent is not None  # validator enforces this
            parent_exp = get_golden(experiment.parent)
            parent_run_dir = _resolve_run_dir(parent_exp, results_root)
            if not (parent_run_dir / "deploy").is_dir() and not skip_parent:
                logger.info("Parent codec %s not yet trained; chaining its training.", parent_exp.experiment_id)
                parent_spec = get_recipe(parent_exp.recipe)
                parent_cfg = parent_spec.config_cls.from_yaml(str(parent_exp.config_path))  # type: ignore[attr-defined]
                parent_spec.train_fn(parent_cfg)
                parent_trained = True
            elif skip_parent:
                logger.info(
                    "--skip-parent: assuming %s is already trained at %s", parent_exp.experiment_id, parent_run_dir
                )
            spec = get_recipe(experiment.recipe)
            logger.info(
                "Training prior %s via recipe %r with %s",
                experiment.experiment_id,
                experiment.recipe,
                experiment.config_path,
            )
            cfg = spec.config_cls.from_yaml(str(experiment.config_path))  # type: ignore[attr-defined]
            cfg = cfg.model_copy(update={"parent_run_dir": parent_run_dir})
            prior_summary = spec.train_fn(cfg)
            trained = True
        else:
            spec = get_recipe(experiment.recipe)
            logger.info(
                "Training %s via recipe %r with %s",
                experiment.experiment_id,
                experiment.recipe,
                experiment.config_path,
            )
            cfg = spec.config_cls.from_yaml(str(experiment.config_path))  # type: ignore[attr-defined]
            spec.train_fn(cfg)
            trained = True

    published = False
    publish_rc: int | None = None
    if publish:
        publish_rc = _publish(experiment, run_dir, dry_run=dry_run)
        published = publish_rc == 0

    summary: dict[str, object] = {
        "experiment_id": experiment.experiment_id,
        "run_dir": str(run_dir),
        "trained": trained,
        "published": published,
        "publish_returncode": publish_rc,
    }
    if experiment.family == "two_stage":
        summary["parent_trained"] = parent_trained
        summary["prior_summary"] = prior_summary
    return summary


__all__ = ["run_golden"]
