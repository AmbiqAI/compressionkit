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
    skip_train: bool = False,
    publish: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run a single golden experiment end-to-end.

    Args:
        experiment_id: Registered golden experiment id.
        results_root: Root directory for run outputs (defaults to ``results/``).
        skip_train: If True, only publish an already-trained run (requires the
            deploy directory to exist).
        publish: If True, upload the deploy package to HuggingFace after training.
        dry_run: If True (and ``publish`` is set), generate the staging
            directory but do not upload.

    Returns:
        A dict summarising the run: ``experiment_id``, ``run_dir``,
        ``trained`` (bool), ``published`` (bool), ``publish_returncode``.
    """
    experiment = get_golden(experiment_id)
    results_root = Path(results_root)
    run_dir = _resolve_run_dir(experiment, results_root)

    trained = False
    if not skip_train:
        if not experiment.config_path.is_file():
            raise FileNotFoundError(f"config for {experiment.experiment_id!r} not found: {experiment.config_path}")
        spec = get_recipe(experiment.recipe)
        logger.info(
            "Training %s via recipe %r with %s", experiment.experiment_id, experiment.recipe, experiment.config_path
        )
        cfg = spec.config_cls.from_yaml(str(experiment.config_path))  # type: ignore[attr-defined]
        spec.train_fn(cfg)
        trained = True

    published = False
    publish_rc: int | None = None
    if publish:
        publish_rc = _publish(experiment, run_dir, dry_run=dry_run)
        published = publish_rc == 0

    return {
        "experiment_id": experiment.experiment_id,
        "run_dir": str(run_dir),
        "trained": trained,
        "published": published,
        "publish_returncode": publish_rc,
    }


__all__ = ["run_golden"]
