"""Golden experiment lifecycle runner.

Single entry point that drives a registered golden experiment through:

1. **Config load** — read the recipe's Pydantic config (for trainable
   methods) or derive the codec parameters from the experiment fields
   (for DSP-only methods such as SPIHT).
2. **Build + evaluate + export** — for ``method == "rvq"`` this invokes
   the registered training recipe; for ``method == "spiht"`` it
   instantiates a :class:`SpihtCodec` with the modality defaults and
   writes a weightless deploy package via
   :func:`compressionkit.export.export_spiht_deploy`.
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
from compressionkit.export.validate import DeployValidationResult, validate_deploy_package
from compressionkit.recipes import get_recipe

logger = logging.getLogger(__name__)


# Modality defaults for DSP-only SPIHT goldens. Mirrors AGENTS.md
# guidance (PPG → coif5 L=6; ECG → bior4.4 L=6) and the frame sizes
# already used by the RVQ goldens at the same sample rate.
class _SpihtModalityDefaults:
    """Typed per-modality SPIHT operating-point defaults."""

    __slots__ = ("frame_size", "levels", "use_ac", "wavelet")

    def __init__(self, *, wavelet: str, levels: int, frame_size: int, use_ac: bool) -> None:
        self.wavelet = wavelet
        self.levels = levels
        self.frame_size = frame_size
        self.use_ac = use_ac


_SPIHT_DEFAULTS: dict[str, _SpihtModalityDefaults] = {
    "ppg": _SpihtModalityDefaults(wavelet="coif5", levels=6, frame_size=320, use_ac=True),
    "ecg": _SpihtModalityDefaults(wavelet="bior4.4", levels=6, frame_size=512, use_ac=True),
}


def _resolve_run_dir(experiment: GoldenExperiment, results_root: Path) -> Path:
    return results_root / experiment.run_name


def _build_spiht(experiment: GoldenExperiment, run_dir: Path) -> dict[str, object]:
    """Build the SPIHT deploy package for a DSP-only golden."""
    from compressionkit.export.spiht_deploy import export_spiht_deploy
    from compressionkit.runtime.spiht import SpihtCodec

    defaults = _SPIHT_DEFAULTS.get(experiment.modality)
    if defaults is None:
        raise ValueError(
            f"No SPIHT defaults registered for modality {experiment.modality!r}; "
            "extend _SPIHT_DEFAULTS in compressionkit.experiments.runner."
        )

    codec = SpihtCodec(
        modality=experiment.modality,
        sample_rate=experiment.sample_rate,
        frame_size=defaults.frame_size,
        target_cr=float(experiment.compression_ratio),
        wavelet=defaults.wavelet,
        levels=defaults.levels,
        use_ac=defaults.use_ac,
        name=experiment.experiment_id.replace("-", "_"),
    )

    deploy_dir = run_dir / "deploy"
    model_card_info: dict[str, object] = {
        "experiment_id": experiment.experiment_id,
        "modality": experiment.modality,
        "method": experiment.method,
        "sample_rate": experiment.sample_rate,
        "compression_ratio": experiment.compression_ratio,
        "wavelet": codec.wavelet,
        "levels": codec.levels,
        "use_ac": codec.use_ac,
        "frame_size": codec.frame_size,
        "dataset_id": experiment.dataset_id,
        "license": "Apache-2.0",
    }
    scorecard = dict(experiment.expected_metrics or {})

    logger.info(
        "Building SPIHT deploy for %s (wavelet=%s, levels=%d, frame_size=%d, cr=%g)",
        experiment.experiment_id,
        codec.wavelet,
        codec.levels,
        codec.frame_size,
        codec.target_cr,
    )
    arts = export_spiht_deploy(
        codec,
        output_dir=deploy_dir,
        model_card_info=model_card_info,
        scorecard_summary=scorecard,
    )
    return {"deploy_dir": str(deploy_dir), "artifacts": arts.as_dict()}


def _summarize_validation(result: DeployValidationResult) -> dict[str, object]:
    return {
        "ok": result.ok,
        "family": result.family,
        "checked_files": result.checked_files,
        "warnings": result.warnings,
        "errors": result.errors,
    }


def _validate_release_package(
    experiment: GoldenExperiment,
    run_dir: Path,
    *,
    strict_release: bool,
) -> DeployValidationResult:
    deploy_dir = run_dir / "deploy"
    if not deploy_dir.is_dir():
        raise FileNotFoundError(f"deploy directory missing for {experiment.experiment_id!r}: {deploy_dir}")
    result = validate_deploy_package(deploy_dir, strict_release=strict_release)
    if result.errors:
        details = "; ".join(result.errors)
        raise RuntimeError(f"deploy validation failed for {experiment.experiment_id!r}: {details}")
    if result.warnings:
        logger.warning("deploy validation warnings for %s: %s", experiment.experiment_id, "; ".join(result.warnings))
    logger.info("deploy validation passed for %s (%s)", experiment.experiment_id, result.family)
    return result


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
    validate: bool = True,
    strict_release_validation: bool = False,
) -> dict[str, object]:
    """Run a single golden experiment end-to-end.

    For trainable methods (``method == "rvq"``) this delegates to the
    registered recipe. For DSP-only methods (``method == "spiht"``) it
    instantiates the codec directly and writes a weightless deploy
    package. For ``two_stage`` experiments the parent codec is trained
    first (unless already on disk or ``skip_parent`` is set), then the
    prior is trained against the parent's run directory.

    Args:
        experiment_id: Registered golden experiment id.
        results_root: Root directory for run outputs (defaults to ``results/``).
        datasets_root: Optional override for the dataset root used by the
            pre-flight check.
        skip_train: If True, only publish an already-built run.
        skip_parent: For two_stage entries: assume the parent codec is
            already trained. Skips the parent training step.
        skip_dataset_check: If True, do not pre-flight dataset availability.
        publish: If True, upload the deploy package to HuggingFace.
        dry_run: If True (and ``publish`` is set), stage without upload.
        validate: If True, validate the deploy package after build/resume and
            before optional publishing.
        strict_release_validation: If True, require release-contract extras
            such as model cards and scorecards during deploy validation.

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
    spiht_summary: dict[str, object] | None = None
    trained = False

    if not skip_train:
        # DSP-only SPIHT path: no dataset / recipe needed, just build + export.
        if experiment.method == "spiht":
            run_dir.mkdir(parents=True, exist_ok=True)
            spiht_summary = _build_spiht(experiment, run_dir)
            trained = True
        else:
            if experiment.config_path is None or not experiment.config_path.is_file():
                raise FileNotFoundError(f"config for {experiment.experiment_id!r} not found: {experiment.config_path}")
            if experiment.recipe is None:
                raise ValueError(
                    f"experiment {experiment.experiment_id!r} has method={experiment.method!r} but no recipe declared"
                )
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
                    assert parent_exp.recipe is not None and parent_exp.config_path is not None
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

    validation: dict[str, object] | None = None
    if validate:
        validation_result = _validate_release_package(
            experiment,
            run_dir,
            strict_release=strict_release_validation,
        )
        validation = _summarize_validation(validation_result)

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
        "validation": validation,
    }
    if experiment.family == "two_stage":
        summary["parent_trained"] = parent_trained
        summary["prior_summary"] = prior_summary
    if experiment.method == "spiht":
        summary["spiht_build"] = spiht_summary
    return summary


__all__ = ["run_golden"]
