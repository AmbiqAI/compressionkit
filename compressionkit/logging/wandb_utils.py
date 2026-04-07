"""Weights & Biases helpers for training scripts."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import keras

LOGGER = logging.getLogger(__name__)


def _flatten_dict(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        k = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, k))
        else:
            flat[k] = value
    return flat


def _resolve_mode(mode: str) -> str:
    if mode in {"online", "offline", "disabled"}:
        return mode
    if mode != "auto":
        LOGGER.warning("Unsupported W&B mode '%s'; falling back to auto.", mode)
    return "online" if os.getenv("WANDB_API_KEY") else "offline"


def init_wandb_run(cfg: dict[str, Any], run_name: str, run_dir: Path):
    """Initialize a W&B run from config, returning ``None`` if disabled/unavailable."""
    wandb_cfg = cfg.get("output", {}).get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import failure is environment dependent
        LOGGER.warning("W&B enabled but import failed: %s", exc)
        return None

    mode = _resolve_mode(str(wandb_cfg.get("mode", "auto")))
    if mode == "disabled":
        return None

    try:
        run = wandb.init(
            project=wandb_cfg.get("project", "compression-kit"),
            entity=wandb_cfg.get("entity"),
            group=wandb_cfg.get("group"),
            job_type=wandb_cfg.get("job_type", "train"),
            tags=wandb_cfg.get("tags", []),
            name=run_name,
            dir=str(run_dir),
            mode=mode,
            config=_flatten_dict(cfg),
        )
        return run
    except Exception as exc:  # pragma: no cover - network/auth/runtime dependent
        LOGGER.warning("Failed to initialize W&B run: %s", exc)
        return None


def build_wandb_callbacks(run, log_model: bool) -> list[keras.callbacks.Callback]:
    """Build Keras callbacks for W&B metric logging."""
    del log_model
    if run is None:
        return []

    callbacks: list[keras.callbacks.Callback] = []
    try:
        from wandb.integration.keras import WandbMetricsLogger  # type: ignore[import-not-found]

        callbacks.append(WandbMetricsLogger())
        return callbacks
    except Exception as exc:  # pragma: no cover - optional integration path
        LOGGER.warning("Falling back to manual W&B metric logger: %s", exc)

    class _ManualWandbMetricsLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
            metrics = dict(logs or {})
            metrics["epoch"] = epoch + 1
            run.log(metrics)

    callbacks.append(_ManualWandbMetricsLogger())
    return callbacks


def finalize_wandb_run(
    run,
    summary: dict[str, Any],
    run_dir: Path,
    artifact_summary_only: bool,
) -> None:
    """Push summary metrics/artifacts and finish the W&B run."""
    if run is None:
        return

    try:
        final_metrics = summary.get("metrics", {}).get("final", {})
        compression = summary.get("compression", {})
        for key, value in final_metrics.items():
            run.summary[key] = value
        for key, value in compression.items():
            run.summary[f"compression_{key}"] = value

        import wandb  # type: ignore[import-not-found]

        artifact = wandb.Artifact(name=f"{run.name}-summary", type="experiment-summary")
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            artifact.add_file(str(summary_path), name="summary.json")

        for cfg_path in sorted(run_dir.glob("*.yaml")):
            artifact.add_file(str(cfg_path), name=cfg_path.name)

        if not artifact_summary_only:
            for filename in ("model.keras", "encoder.keras", "decoder.keras", "rvq_weights.npz"):
                path = run_dir / filename
                if path.exists():
                    artifact.add_file(str(path), name=filename)

        run.log_artifact(artifact)
    except Exception as exc:  # pragma: no cover - runtime/network dependent
        LOGGER.warning("Failed while finalizing W&B run: %s", exc)
    finally:
        run.finish()
