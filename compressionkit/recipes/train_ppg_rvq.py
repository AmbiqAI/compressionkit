"""Golden PPG RVQ training recipe.

Read this file top-to-bottom to understand the full PPG training flow.
Every step delegates to a reusable building block so individual pieces can
be swapped or tuned without touching the recipe itself. To customize,
copy this file into the top-level ``recipes/`` folder and edit freely.

The YAML config controls *what varies across runs* (dataset paths, model
width, augmentation, loss weights). The recipe controls *how the pieces
fit together*. Prefer editing the recipe over widening the config.
"""

from __future__ import annotations

from typing import Any

import keras

from compressionkit.configs.ppg_rvq import PpgRvqConfig
from compressionkit.preprocessing.ppg import build_augmenter, build_preprocessor
from compressionkit.recipes._registry import recipe
from compressionkit.recipes.base_rvq import BaseRVQTrainer
from compressionkit.trainers.ppg_rvq import (
    assemble_summary,
    build_compression_stats,
    build_datasets,
    build_model,
    compile_model,
    logger,
    run_evaluation,
)


class PpgRVQTrainer(BaseRVQTrainer[PpgRvqConfig]):
    """PPG-specific hooks for the shared RVQ recipe flow."""

    @property
    def logger(self):
        return logger

    def build_preprocessor(self) -> keras.layers.Layer:
        data = self.cfg.data
        return build_preprocessor(frame_size=data.frame_size, epsilon=data.epsilon)

    def build_augmenter(self) -> keras.layers.Layer:
        return build_augmenter(
            tuple(self.cfg.data.gaussian_noise),
            aug_cfg=self.cfg.data.augmentation,
        )

    def build_datasets(
        self,
        preprocessor: keras.layers.Layer,
        augmenter: keras.layers.Layer,
    ) -> tuple[Any, Any, int | None, dict[str, Any]]:
        train_ds, val_ds, validation_steps, ds_info = build_datasets(self.cfg, preprocessor, augmenter)
        # Stash the artifact-suite augmenter (if any) so the curriculum callback
        # can ramp its severity scale during training.
        self._artifact_suite_augmenter = ds_info.get("artifact_suite_augmenter")
        return train_ds, val_ds, validation_steps, ds_info

    def build_extra_callbacks(self, model: keras.Model) -> list[keras.callbacks.Callback]:
        """Append the artifact-suite curriculum callback when configured."""
        from compressionkit.preprocessing.artifact_suite import make_curriculum_callback

        augmenter = getattr(self, "_artifact_suite_augmenter", None)
        if augmenter is None:
            return []
        callback = make_curriculum_callback(augmenter)
        return [callback] if callback is not None else []

    def build_model(self) -> keras.Model:
        return build_model(self.cfg)

    def compile_model(
        self,
        model: keras.Model,
        *,
        learning_rate: float | keras.optimizers.schedules.LearningRateSchedule,
    ) -> None:
        compile_model(model, self.cfg, learning_rate=learning_rate)

    def run_evaluation(
        self,
        *,
        model: keras.Model,
        val_ds: Any,
        run_dir,
        validation_steps: int | None,
    ) -> dict[str, Any]:
        return run_evaluation(
            self.cfg,
            model=model,
            val_ds=val_ds,
            run_dir=run_dir,
            validation_steps=validation_steps,
        )

    def build_compression_stats(self) -> dict[str, Any]:
        return build_compression_stats(self.cfg)

    def assemble_summary(
        self,
        *,
        cfg_dump: dict[str, Any],
        history: dict[str, list[float]],
        eval_results: dict[str, Any],
        ds_info: dict[str, Any],
        compression: dict[str, Any],
        deploy_artifacts: dict[str, Any],
        model_artifacts: dict[str, str],
        best_epoch: int,
        best_metrics: dict[str, float],
        final_metrics: dict[str, float],
        selection_metric: str,
        validation_steps: int | None,
    ) -> dict[str, Any]:
        return assemble_summary(
            self.cfg,
            cfg_dump=cfg_dump,
            history=history,
            eval_results=eval_results,
            ds_info=ds_info,
            compression=compression,
            deploy_artifacts=deploy_artifacts,
            model_artifacts=model_artifacts,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            final_metrics=final_metrics,
            selection_metric=selection_metric,
            validation_steps=validation_steps,
        )


@recipe("train-ppg-rvq", config_cls=PpgRvqConfig)
def train(cfg: PpgRvqConfig) -> dict[str, Any]:
    """Run the full PPG RVQ training pipeline.

    Args:
        cfg: Validated pipeline configuration.

    Returns:
        Summary dictionary with metrics, compression stats, and artifact paths.
    """
    return PpgRVQTrainer(cfg).train()


if __name__ == "__main__":
    import sys

    sys.exit(main())  # noqa: F821  # main is injected by @recipe


__all__ = ["train"]
