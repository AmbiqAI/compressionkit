"""Golden ECG RVQ training recipe.

Read this file top-to-bottom to understand the full ECG training flow. It
parallels :mod:`compressionkit.recipes.train_ppg_rvq` but handles the
transform-domain (raw / DWT / STFT) model branch used for ECG. Copy into
the top-level ``recipes/`` folder and edit freely to build on it.
"""

from __future__ import annotations

from typing import Any

import keras

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.preprocessing.ecg import build_augmenter, build_preprocessor
from compressionkit.recipes._registry import recipe
from compressionkit.recipes.base_rvq import BaseRVQTrainer
from compressionkit.trainers.ecg_rvq import (
    assemble_summary,
    build_compression_stats,
    build_datasets,
    build_model,
    compile_model,
    logger,
    run_evaluation,
)


class RvqPrefixLossScale(keras.callbacks.Callback):
    """Schedule ``model.prefix_loss_scale`` for prefix-supervised RVQ."""

    def __init__(self, start_epoch: int, ramp_epochs: int):
        super().__init__()
        self.start_epoch = max(0, int(start_epoch))
        self.ramp_epochs = max(0, int(ramp_epochs))

    def on_epoch_begin(self, epoch: int, logs=None):
        if epoch < self.start_epoch:
            scale = 0.0
        elif self.ramp_epochs > 0:
            scale = min(1.0, (epoch - self.start_epoch + 1) / float(self.ramp_epochs))
        else:
            scale = 1.0
        self.model.prefix_loss_scale.assign(scale)


class EcgRVQTrainer(BaseRVQTrainer[EcgRvqConfig]):
    """ECG-specific hooks for the shared RVQ recipe flow."""

    @property
    def logger(self):
        return logger

    def build_preprocessor(self) -> keras.layers.Layer:
        data = self.cfg.data
        return build_preprocessor(frame_size=data.frame_size, epsilon=data.epsilon)

    def build_augmenter(self) -> keras.layers.Layer:
        data = self.cfg.data
        return build_augmenter(aug_cfg=data.augmentation, sample_rate=data.effective_sample_rate)

    def build_datasets(
        self,
        preprocessor: keras.layers.Layer,
        augmenter: keras.layers.Layer,
    ) -> tuple[Any, Any, int | None, dict[str, Any]]:
        return build_datasets(self.cfg, preprocessor, augmenter)

    def build_model(self) -> keras.Model:
        return build_model(self.cfg)

    def compile_model(
        self,
        model: keras.Model,
        *,
        learning_rate: float | keras.optimizers.schedules.LearningRateSchedule,
    ) -> None:
        compile_model(model, self.cfg, learning_rate=learning_rate)

    def build_extra_callbacks(self, model: keras.Model) -> list[keras.callbacks.Callback]:
        callbacks: list[keras.callbacks.Callback] = []

        beta_cfg = self.cfg.training.beta_anneal
        if beta_cfg.enabled:
            from compressionkit.callbacks import BetaAnneal

            callbacks.append(
                BetaAnneal(
                    model.vq,
                    start=beta_cfg.start,
                    end=beta_cfg.end,
                    epochs=beta_cfg.epochs,
                    mode=beta_cfg.mode,
                )
            )
            logger.info(
                "BetaAnneal enabled: %.3f -> %.3f over %d epochs (%s)",
                beta_cfg.start,
                beta_cfg.end,
                beta_cfg.epochs,
                beta_cfg.mode,
            )

        prefix_cfg = self.cfg.training.rvq_prefix_loss
        if prefix_cfg.enabled and hasattr(model, "prefix_loss_scale"):
            callbacks.append(
                RvqPrefixLossScale(
                    start_epoch=prefix_cfg.start_epoch,
                    ramp_epochs=prefix_cfg.ramp_epochs,
                )
            )
            logger.info(
                "RVQ prefix loss schedule enabled: start_epoch=%d, ramp_epochs=%d",
                prefix_cfg.start_epoch,
                prefix_cfg.ramp_epochs,
            )

        return callbacks

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


@recipe("train-ecg-rvq", config_cls=EcgRvqConfig)
def train(cfg: EcgRvqConfig) -> dict[str, Any]:
    """Run the full ECG RVQ training pipeline.

    Args:
        cfg: Validated pipeline configuration.

    Returns:
        Summary dictionary with metrics, compression stats, and artifact paths.
    """
    return EcgRVQTrainer(cfg).train()


if __name__ == "__main__":
    import sys

    sys.exit(main())  # noqa: F821  # main is injected by @recipe


__all__ = ["train"]
