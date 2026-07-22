"""Shared orchestration for RVQ training recipes.

The ECG and PPG golden recipes follow the same top-level flow: create a run
directory, build data and model objects, fit, evaluate, export deploy
artifacts, and serialize a summary. This module keeps that orchestration in one
place while leaving modality-specific details in thin subclasses.

This is intentionally a convenience layer for the shipped golden recipes, not a
required framework entry point for all experiments. New experiments should be
able to import smaller blocks directly and only adopt this shared flow when it
actually reduces boilerplate.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import keras

from compressionkit.evaluation.scorecard import write_quality_scorecard
from compressionkit.export.deploy import export_for_deployment, sync_scorecard_to_deploy
from compressionkit.export.release import build_release_metadata
from compressionkit.export.validate import validate_deploy_package
from compressionkit.logging.wandb_utils import finalize_wandb_run, init_wandb_run
from compressionkit.trainers.common import (
    BEST_CKPT_NAME,
    collect_disjoint_quantization_datasets,
    extract_history_metrics,
    reload_best_weights,
    save_config_snapshot,
    save_model_artifacts,
    setup_run_dir,
    write_long_recording_eval,
    write_summary,
)
from compressionkit.trainers.utils import build_callbacks, build_learning_rate, setup_logger


def _build_long_recording_payload(eval_results: dict[str, Any]) -> dict[str, Any] | None:
    """Extract long-recording HR/HRV results for ``long_recording_eval.json``.

    Returns ``None`` when the run did not produce long-recording results
    (typical for short smoke runs or modalities with the eval disabled).
    Packaged shape:

    - PPG: ``{"modality": "ppg", "summary": {...}, "per_recording": [...]}``
    - ECG: ``{"modality": "ecg", "stitching": {...}}`` where the stitching
      block now includes per-method HR/HRV aggregates (issue #3).
    """
    long_ppg = eval_results.get("long_recording_metrics")
    if long_ppg is not None:
        return {
            "modality": "ppg",
            "summary": long_ppg,
            "per_recording": eval_results.get("long_recording_per_recording") or [],
        }
    stitching = eval_results.get("stitching")
    if stitching is not None:
        return {"modality": "ecg", "stitching": stitching}
    return None


class BaseRVQTrainer[ConfigT](ABC):
    """Optional template-method orchestrator for RVQ recipes.

    The contract for an experiment is the artifacts it can produce, not whether
    it subclasses this type. Use this when it helps; bypass it when a custom
    experiment needs a different control flow.
    """

    def __init__(self, cfg: ConfigT):
        self.cfg = cfg

    @property
    @abstractmethod
    def logger(self):
        """Return the modality-specific logger."""

    @abstractmethod
    def build_preprocessor(self) -> keras.layers.Layer:
        """Build the input preprocessor layer."""

    @abstractmethod
    def build_augmenter(self) -> keras.layers.Layer:
        """Build the augmentation layer."""

    @abstractmethod
    def build_datasets(
        self,
        preprocessor: keras.layers.Layer,
        augmenter: keras.layers.Layer,
    ) -> tuple[Any, Any, int | None, dict[str, Any]]:
        """Build train and validation datasets plus metadata."""

    @abstractmethod
    def build_model(self) -> keras.Model:
        """Construct the trainable model."""

    @abstractmethod
    def compile_model(
        self,
        model: keras.Model,
        *,
        learning_rate: float | keras.optimizers.schedules.LearningRateSchedule,
    ) -> None:
        """Compile the trainable model."""

    @abstractmethod
    def run_evaluation(
        self,
        *,
        model: keras.Model,
        val_ds: Any,
        run_dir: Path,
        validation_steps: int | None,
    ) -> dict[str, Any]:
        """Run post-training evaluation and return summary inputs."""

    @abstractmethod
    def build_compression_stats(self) -> dict[str, Any]:
        """Compute compression statistics for the current config."""

    @abstractmethod
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
        """Build the run summary payload."""

    def build_callbacks(
        self,
        *,
        run_dir: Path,
        lr_value: float | keras.optimizers.schedules.LearningRateSchedule,
        wandb_run: Any,
        model: keras.Model,
    ) -> list[keras.callbacks.Callback]:
        """Build base callbacks plus modality-specific additions."""
        callbacks = build_callbacks(self.cfg, run_dir=run_dir, lr_value=lr_value, wandb_run=wandb_run)
        callbacks.extend(self.build_extra_callbacks(model))
        return callbacks

    def build_extra_callbacks(self, model: keras.Model) -> list[keras.callbacks.Callback]:
        """Return modality-specific callbacks appended before fit()."""
        return []

    def maybe_warm_start_codebooks(self, model: keras.Model, train_ds: Any) -> None:
        """Warm-start EMA RVQ codebooks from a few encoder activations."""
        from compressionkit.layers import EmaResidualVectorQuantizer

        if not self.cfg.model.kmeans_init or not isinstance(getattr(model, "vq", None), EmaResidualVectorQuantizer):
            return

        import numpy as np

        try:
            import tensorflow as tf

            peek_ctx = tf.device("/CPU:0")
        except Exception:
            peek_ctx = nullcontext()

        peek_batches: list[Any] = []
        target_rows = max(8 * model.vq.Ks[0], 4096)
        seen = 0
        with peek_ctx:
            for x_batch, _y in train_ds.take(8):
                z = model.encoder(x_batch, training=False)
                peek_batches.append(np.asarray(z))
                seen += int(np.prod(np.asarray(z).shape[:-1]))
                if seen >= target_rows:
                    break
        if not peek_batches:
            return

        z_concat = np.concatenate(
            [batch.reshape(-1, batch.shape[-1]) for batch in peek_batches],
            axis=0,
        )
        if not model.vq.built:
            model.vq.build(peek_batches[0].shape)
        model.vq.warm_start_kmeans(z_concat)
        self.logger.info(
            "EMA RVQ k-means warm-start: %d residuals across %d levels (K=%s)",
            z_concat.shape[0],
            len(model.vq.Ks),
            model.vq.Ks,
        )

    def train(self) -> dict[str, Any]:
        """Run the full end-to-end RVQ recipe."""
        cfg_dump = self.cfg.model_dump()
        run_dir = setup_run_dir(self.cfg.output.results_root, self.cfg.run_name)
        setup_logger(self.logger, run_dir, self.cfg.output.log_file)
        self.logger.info("Results directory: %s", run_dir)

        save_config_snapshot(cfg_dump, run_dir)
        wandb_run = init_wandb_run(cfg=cfg_dump, run_name=self.cfg.run_name, run_dir=run_dir)

        data = self.cfg.data
        preprocessor = self.build_preprocessor()
        augmenter = self.build_augmenter()
        train_ds, val_ds, validation_steps, ds_info = self.build_datasets(preprocessor, augmenter)
        self.logger.info("Dataset mode: %s", ds_info["mode"])

        model = self.build_model()
        lr_value = build_learning_rate(self.cfg, steps_per_epoch=data.steps_per_epoch)
        self.compile_model(model, learning_rate=lr_value)
        self.maybe_warm_start_codebooks(model, train_ds)

        callbacks = self.build_callbacks(
            run_dir=run_dir,
            lr_value=lr_value,
            wandb_run=wandb_run,
            model=model,
        )
        history = model.fit(
            train_ds,
            steps_per_epoch=data.steps_per_epoch,
            epochs=data.epochs,
            validation_data=val_ds,
            validation_steps=validation_steps,
            verbose=2,
            callbacks=callbacks,
        )

        reload_best_weights(model, run_dir)
        model_artifacts = save_model_artifacts(model, run_dir)

        eval_results = self.run_evaluation(
            model=model,
            val_ds=val_ds,
            run_dir=run_dir,
            validation_steps=validation_steps,
        )
        compression = self.build_compression_stats()

        # Prefer the effective (post-downsample) operating rate when the
        # config declares one (e.g. ECG's target_sample_rate=256 vs a
        # sampling_rate=500 source); only fall back to sampling_rate for
        # configs that don't distinguish source vs operating rate (e.g. PPG).
        sample_rate = getattr(self.cfg.data, "effective_sample_rate", None)
        if sample_rate is None:
            sample_rate = getattr(self.cfg.data, "sampling_rate", None)
        unified_cache = getattr(self.cfg.data, "unified_cache", None)
        dataset_sources: list[str] | None = None
        if unified_cache is not None and getattr(unified_cache, "enabled", False):
            dataset_sources = [source.slug for source in unified_cache.sources]
        model_card_info = build_release_metadata(
            run_name=self.cfg.run_name,
            modality=str(self.cfg.run_name).split("_", 1)[0],
            sample_rate=sample_rate,
            compression_ratio=compression.get("compression_ratio"),
            dataset_sources=dataset_sources,
        )

        rep_dataset, quantization_validation_dataset = collect_disjoint_quantization_datasets(
            val_ds,
            calibration_frames=getattr(self.cfg.evaluation, "int8_calibration_frames", 4096),
            validation_frames=getattr(self.cfg.evaluation, "int8_validation_frames", 2048),
            sampling_pool_frames=getattr(self.cfg.evaluation, "int8_sampling_pool_frames", 65_536),
            seed=getattr(self.cfg.data, "shuffle_seed", 42),
        )
        deploy = export_for_deployment(
            model.encoder,
            model.decoder,
            model.vq.get_weights(),
            rep_dataset=rep_dataset,
            quantization_validation_dataset=quantization_validation_dataset,
            output_dir=run_dir / "deploy",
            sample_inputs=eval_results["sample_inputs"],
            sample_targets=eval_results["sample_targets"],
            sample_reconstructions=eval_results["sample_reconstructions"],
            model_name=self.cfg.run_name,
            model_card_info=model_card_info,
        )

        best_epoch, best_metrics, final_metrics, selection_metric = extract_history_metrics(
            history.history,
            selection_metric=self.cfg.training.selection_metric,
        )

        model_artifacts["best_model"] = BEST_CKPT_NAME
        summary = self.assemble_summary(
            cfg_dump=cfg_dump,
            history=history.history,
            eval_results=eval_results,
            ds_info=ds_info,
            compression=compression,
            deploy_artifacts=deploy.as_dict(),
            model_artifacts=model_artifacts,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            final_metrics=final_metrics,
            selection_metric=selection_metric,
            validation_steps=validation_steps,
        )
        write_summary(summary, run_dir)

        if sample_rate is None:
            self.logger.info("Skipping quality scorecard build because sample_rate metadata is unavailable.")
        else:
            try:
                scorecard_path = write_quality_scorecard(
                    run_dir,
                    modality=model_card_info["modality"],
                    sample_rate=int(sample_rate),
                )
            except Exception:
                self.logger.exception("Quality scorecard build failed; continuing without deploy scorecard sync.")
            else:
                scorecard_summary = json.loads(scorecard_path.read_text())
                deploy.scorecard = sync_scorecard_to_deploy(run_dir / "deploy", scorecard_summary)
                deploy.checksums = deploy.output_dir / "checksums.json"
                summary["artifacts"]["deploy"] = deploy.as_dict()

                try:
                    validation_result = validate_deploy_package(run_dir / "deploy", strict_release=True)
                except Exception:
                    self.logger.exception("Deploy package self-validation crashed; treat this release as unverified.")
                else:
                    if validation_result.errors:
                        self.logger.error(
                            "Deploy package failed strict release validation: %s",
                            "; ".join(validation_result.errors),
                        )
                    else:
                        self.logger.info("Deploy package passed strict release validation.")
                    summary["artifacts"]["deploy_validation"] = {
                        "ok": validation_result.ok,
                        "errors": validation_result.errors,
                        "warnings": validation_result.warnings,
                    }
                write_summary(summary, run_dir)

        _long_payload = _build_long_recording_payload(eval_results)
        if _long_payload is not None:
            write_long_recording_eval(_long_payload, run_dir)
        self.logger.info(
            "Best epoch by %s: %d (value=%.6f)",
            selection_metric,
            best_epoch,
            best_metrics[selection_metric],
        )

        finalize_wandb_run(
            run=wandb_run,
            summary=summary,
            run_dir=run_dir,
            artifact_summary_only=self.cfg.output.wandb.artifact_summary_only,
        )
        return summary


__all__ = ["BaseRVQTrainer"]
