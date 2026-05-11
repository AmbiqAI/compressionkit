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
from compressionkit.export.deploy import export_for_deployment
from compressionkit.logging.wandb_utils import finalize_wandb_run, init_wandb_run
from compressionkit.preprocessing.ecg import build_augmenter, build_preprocessor
from compressionkit.recipes._registry import recipe
from compressionkit.trainers.common import (
    BEST_CKPT_NAME,
    collect_rep_dataset,
    extract_history_metrics,
    reload_best_weights,
    save_config_snapshot,
    save_model_artifacts,
    setup_run_dir,
    write_summary,
)
from compressionkit.trainers.ecg_rvq import (
    assemble_summary,
    build_compression_stats,
    build_datasets,
    build_model,
    compile_model,
    logger,
    run_evaluation,
)
from compressionkit.trainers.utils import build_callbacks, build_learning_rate, setup_logger


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


@recipe("train-ecg-rvq", config_cls=EcgRvqConfig)
def train(cfg: EcgRvqConfig) -> dict[str, Any]:
    """Run the full ECG RVQ training pipeline.

    Args:
        cfg: Validated pipeline configuration.

    Returns:
        Summary dictionary with metrics, compression stats, and artifact paths.
    """
    # ------------------------------------------------------------------
    # 1. Run setup: directory, logging, config snapshot, W&B
    # ------------------------------------------------------------------
    run_dir = setup_run_dir(cfg.output.results_root, cfg.run_name)
    setup_logger(logger, run_dir, cfg.output.log_file)
    logger.info("Results directory: %s", run_dir)

    cfg_dump = cfg.model_dump()
    save_config_snapshot(cfg_dump, run_dir)
    wandb_run = init_wandb_run(cfg=cfg_dump, run_name=cfg.run_name, run_dir=run_dir)

    # ------------------------------------------------------------------
    # 2. Data pipeline: preprocessing, augmentation, train/val datasets
    # ------------------------------------------------------------------
    data = cfg.data
    preprocessor = build_preprocessor(frame_size=data.frame_size, epsilon=data.epsilon)
    augmenter = build_augmenter(aug_cfg=data.augmentation, sample_rate=data.effective_sample_rate)
    train_ds, val_ds, validation_steps, ds_info = build_datasets(cfg, preprocessor, augmenter)
    logger.info("Dataset mode: %s", ds_info["mode"])

    # ------------------------------------------------------------------
    # 3. Model: RVQ autoencoder (1-D or 2-D spatial for STFT) + compile
    # ------------------------------------------------------------------
    model = build_model(cfg)
    lr_value = build_learning_rate(cfg, steps_per_epoch=data.steps_per_epoch)
    compile_model(model, cfg, learning_rate=lr_value)

    # Optional: mini-batch k-means warm start for EMA RVQ codebooks. Run
    # eagerly on a few peeked batches before fit so the first training step
    # already sees a meaningful codebook and avoids the "everyone-on-one-code"
    # cold start that often causes silent collapse with K=256+.
    from compressionkit.layers import EmaResidualVectorQuantizer

    if (
        cfg.model.kmeans_init
        and isinstance(getattr(model, "vq", None), EmaResidualVectorQuantizer)
    ):
        import numpy as _np
        # Some TF GPU kernels (DepthwiseConv2D with stride=(1,2)) only work
        # under XLA, which model.fit enables but a bare eager call does not.
        # Force the warm-start peek onto CPU so it runs in any backend.
        try:
            import tensorflow as _tf
            _peek_ctx = _tf.device("/CPU:0")
        except Exception:  # non-TF backend
            from contextlib import nullcontext
            _peek_ctx = nullcontext()

        peek_batches: list = []
        target_rows = max(8 * model.vq.Ks[0], 4096)
        seen = 0
        with _peek_ctx:
            for x_batch, _y in train_ds.take(8):
                z = model.encoder(x_batch, training=False)
                peek_batches.append(_np.asarray(z))
                seen += int(_np.prod(_np.asarray(z).shape[:-1]))
                if seen >= target_rows:
                    break
        if peek_batches:
            z_concat = _np.concatenate(
                [b.reshape(-1, b.shape[-1]) for b in peek_batches], axis=0,
            )
            # Force the VQ layer to build (creates _codebooks / _ema_*) before
            # we try to assign new centroids into it.
            if not model.vq.built:
                model.vq.build(peek_batches[0].shape)
            model.vq.warm_start_kmeans(z_concat)
            logger.info(
                "EMA RVQ k-means warm-start: %d residuals across %d levels (K=%s)",
                z_concat.shape[0], len(model.vq.Ks), model.vq.Ks,
            )

    # ------------------------------------------------------------------
    # 4. Fit: standard Keras training loop with our callbacks
    # ------------------------------------------------------------------
    callbacks = build_callbacks(cfg, run_dir=run_dir, lr_value=lr_value, wandb_run=wandb_run)

    # Optional: anneal RVQ commitment-loss weight (beta) over training.
    # Must be attached BEFORE the first fit step so the swap from float
    # to keras.Variable happens before the train function is traced.
    beta_cfg = cfg.training.beta_anneal
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
            beta_cfg.start, beta_cfg.end, beta_cfg.epochs, beta_cfg.mode,
        )

    prefix_cfg = cfg.training.rvq_prefix_loss
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

    history = model.fit(
        train_ds,
        steps_per_epoch=data.steps_per_epoch,
        epochs=data.epochs,
        validation_data=val_ds,
        validation_steps=validation_steps,
        verbose=2,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------
    # 5. Restore best checkpoint and persist model artifacts
    # ------------------------------------------------------------------
    reload_best_weights(model, run_dir)
    model_artifacts = save_model_artifacts(model, run_dir)

    # ------------------------------------------------------------------
    # 6. Evaluation: reconstruction samples + band metrics
    # ------------------------------------------------------------------
    eval_results = run_evaluation(
        cfg, model=model, val_ds=val_ds, run_dir=run_dir, validation_steps=validation_steps,
    )

    # ------------------------------------------------------------------
    # 7. Deployment export: encoder/decoder TFLite + codebook + C headers
    # ------------------------------------------------------------------
    rep_dataset = collect_rep_dataset(
        val_ds,
        num_batches=cfg.evaluation.tflite_rep_batches,
        fallback=eval_results["sample_inputs"],
    )
    deploy = export_for_deployment(
        model.encoder,
        model.decoder,
        model.vq.get_weights(),
        rep_dataset=rep_dataset,
        output_dir=run_dir / "deploy",
        sample_inputs=eval_results["sample_inputs"],
        sample_targets=eval_results["sample_targets"],
        sample_reconstructions=eval_results["sample_reconstructions"],
        model_name=cfg.run_name,
    )

    # ------------------------------------------------------------------
    # 8. Summary assembly: best epoch, compression stats, summary.json
    # ------------------------------------------------------------------
    best_epoch, best_metrics, final_metrics, selection_metric = extract_history_metrics(
        history.history, selection_metric=cfg.training.selection_metric,
    )
    compression = build_compression_stats(cfg)

    model_artifacts["best_model"] = BEST_CKPT_NAME
    summary = assemble_summary(
        cfg,
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
    logger.info(
        "Best epoch by %s: %d (value=%.6f)",
        selection_metric, best_epoch, best_metrics[selection_metric],
    )

    # ------------------------------------------------------------------
    # 9. Finalize W&B (uploads summary, metadata, optional artifacts)
    # ------------------------------------------------------------------
    finalize_wandb_run(
        run=wandb_run,
        summary=summary,
        run_dir=run_dir,
        artifact_summary_only=cfg.output.wandb.artifact_summary_only,
    )
    return summary


if __name__ == "__main__":
    import sys

    sys.exit(main())  # noqa: F821  # main is injected by @recipe


__all__ = ["train"]
