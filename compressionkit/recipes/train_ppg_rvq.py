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

from compressionkit.configs.ppg_rvq import PpgRvqConfig
from compressionkit.export.deploy import export_for_deployment
from compressionkit.logging.wandb_utils import finalize_wandb_run, init_wandb_run
from compressionkit.preprocessing.ppg import build_augmenter, build_preprocessor
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
from compressionkit.trainers.ppg_rvq import (
    assemble_summary,
    build_compression_stats,
    build_datasets,
    build_model,
    compile_model,
    logger,
    run_evaluation,
)
from compressionkit.trainers.utils import build_callbacks, build_learning_rate, setup_logger


@recipe("train-ppg-rvq", config_cls=PpgRvqConfig)
def train(cfg: PpgRvqConfig) -> dict[str, Any]:
    """Run the full PPG RVQ training pipeline.

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
    augmenter = build_augmenter(tuple(data.gaussian_noise))
    train_ds, val_ds, validation_steps, ds_info = build_datasets(cfg, preprocessor, augmenter)
    logger.info("Dataset mode: %s", ds_info["mode"])

    # ------------------------------------------------------------------
    # 3. Model: RVQ autoencoder + compile (optimizer, losses, metrics)
    # ------------------------------------------------------------------
    model = build_model(cfg)
    lr_value = build_learning_rate(cfg, steps_per_epoch=data.steps_per_epoch)
    compile_model(model, cfg, learning_rate=lr_value)

    # Optional: mini-batch k-means warm start for EMA RVQ codebooks.
    from compressionkit.layers import EmaResidualVectorQuantizer

    if (
        cfg.model.kmeans_init
        and isinstance(getattr(model, "vq", None), EmaResidualVectorQuantizer)
    ):
        import numpy as _np

        try:
            import tensorflow as _tf
            _peek_ctx = _tf.device("/CPU:0")
        except Exception:
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
    # 6. Evaluation: reconstruction samples, band metrics, physiokit, long-recording
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
