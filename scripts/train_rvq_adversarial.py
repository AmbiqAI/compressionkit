"""Adversarial fine-tuning of a pretrained RVQ autoencoder.

Loads a frozen-checkpoint compressor (encoder + RVQ + decoder), builds a
multi-scale discriminator, wraps them in ``AdversarialVQAutoencoder``,
and fine-tunes with the combined loss::

    L_G = L_recon + λ_adv · L_hinge + λ_feat · L_feat_match

The discriminator is trained with hinge loss.  It is **discarded** after
training — the deployed model is the same encoder + RVQ + decoder.

Reuses the dataset / preprocessing pipeline from
``compressionkit.trainers.ecg_rvq`` so the comparison vs the baseline
result is apples-to-apples.

Example::

    uv run python scripts/train_rvq_adversarial.py \\
        --run-dir results/ecg_rvq_256hz_32x_golden \\
        --output-dir results/ecg_rvq_256hz_32x_golden_adv_v1 \\
        --epochs 30 --adv-weight 1.0 --feat-weight 10.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from compressionkit.configs.ecg_rvq import EcgRvqConfig

logger = logging.getLogger("rvq-adv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_compressor(
    cfg: EcgRvqConfig, run_dir: Path, *, load_weights: bool = True,
) -> keras.Model:
    from compressionkit.trainers.ecg_rvq import build_model

    model = build_model(cfg)
    dummy = np.zeros(
        (1, 1, cfg.data.frame_size, max(1, cfg.data.num_leads or 1)),
        dtype=np.float32,
    )

    @tf.function(jit_compile=True)
    def _build_call(x):
        return model(x, training=False)
    _build_call(dummy)

    for name in ("best_model.weights.h5", "model.weights.h5"):
        p = run_dir / name
        if p.exists():
            model.load_weights(p)
            logger.info("Loaded compressor weights from %s", p)
            return model
    if load_weights:
        raise FileNotFoundError(f"No weights found in {run_dir}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Existing trained RVQ run directory to fine-tune.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write the fine-tuned weights and metadata.")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gen-lr", type=float, default=1e-4,
                        help="Generator (autoencoder) learning rate.")
    parser.add_argument("--disc-lr", type=float, default=1e-4,
                        help="Discriminator learning rate.")

    # Adversarial loss weights
    parser.add_argument("--adv-weight", type=float, default=1.0,
                        help="Weight for the hinge adversarial generator loss.")
    parser.add_argument("--feat-weight", type=float, default=10.0,
                        help="Weight for L1 feature-matching loss.")
    parser.add_argument("--disc-start-epoch", type=int, default=0,
                        help="Epoch at which discriminator begins training. "
                             "Earlier epochs use reconstruction-only loss.")

    # Discriminator architecture
    parser.add_argument("--disc-scales", type=int, default=3,
                        help="Number of resolution scales in multi-scale discriminator.")
    parser.add_argument("--disc-channels", type=int, nargs="+",
                        default=[16, 32, 64, 128],
                        help="Channel widths for each strided-conv stage.")
    parser.add_argument("--disc-kernel", type=int, default=15,
                        help="Kernel width for discriminator conv stages.")

    # Optional
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    keras.utils.set_random_seed(args.seed)

    out_dir: Path = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir.resolve()

    # ------------------------------------------------------------------
    # 1. Load compressor
    # ------------------------------------------------------------------
    logger.info("Loading compressor from %s ...", run_dir)
    cfg = EcgRvqConfig.model_validate_json((run_dir / "config.json").read_text())
    compressor = _load_compressor(cfg, run_dir)

    # Compile the inner autoencoder with its reconstruction loss so
    # AdversarialVQAutoencoder.train_step can delegate compute_loss.
    from compressionkit.trainers.ecg_rvq import build_extra_losses
    extras = build_extra_losses(cfg)
    compressor.compile(
        optimizer=keras.optimizers.Adam(args.gen_lr),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError(name="mse")],
        extra_losses=extras or None,
    )

    # ------------------------------------------------------------------
    # 2. Build discriminator
    # ------------------------------------------------------------------
    from compressionkit.layers.discriminator import MultiScaleDiscriminator

    frame_size = cfg.data.frame_size
    disc = MultiScaleDiscriminator(
        frame_size,
        num_scales=args.disc_scales,
        channels=tuple(args.disc_channels),
        kernel_width=args.disc_kernel,
    )
    # Build it
    dummy = np.zeros((1, 1, frame_size, 1), dtype=np.float32)
    disc(dummy)
    disc_params = sum(np.prod(v.shape) for v in disc.trainable_variables)
    logger.info(
        "Discriminator: %d scales, channels=%s, %d params (%.1f KB)",
        args.disc_scales, args.disc_channels, disc_params, disc_params * 4 / 1024,
    )

    # ------------------------------------------------------------------
    # 3. Wrap with adversarial objective
    # ------------------------------------------------------------------
    from compressionkit.trainers.adversarial import (
        AdversarialVQAutoencoder,
        SetEpochCallback,
    )

    adv_model = AdversarialVQAutoencoder(
        autoencoder=compressor,
        discriminator=disc,
        adv_weight=args.adv_weight,
        feat_weight=args.feat_weight,
        disc_start_epoch=args.disc_start_epoch,
    )
    adv_model.compile(
        gen_optimizer=keras.optimizers.Adam(args.gen_lr, beta_1=0.5, beta_2=0.9),
        disc_optimizer=keras.optimizers.Adam(args.disc_lr, beta_1=0.5, beta_2=0.9),
    )

    # ------------------------------------------------------------------
    # 4. Build datasets
    # ------------------------------------------------------------------
    from compressionkit.trainers.ecg_rvq import build_datasets
    from compressionkit.preprocessing.ecg import build_preprocessor, build_augmenter

    pre = build_preprocessor(
        frame_size=cfg.data.frame_size, epsilon=cfg.data.epsilon,
    )
    aug = build_augmenter(
        aug_cfg=cfg.data.augmentation,
        sample_rate=cfg.data.effective_sample_rate,
    )
    train_ds, val_ds, validation_steps, info = build_datasets(cfg, pre, aug)
    if args.validation_steps is not None:
        validation_steps = args.validation_steps
    logger.info("Datasets ready (%s); val steps=%s", info.get("mode"), validation_steps)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    callbacks = [
        SetEpochCallback(),
        keras.callbacks.CSVLogger(str(out_dir / "history.csv"), append=False),
    ]
    history = adv_model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        validation_steps=validation_steps,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
        verbose=2,
    )

    # ------------------------------------------------------------------
    # 6. Persist (only the compressor — discriminator is discarded)
    # ------------------------------------------------------------------
    compressor.save_weights(out_dir / "best_model.weights.h5")
    # Save config so downstream tooling treats this as a normal RVQ run.
    (out_dir / "config.json").write_text(cfg.model_dump_json(indent=2))

    metadata = {
        "source_run": str(run_dir),
        "adv_weight": args.adv_weight,
        "feat_weight": args.feat_weight,
        "disc_start_epoch": args.disc_start_epoch,
        "disc_scales": args.disc_scales,
        "disc_channels": [int(c) for c in args.disc_channels],
        "disc_kernel": args.disc_kernel,
        "disc_params": int(disc_params),
        "gen_lr": args.gen_lr,
        "disc_lr": args.disc_lr,
        "epochs": args.epochs,
        "history": {
            k: [float(v) for v in vals]
            for k, vals in history.history.items()
        },
    }
    (out_dir / "adv_metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Wrote: %s", sorted(p.name for p in out_dir.iterdir()))
    logger.info(
        "Done. Discriminator discarded. Evaluate with "
        "scripts/evaluate_rvq.py or similar on %s.", out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
