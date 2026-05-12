"""Train an SSM-based ECG autoencoder for comparison against RVQ.

Reuses the same cached PTB-XL TFRecord pipeline as the golden RVQ runs, so
results are directly comparable.  By default, this targets the same setup as
``configs/ecg_rvq_256hz_32x_golden.yaml`` (frame_size=512, num_stages=4,
~32× spatial downsampling) — but with a continuous SSM bottleneck instead of
RVQ.

Usage
-----
    python scripts/train_ssm_ecg.py \\
        --rvq-config configs/ecg_rvq_256hz_32x_golden.yaml \\
        --epochs 30 \\
        --steps-per-epoch 200 \\
        --latent-dim 4 \\
        --state-size 32 \\
        --num-ssm-blocks 2 \\
        --output-suffix v0

Reports val MSE / PRD / cosine similarity, parameter count, and effective
compression ratio (assuming float32 latent storage).  Saves the model and
metrics under ``results/ecg_ssm_<run_name>/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Pin TF backend before any keras import so the comparison runs consistently.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import tensorflow as tf
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.layers import (
    EmaResidualVectorQuantizer,
    FiniteScalarQuantizer,
    ResidualVectorQuantizer,
)
from compressionkit.models.ssm_autoencoder import (
    build_ssm_autoencoder,
    compute_compression_ratio,
)
from compressionkit.preprocessing.ecg import (
    build_augmenter,
    build_preprocessor,
)
from compressionkit.trainers.ecg_rvq import build_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_ssm_ecg")


def _squeeze_h(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Convert (B, 1, T, C) → (B, T, C) for a 1D model."""
    return ds.map(
        lambda x, y: (tf.squeeze(x, axis=1), tf.squeeze(y, axis=1)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def _prd_metric(y_true, y_pred):
    """Percent Root-mean-square Difference (lower is better)."""
    num = tf.reduce_sum(tf.square(y_true - y_pred), axis=(1, 2))
    den = tf.reduce_sum(tf.square(y_true), axis=(1, 2)) + 1e-8
    return 100.0 * tf.reduce_mean(tf.sqrt(num / den))


def _make_loss(derivative_weight: float):
    """MSE + optional first-difference (derivative) loss."""
    mse = keras.losses.MeanSquaredError()

    def loss(y_true, y_pred):
        l = mse(y_true, y_pred)
        if derivative_weight > 0.0:
            dy_t = y_true[:, 1:, :] - y_true[:, :-1, :]
            dy_p = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            l = l + derivative_weight * tf.reduce_mean(tf.square(dy_t - dy_p))
        return l

    return loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rvq-config", type=str, required=True, help="Path to a golden RVQ YAML to reuse data + train hparams."
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--validation-steps", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=4)
    p.add_argument("--base-filters", type=int, default=16)
    p.add_argument("--multiplier", type=float, default=1.5)
    p.add_argument("--state-size", type=int, default=32)
    p.add_argument("--num-ssm-blocks", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument(
        "--derivative-weight", type=float, default=0.1, help="Weight on first-difference loss. Set 0 to disable."
    )
    p.add_argument(
        "--cosine-restarts", action="store_true", help="Use cosine-restart LR schedule (matches golden RVQ)."
    )
    p.add_argument(
        "--quantizer",
        choices=["none", "fsq", "rvq"],
        default="none",
        help="Bottleneck quantizer applied to the SSM latent.",
    )
    p.add_argument(
        "--fsq-levels", type=str, default="8,5,5,5", help="Comma-separated per-dim levels for FSQ (D=len(levels))."
    )
    p.add_argument("--rvq-num-levels", type=int, default=1)
    p.add_argument("--rvq-codebook-size", type=int, default=256)
    p.add_argument("--rvq-beta", type=float, default=0.25)
    p.add_argument("--rvq-ema", action="store_true", help="Use EmaResidualVectorQuantizer (matches golden RVQ).")
    p.add_argument("--rvq-ema-decay", type=float, default=0.99)
    p.add_argument("--output-suffix", type=str, default="v0")
    p.add_argument("--results-root", type=str, default="results")
    p.add_argument("--smoke-test", action="store_true", help="Run 1 epoch with 5 steps to verify the pipeline.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.rvq_config)
    with cfg_path.open() as f:
        cfg_dict = yaml.safe_load(f)
    cfg = EcgRvqConfig(**cfg_dict)

    # Build the same preprocessor/augmenter the RVQ run uses.
    preprocessor = build_preprocessor(
        frame_size=cfg.data.frame_size,
        epsilon=cfg.data.epsilon,
    )
    augmenter = build_augmenter(noise_factor=tuple(cfg.data.gaussian_noise))

    train_ds, val_ds, validation_steps, info = build_datasets(
        cfg,
        preprocessor,
        augmenter,
    )
    train_ds = _squeeze_h(train_ds)
    val_ds = _squeeze_h(val_ds)
    if args.validation_steps is not None:
        validation_steps = args.validation_steps
    logger.info("Dataset info: %s, validation_steps=%s", info, validation_steps)

    num_stages = cfg.model.num_stages

    # ----- Build the bottleneck quantizer (if any).
    quantizer: keras.layers.Layer | None = None
    bits_per_token: float | None = None
    effective_latent_dim = args.latent_dim
    if args.quantizer == "fsq":
        levels = [int(v) for v in args.fsq_levels.split(",") if v]
        effective_latent_dim = len(levels)
        bits_per_token = sum(np.log2(L) for L in levels)
        quantizer = FiniteScalarQuantizer(levels=levels, name="fsq")
        logger.info("FSQ levels=%s  bits/token=%.2f  codebook=%d", levels, bits_per_token, int(np.prod(levels)))
    elif args.quantizer == "rvq":
        bits_per_token = args.rvq_num_levels * np.log2(args.rvq_codebook_size)
        if args.rvq_ema:
            quantizer = EmaResidualVectorQuantizer(
                num_levels=args.rvq_num_levels,
                num_embeddings=args.rvq_codebook_size,
                embedding_dim=args.latent_dim,
                beta=args.rvq_beta,
                ema_decay=args.rvq_ema_decay,
                name="rvq_ema",
            )
        else:
            quantizer = ResidualVectorQuantizer(
                num_levels=args.rvq_num_levels,
                num_embeddings=args.rvq_codebook_size,
                embedding_dim=args.latent_dim,
                beta=args.rvq_beta,
                name="rvq",
            )
        logger.info(
            "RVQ%s levels=%d  K=%d  D=%d  bits/token=%.2f",
            "(EMA)" if args.rvq_ema else "",
            args.rvq_num_levels,
            args.rvq_codebook_size,
            args.latent_dim,
            bits_per_token,
        )

    enc, dec, model = build_ssm_autoencoder(
        frame_size=cfg.data.frame_size,
        latent_dim=effective_latent_dim,
        base_filters=args.base_filters,
        multiplier=args.multiplier,
        num_stages=num_stages,
        state_size=args.state_size,
        num_ssm_blocks=args.num_ssm_blocks,
        quantizer=quantizer,
    )

    cr_float32 = compute_compression_ratio(
        frame_size=cfg.data.frame_size,
        latent_dim=effective_latent_dim,
        num_stages=num_stages,
        input_bits=cfg.evaluation.input_bit_depth,
        latent_bits=32,
    )
    cr_int8 = compute_compression_ratio(
        frame_size=cfg.data.frame_size,
        latent_dim=effective_latent_dim,
        num_stages=num_stages,
        input_bits=cfg.evaluation.input_bit_depth,
        latent_bits=8,
    )
    # True compression when a quantizer is present: bit-exact bits/token.
    cr_quant: float | None = None
    if bits_per_token is not None:
        latent_time = cfg.data.frame_size // (2**num_stages)
        in_bits = cfg.data.frame_size * cfg.evaluation.input_bit_depth
        out_bits = latent_time * bits_per_token
        cr_quant = in_bits / out_bits
    logger.info(
        "Model params: enc=%d, dec=%d, total=%d  |  CR float32=%.2fx, INT8=%.2fx, QUANT=%s",
        enc.count_params(),
        dec.count_params(),
        model.count_params(),
        cr_float32,
        cr_int8,
        f"{cr_quant:.2f}x" if cr_quant else "n/a",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            keras.optimizers.schedules.CosineDecayRestarts(
                args.learning_rate,
                first_decay_steps=200,
                t_mul=2.0,
                m_mul=1.0,
                alpha=0.01,
            )
            if args.cosine_restarts
            else args.learning_rate
        ),
        loss=_make_loss(args.derivative_weight),
        metrics=[
            keras.metrics.MeanSquaredError(name="mse"),
            keras.metrics.CosineSimilarity(name="cos", axis=-2),
            _prd_metric,
        ],
    )

    run_name = f"ecg_ssm_{cfg_path.stem}_{args.output_suffix}"
    out_dir = Path(args.results_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = 1 if args.smoke_test else args.epochs
    steps = 5 if args.smoke_test else args.steps_per_epoch
    val_steps = 2 if args.smoke_test else validation_steps
    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        verbose=2,
        callbacks=[
            keras.callbacks.CSVLogger(out_dir / "history.csv"),
            keras.callbacks.EarlyStopping(
                monitor="val_mse",
                mode="min",
                patience=15,
                restore_best_weights=True,
            ),
        ],
    )

    summary = {
        "run_name": run_name,
        "cfg_path": str(cfg_path),
        "frame_size": cfg.data.frame_size,
        "num_stages": num_stages,
        "latent_dim": args.latent_dim,
        "state_size": args.state_size,
        "num_ssm_blocks": args.num_ssm_blocks,
        "params": {
            "encoder": int(enc.count_params()),
            "decoder": int(dec.count_params()),
            "total": int(model.count_params()),
        },
        "compression_ratio": {
            "float32": cr_float32,
            "int8": cr_int8,
            "quantizer": cr_quant,
            "bits_per_token": bits_per_token,
        },
        "final": {k: float(v[-1]) for k, v in history.history.items()},
        "best_val_mse": float(min(history.history.get("val_mse", [float("inf")]))),
        "dataset_info": info,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    model.save(out_dir / "model.keras")
    logger.info("Saved results to %s", out_dir)
    logger.info("Best val_mse=%.6f  (golden RVQ 32x baseline ≈ 0.0188)", summary["best_val_mse"])


if __name__ == "__main__":
    main()
