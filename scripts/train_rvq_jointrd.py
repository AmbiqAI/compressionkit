"""Warm-start joint rate-distortion fine-tuning of an existing RVQ run.

Loads a frozen-checkpoint compressor (encoder + RVQ + decoder) and an
optional pretrained CNN prior, wraps them in
``RateDistortionVQAutoencoder``, and fine-tunes for a small number of
epochs with the joint loss::

    L = D + λ · R

Reuses the dataset / preprocessing pipeline from
``compressionkit.trainers.ecg_rvq`` so the comparison vs the baseline
two-stage result is apples-to-apples.

After fine-tuning, the script:

* writes the updated weights into ``<output-dir>/best_model.weights.h5``,
* writes the prior weights into ``<output-dir>/prior.weights.h5``,
* calls ``scripts/measure_rvq_entropy.py`` is *not* invoked here — run
  it separately on the fine-tuned ``output-dir`` to read off the new
  bits/token under whichever prior you'd like to evaluate against.

Example::

    uv run python scripts/train_rvq_jointrd.py \\
        --run-dir results/ecg_rvq_256hz_32x_golden \\
        --output-dir results/ecg_rvq_256hz_32x_golden_jointrd_lambda01 \\
        --epochs 30 --rate-weight 0.01

The script intentionally keeps the schema small: it does **not** add new
fields to ``EcgRvqConfig`` — joint-R-D parameters live on the CLI so the
existing config snapshot in ``run_dir/config.json`` remains the source
of truth for the underlying compressor.
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

logger = logging.getLogger("rvq-jointrd")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_compressor(
    cfg: EcgRvqConfig, run_dir: Path | None, *, load_weights: bool = True,
) -> keras.Model:
    from compressionkit.trainers.ecg_rvq import build_model

    if cfg.model.num_levels != 1:
        raise ValueError(
            f"Joint R-D fine-tuning only supports num_levels==1; got {cfg.model.num_levels}"
        )
    model = build_model(cfg)
    dummy = np.zeros(
        (1, 1, cfg.data.frame_size, max(1, cfg.data.num_leads or 1)),
        dtype=np.float32,
    )
    # Build via XLA-compiled tf.function: DepthwiseConv2D with stride=(1,2)
    # only works under XLA on GPU.  Variables get created on the same
    # device as the call (GPU when available) so fit() can use them.
    @tf.function(jit_compile=True)
    def _build_call(x):
        return model(x, training=False)
    _build_call(dummy)
    for name in ("best_model.weights.h5", "model.weights.h5"):
        p = run_dir / name if run_dir is not None else None
        if p is not None and p.exists():
            model.load_weights(p)
            logger.info("Loaded compressor weights from %s", p)
            break
    else:
        if load_weights:
            raise FileNotFoundError(f"No weights in {run_dir}")
        logger.info("From-scratch mode: compressor initialised randomly.")
    return model


def _build_cnn_prior(
    *,
    vocab_size: int,
    context_length: int | None,
    embed_dim: int,
    num_layers: int,
    kernel_size: int,
    dropout: float = 0.0,
    name: str = "rvq_cnn_prior",
) -> keras.Model:
    """Causal dilated Conv1D prior — duplicate of measure_rvq_entropy.py.

    ``context_length=None`` builds a variable-length prior (``shape=(None,)``),
    which is what we want for joint training where each forward pass sees
    per-frame token sequences (length ``tokens_per_frame``) rather than the
    full context window the prior was originally trained on.  Conv1D with
    causal padding has length-independent weights, so weights from a
    fixed-length pretrained prior load cleanly into this model.
    """
    seq_len = context_length  # may be None for variable length
    tokens_in = keras.Input(shape=(seq_len,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim, name="token_embedding",
    )(tokens_in)
    for i in range(num_layers):
        x = keras.layers.Conv1D(
            filters=embed_dim,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=2 ** i,
            activation="relu",
            name=f"causal_conv_d{2 ** i}",
        )(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"drop_d{2 ** i}")(x)
    x = keras.layers.LayerNormalization(epsilon=1e-5, name="final_ln")(x)
    logits = keras.layers.Dense(vocab_size, name="lm_head")(x)
    return keras.Model(tokens_in, logits, name=name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, default=None,
                     help="Existing trained RVQ run directory to fine-tune.")
    src.add_argument("--config", type=Path, default=None,
                     help="YAML config for from-scratch joint R-D training. "
                          "No warm-start — encoder/decoder/codebook/prior all init random.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write the fine-tuned weights and metadata.")
    parser.add_argument("--prior-weights", type=Path, default=None,
                        help="Optional pretrained CNN-prior weights (.weights.h5). "
                             "If omitted, the prior is initialised from scratch.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Lower than the original training LR — this is fine-tuning.")
    parser.add_argument("--rate-weight", type=float, default=0.01,
                        help="λ on the rate term (in units of bits/token).")
    parser.add_argument("--soft-temperature", type=float, default=1.0,
                        help="Temperature τ for soft codebook assignments.")
    parser.add_argument("--freeze-decoder", action="store_true",
                        help="Freeze the decoder during fine-tuning. Encoder + RVQ + prior only.")
    parser.add_argument("--freeze-prior", action="store_true",
                        help="Freeze the prior — only encoder + decoder + codebook learn.")
    # Prior architecture (default: matches deployed dilated CNN at 8s context @ 32x golden)
    parser.add_argument("--prior-context-frames", type=int, default=8,
                        help="Token context expressed in compressor frames.")
    parser.add_argument("--prior-embed-dim", type=int, default=64)
    parser.add_argument("--prior-num-layers", type=int, default=7,
                        help="Auto-clamped so receptive field >= context length.")
    parser.add_argument("--prior-kernel", type=int, default=5)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--train-frame-size", type=int, default=None,
                        help="Override frame_size for joint R-D training so each "
                             "training window spans multiple deployment frames "
                             "and the prior sees real cross-frame context. "
                             "Encoder/decoder are fully conv so weights apply natively. "
                             "For PTB-XL @ 256Hz, max ~2560; use 2048.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    keras.utils.set_random_seed(args.seed)

    out_dir: Path = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    from_scratch = args.config is not None

    # ------------------------------------------------------------------
    # 1. Load / build compressor
    # ------------------------------------------------------------------
    if from_scratch:
        cfg_path: Path = args.config.resolve()
        logger.info("Building compressor from scratch using config %s ...", cfg_path)
        cfg = EcgRvqConfig.from_yaml(str(cfg_path))
        run_dir = None
    else:
        run_dir = args.run_dir.resolve()
        logger.info("Loading compressor from %s ...", run_dir)
        cfg = EcgRvqConfig.model_validate_json((run_dir / "config.json").read_text())

    # Optional: train at a larger frame_size than the deployment frame.
    # The encoder/decoder are fully convolutional, so weights transfer
    # natively (we re-instantiate the model under the new InputSpec and
    # load weights into it).  Each training window then spans multiple
    # deployment frames, giving the prior real cross-frame context
    # inside a single forward pass.
    base_frame_size = cfg.data.frame_size
    if args.train_frame_size is not None and args.train_frame_size != base_frame_size:
        if args.train_frame_size % (2 ** cfg.model.num_stages) != 0:
            raise ValueError(
                f"--train-frame-size ({args.train_frame_size}) must be a multiple "
                f"of 2**num_stages ({2 ** cfg.model.num_stages})."
            )
        if args.train_frame_size % base_frame_size != 0:
            logger.warning(
                "train_frame_size %d is not an integer multiple of base frame_size %d; "
                "continuing but cross-frame alignment is non-trivial.",
                args.train_frame_size, base_frame_size,
            )
        cfg = cfg.model_copy(deep=True)
        cfg.data.frame_size = args.train_frame_size
        cfg.data.segment_samples = max(
            cfg.data.segment_samples,
            int(args.train_frame_size * cfg.data.cache.min_segment_scale),
        )
        logger.info(
            "Overriding training frame_size: %d -> %d (segment_samples=%d). "
            "This will trigger a new TFRecord cache build if not already present.",
            base_frame_size, args.train_frame_size, cfg.data.segment_samples,
        )

    compressor = _load_compressor(cfg, run_dir, load_weights=not from_scratch)

    from compressionkit.preprocessing.ecg import build_augmenter, build_preprocessor
    from compressionkit.trainers.ecg_rvq import build_datasets
    pre = build_preprocessor(frame_size=cfg.data.frame_size, epsilon=cfg.data.epsilon)
    aug = build_augmenter(
        aug_cfg=cfg.data.augmentation, sample_rate=cfg.data.effective_sample_rate,
    )

    # Keep the same preprocessing/aug that produced the baseline checkpoint.
    train_ds, val_ds, validation_steps, info = build_datasets(cfg, pre, aug)
    if args.validation_steps is not None:
        validation_steps = args.validation_steps
    logger.info("Datasets ready (%s); val steps=%s", info.get("mode"), validation_steps)

    # ------------------------------------------------------------------
    # 2. Build the prior
    # ------------------------------------------------------------------
    vocab_size = int(cfg.model.latent_width)
    tokens_per_frame = cfg.data.frame_size // (2 ** cfg.model.num_stages)
    context_length = args.prior_context_frames * tokens_per_frame
    # Clamp num_layers so dilated receptive field >= context, but only
    # when training the prior from scratch.  When loading pretrained
    # weights the architecture is fixed by the checkpoint, and a prior
    # with RF < context_length is still valid (it just attends to its
    # full RF, which is the same as inference behaviour).
    def rf(n: int) -> int:
        return 1 + (args.prior_kernel - 1) * (2 ** n - 1)

    n_layers = args.prior_num_layers
    if args.prior_weights is None:
        while rf(n_layers) < context_length and n_layers < 12:
            n_layers += 1
    logger.info(
        "Prior: vocab=%d ctx=%d (=%d frames × %d tok/frame), layers=%d, RF=%d",
        vocab_size, context_length, args.prior_context_frames, tokens_per_frame,
        n_layers, rf(n_layers),
    )
    prior = _build_cnn_prior(
        vocab_size=vocab_size,
        context_length=None,  # variable-length: takes per-frame token batches
        embed_dim=args.prior_embed_dim,
        num_layers=n_layers,
        kernel_size=args.prior_kernel,
    )
    if args.prior_weights and args.prior_weights.exists():
        prior.load_weights(args.prior_weights)
        logger.info("Loaded prior weights from %s", args.prior_weights)
    else:
        logger.info("Prior initialised from scratch (no weights provided).")

    # NOTE: the wrapper feeds the prior tokens of length T_lat (= tokens
    # per *frame*), not context_length — the prior's causal convolution
    # naturally handles whichever length is fed at runtime as long as
    # T_lat <= context_length receptive field.  Smoke-checked below.
    if tokens_per_frame > context_length:
        raise ValueError(
            f"Tokens per frame ({tokens_per_frame}) exceeds prior context "
            f"({context_length}); increase --prior-context-frames."
        )

    # ------------------------------------------------------------------
    # 3. Wrap with R-D objective
    # ------------------------------------------------------------------
    from compressionkit.trainers.rate_distortion import RateDistortionVQAutoencoder
    rd_model = RateDistortionVQAutoencoder(
        autoencoder=compressor,
        prior=prior,
        rate_weight=args.rate_weight,
        soft_assignment_temperature=args.soft_temperature,
        freeze_prior=args.freeze_prior,
    )

    if args.freeze_decoder:
        compressor.decoder.trainable = False
        logger.info("Decoder frozen; only encoder + RVQ codebook + prior train.")

    # The wrapper's compute_loss needs a base loss function on the inner
    # autoencoder; preserve the existing reconstruction loss + extras.
    from compressionkit.trainers.ecg_rvq import build_extra_losses
    extras = build_extra_losses(cfg)
    compressor.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError(name="mse")],
        extra_losses=extras or None,
    )

    rd_model.compile(optimizer=keras.optimizers.Adam(args.learning_rate))

    # Skip eager smoke test: some encoders use stride=(1,2) DepthwiseConv2D
    # which fails on GPU in eager mode but works under model.fit's XLA path.

    # ------------------------------------------------------------------
    # 4. Fine-tune
    # ------------------------------------------------------------------
    callbacks = [
        keras.callbacks.CSVLogger(str(out_dir / "history.csv"), append=False),
    ]
    history = rd_model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        validation_steps=validation_steps,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
        verbose=2,
    )

    # ------------------------------------------------------------------
    # 5. Persist
    # ------------------------------------------------------------------
    compressor.save_weights(out_dir / "best_model.weights.h5")
    prior.save_weights(out_dir / "prior.weights.h5")
    # Reuse the original config so downstream tooling
    # (measure_rvq_entropy.py, etc.) treats this as a normal RVQ run.
    # Restore the deployment frame_size if we trained at a larger window.
    cfg_to_save = cfg.model_copy(deep=True)
    if args.train_frame_size is not None and args.train_frame_size != base_frame_size:
        cfg_to_save.data.frame_size = base_frame_size
    (out_dir / "config.json").write_text(cfg_to_save.model_dump_json(indent=2))
    metadata = {
        "source_run": str(run_dir) if run_dir is not None else None,
        "source_config": str(args.config) if from_scratch else None,
        "from_scratch": from_scratch,
        "rate_weight": args.rate_weight,
        "soft_temperature": args.soft_temperature,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "freeze_decoder": args.freeze_decoder,
        "freeze_prior": args.freeze_prior,
        "prior_context_length": context_length,
        "prior_num_layers": n_layers,
        "prior_embed_dim": args.prior_embed_dim,
        "prior_kernel": args.prior_kernel,
        "final_loss": float(history.history.get("loss", [float("nan")])[-1]),
        "final_rate_bits": float(history.history.get("rate_bits", [float("nan")])[-1]),
        "final_distortion": float(history.history.get("distortion", [float("nan")])[-1]),
    }
    (out_dir / "jointrd_metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Wrote: %s", sorted(p.name for p in out_dir.iterdir()))
    logger.info("Done. Run scripts/measure_rvq_entropy.py against %s for final bpt.", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
