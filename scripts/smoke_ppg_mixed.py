"""Mixed-source PPG smoke test.

Audits sanitize reject rates and runs a tiny training loop over a mixture of
canonical h5 PPG datasets to verify the loader + sanitizer pipeline. Not a real
training recipe — the goal is end-to-end shape and stability.

Usage::

    python scripts/smoke_ppg_mixed.py
    python scripts/smoke_ppg_mixed.py --sources bidmc ppg_dalia --steps 25
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import keras
import tensorflow as tf

from compressionkit.datasets.ppg_h5 import (
    DEFAULT_DATASET_ROOT,
    PpgH5Source,
    WindowSpec,
    make_h5_ppg_dataset,
    summarize_sources,
)
from compressionkit.models.rvq_autoencoder import (
    build_rvq_autoencoder,
    compute_compression_stats,
)

DEFAULT_SOURCES = ("bidmc", "ppg_dalia", "wesad", "butppg")


def _build_tiny_autoencoder(window_samples: int) -> keras.Model:
    """Conv1D autoencoder, 1×N → 1×N, ~5k params. Smoke test only."""
    inp = keras.Input(shape=(window_samples, 1), name="ppg")
    x = keras.layers.Conv1D(16, 5, padding="same", activation="relu")(inp)
    x = keras.layers.MaxPooling1D(2, padding="same")(x)
    x = keras.layers.Conv1D(8, 5, padding="same", activation="relu")(x)
    x = keras.layers.UpSampling1D(2)(x)
    x = keras.layers.Conv1D(1, 5, padding="same")(x)
    model = keras.Model(inp, x, name="ppg_smoke_ae")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def _build_rvq(window_samples: int, target_fs: int, log: logging.Logger) -> keras.Model:
    """Real RVQ autoencoder using the same builder as the production recipe."""
    latent_width = 256
    num_levels = 2
    num_stages = 2  # 4× temporal downsample
    encoder, bottleneck, decoder, model = build_rvq_autoencoder(
        frame_size=window_samples,
        embedding_dim=8,
        latent_width=latent_width,
        in_ch=1,
        out_ch=1,
        base_filters=16,
        multiplier=1.25,
        num_stages=num_stages,
        num_levels=num_levels,
        beta=0.25,
        use_ema=True,
    )
    stats = compute_compression_stats(
        frame_size=window_samples,
        bit_depth=16,
        latent_width=latent_width,
        num_levels=num_levels,
        downsample_factor=2**num_stages,
    )
    log.info("RVQ compression stats: %s", json.dumps(stats, indent=2, default=str))
    log.info("sampling rate (Hz): %d", target_fs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3))
    return model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT, help="Parent dir holding per-dataset h5 folders.")
    p.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES), help="Dataset slugs to mix.")
    p.add_argument("--target-fs", type=int, default=64)
    p.add_argument("--window-seconds", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument(
        "--model", choices=["tiny", "rvq"], default="tiny", help="'tiny' Conv1D AE (default) or real RVQ autoencoder."
    )
    p.add_argument(
        "--audit-files-per-source", type=int, default=5, help="Cap files-per-source for the reject-rate audit."
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
        stream=sys.stderr,
    )
    log = logging.getLogger("smoke")
    keras.utils.set_random_seed(args.seed)

    sources = [PpgH5Source(slug=s, root=args.root) for s in args.sources]
    for src in sources:
        n = len(src.files())
        if n == 0:
            log.warning("source %s: no h5 files at %s/%s — skipping", src.slug, src.root, src.slug)
    sources = [s for s in sources if s.files()]
    if not sources:
        log.error("no sources have h5 files; aborting")
        sys.exit(1)

    spec = WindowSpec(
        target_fs=args.target_fs,
        window_seconds=args.window_seconds,
    )

    log.info("auditing reject rates (max %d files/source) …", args.audit_files_per_source)
    audit = summarize_sources(sources, spec, max_files_per_source=args.audit_files_per_source)
    log.info("audit: %s", json.dumps(audit, indent=2, default=str))

    log.info(
        "building tf.data pipeline @ fs=%d, win=%d samples, batch=%d",
        spec.target_fs,
        spec.window_samples,
        args.batch_size,
    )
    ds = make_h5_ppg_dataset(sources, spec, batch_size=args.batch_size, seed=args.seed)

    # Self-supervised reconstruction target = input. Loader yields (B, 1, T);
    # the conv1d model expects (B, T, 1).
    def _to_xy(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.transpose(x, (0, 2, 1))
        return x, x

    ds_xy = ds.map(_to_xy)

    sample = next(iter(ds_xy))
    log.info("first batch: x=%s y=%s", tuple(sample[0].shape), tuple(sample[1].shape))

    if args.model == "rvq":
        model = _build_rvq(spec.window_samples, spec.target_fs, log)
        # 2D conv encoder expects (B, 1, T, 1); loader yields (B, 1, T).
        train_ds = ds.map(lambda x: tf.expand_dims(x, -1))
        model.fit(train_ds.take(args.steps), epochs=1, verbose=2)
    else:
        model = _build_tiny_autoencoder(spec.window_samples)
        model.summary(print_fn=log.info)
        model.fit(ds_xy.take(args.steps), epochs=1, verbose=2)
    log.info("smoke test complete")


if __name__ == "__main__":
    main()
