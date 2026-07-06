"""Pre-build a NOISE-AUGMENTED training token cache for the ECG entropy prior.

The RVQ codec itself is trained with Gaussian-noise augmentation (see each
golden config's ``data.augmentation.gaussian_noise``) on windows drawn from
its own pre-built TFRecord cache (``data.cache``). The entropy prior should
see the SAME data distribution and augmentation the codec was trained on, not
a hand-rolled approximation — so this script reuses the trainer's own
``build_datasets()`` (which transparently reuses the existing TFRecord cache
instead of re-extracting from raw H5 files -- no wasted cycles) and just
takes the already-preprocessed, already-augmented model input batches it
yields, running them through the frozen encoder + VQ to get tokens.

Writes to the exact cache path ``scripts/measure_rvq_entropy.py`` expects
(``<run_dir>/entropy_prior/_token_cache/ecg_train_n<N>_maxNone_fs<F>_l<L>_li<LI>.npy``),
so a subsequent normal ``measure_rvq_entropy.py --num-train-files N`` run
transparently picks up the augmented tokens for training while validation
extraction stays clean (matching how the codec itself is validated).

Usage::

    uv run python3 scripts/build_augmented_ecg_train_cache.py \\
        --run-dir results/ecg_rvq_256hz_08x_golden --num-train-files 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from measure_rvq_entropy import _load_compressor

from compressionkit.preprocessing.ecg import build_augmenter, build_preprocessor
from compressionkit.trainers.ecg_rvq import build_datasets


def _encode_batch(compressor, x_aug: np.ndarray, *, num_leads: int) -> np.ndarray:
    """Encode one already-augmented ``(B, 1, frame_size, num_leads)`` batch -> tokens."""
    with tf.device("/CPU:0"):
        z = compressor.encoder(x_aug, training=False)
    z_np = np.asarray(z)
    latent_shape = z_np.shape
    with tf.device("/CPU:0"):
        indices_list = compressor.vq.encode(z_np)
    tokens_per_frame = int(np.prod(latent_shape[:-1])) // latent_shape[0]
    chunk_tokens = np.zeros((latent_shape[0], tokens_per_frame, len(indices_list)), dtype=np.int16)
    for level, idx in enumerate(indices_list):
        idx_np = np.asarray(idx).reshape(latent_shape[0], tokens_per_frame).astype(np.int16)
        chunk_tokens[..., level] = idx_np
    return chunk_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--num-train-files",
        type=int,
        default=800,
        help="Only used to name the cache file to match measure_rvq_entropy.py's convention.",
    )
    parser.add_argument(
        "--num-frames", type=int, default=7200, help="Number of augmented frames to draw from the training pipeline."
    )
    parser.add_argument(
        "--overwrite-default-cache",
        action="store_true",
        help="DANGEROUS: write to the exact filename measure_rvq_entropy.py's --num-train-files expects, "
        "so it's picked up transparently. This silently shadows the clean cache for ALL future runs at that "
        "file count until manually cleared -- forgetting this is exactly what caused a bad comparison earlier "
        "in this repo's history. Default OFF: writes to a clearly-suffixed '.augmented.npy' file instead.",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    print(f"[1/3] Loading compressor + config from {run_dir} ...", file=sys.stderr)
    compressor, cfg, modality = _load_compressor(run_dir, modality="ecg")
    data = cfg.data
    frame_size = int(data.frame_size)
    num_leads = int(getattr(data, "num_leads", 1) or 1)
    lead_index = int(getattr(data, "lead_index", 1) or 1)

    print(f"      gaussian_noise range: {data.augmentation.gaussian_noise}", file=sys.stderr)
    preprocessor = build_preprocessor(frame_size, epsilon=data.epsilon)
    augmenter = build_augmenter(data.augmentation, sample_rate=int(data.effective_sample_rate))

    print("[2/3] Building train dataset (reuses existing TFRecord cache if present) ...", file=sys.stderr)
    train_ds, _val_ds, _validation_steps, info = build_datasets(cfg, preprocessor, augmenter)
    print(f"      dataset info: mode={info['mode']} cache_dir={info.get('cache_dir')}", file=sys.stderr)

    print("[3/3] Encoding augmented batches through frozen RVQ encoder ...", file=sys.stderr)
    token_chunks: list[np.ndarray] = []
    n_collected = 0
    for x_aug, _target in train_ds:
        chunk_tokens = _encode_batch(compressor, np.asarray(x_aug), num_leads=num_leads)
        token_chunks.append(chunk_tokens)
        n_collected += chunk_tokens.shape[0]
        if n_collected >= args.num_frames:
            break
    tokens = np.concatenate(token_chunks, axis=0)[: args.num_frames]
    print(f"      collected {tokens.shape[0]} augmented frames", file=sys.stderr)

    cache_dir = run_dir / "entropy_prior" / "_token_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"ecg_train_n{args.num_train_files}_maxNone_fs{frame_size}_l{num_leads}_li{lead_index}"
    if args.overwrite_default_cache:
        out_path = cache_dir / f"{base_name}.npy"
        print(
            "      WARNING: writing to the default clean-cache filename -- this will shadow clean "
            "extraction for every future measure_rvq_entropy.py run at this file count until cleared.",
            file=sys.stderr,
        )
    else:
        out_path = cache_dir / f"{base_name}.augmented.npy"
    np.save(out_path, tokens)
    print(f"wrote {out_path}  shape={tokens.shape}", file=sys.stderr)


if __name__ == "__main__":
    main()
