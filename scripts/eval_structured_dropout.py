"""Evaluate a unified structured-dropout ECG RVQ model at each level count.

Loads a trained model, runs inference at each configured dropout level
(e.g., 1, 2, 4, 8 active RVQ levels), and reports per-level MSE/PRD/cosine.

Usage:
    python scripts/eval_structured_dropout.py --results-dir results/ecg_rvq_256hz_unified_sd
"""

from __future__ import annotations

import argparse
import json
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from pathlib import Path

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.trainers.ecg_rvq import build_model

logger = logging.getLogger(__name__)


def prd_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage Root-mean-square Difference."""
    num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    den = np.sqrt(np.mean(y_true**2))
    return 100.0 * num / (den + 1e-12)


def cosine_sim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean cosine similarity across batch."""
    a = y_true.reshape(y_true.shape[0], -1)
    b = y_pred.reshape(y_pred.shape[0], -1)
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return float(np.mean(dot / (norm_a * norm_b + 1e-12)))


def evaluate_at_levels(model, vq_layer, val_data: np.ndarray, num_levels: int) -> dict:
    """Run model with exactly `num_levels` active RVQ stages."""
    encoder = model.encoder
    decoder = model.decoder

    z = encoder(val_data, training=False)
    zq, _indices = vq_layer.call_at_level(z, num_levels=num_levels)
    y_pred = decoder(zq, training=False)

    y_true = val_data
    y_pred_np = np.array(y_pred)
    y_true_np = np.array(y_true)

    mse = float(np.mean((y_true_np - y_pred_np) ** 2))
    prd = prd_percent(y_true_np, y_pred_np)
    cos = cosine_sim(y_true_np, y_pred_np)

    # Compute effective CR
    frame_size = y_true_np.shape[2]
    ds_factor = 2 ** int(np.log2(frame_size // (keras.ops.shape(z)[2])))
    latent_positions = frame_size // ds_factor
    bits_per_frame = latent_positions * num_levels * 8  # 8 bits per index (K=256)
    raw_bits = frame_size * 16  # 16-bit input
    cr = raw_bits / bits_per_frame

    return {
        "num_levels": num_levels,
        "cr": cr,
        "mse": mse,
        "prd_percent": prd,
        "cosine_sim": cos,
        "bits_per_frame": bits_per_frame,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate structured dropout model at each level")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results_dir = Path(args.results_dir)
    config_path = results_dir / "config.json"
    weights_path = results_dir / "best_model.weights.h5"

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {results_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"No best_model.weights.h5 in {results_dir}")

    cfg = EcgRvqConfig.model_validate_json(config_path.read_text())
    model = build_model(cfg)

    # Build model with a dummy forward pass
    dummy = np.zeros((1, 1, cfg.data.frame_size, 1), dtype=np.float32)
    model(dummy, training=False)
    model.load_weights(str(weights_path))
    logger.info("Loaded weights from %s", weights_path)

    # Load validation data from cache
    from compressionkit.trainers.ecg_rvq import build_datasets

    _, val_ds = build_datasets(cfg)

    # Collect validation samples
    val_samples = []
    for batch in val_ds:
        if isinstance(batch, dict):
            x = batch["data"]
        elif isinstance(batch, tuple):
            x, _ = batch
        else:
            x = batch
        val_samples.append(np.array(x))
        if sum(s.shape[0] for s in val_samples) >= args.num_samples:
            break
    val_data = np.concatenate(val_samples, axis=0)[: args.num_samples]
    logger.info("Loaded %d validation samples", val_data.shape[0])

    # Get the VQ layer
    vq_layer = model.vq

    # Determine levels to evaluate
    levels_to_eval = cfg.model.dropout_levels or [2**i for i in range(cfg.model.num_levels.bit_length()) if 2**i <= cfg.model.num_levels]

    print("\n" + "=" * 80)
    print("STRUCTURED DROPOUT EVALUATION — Unified Model at Each Level")
    print("=" * 80)
    print(f"{'Levels':>8} | {'CR':>6} | {'MSE':>10} | {'PRD%':>8} | {'Cosine':>8} | {'Bits/frame':>10}")
    print("-" * 80)

    results = []
    for n_levels in sorted(levels_to_eval):
        r = evaluate_at_levels(model, vq_layer, val_data, n_levels)
        results.append(r)
        print(
            f"{r['num_levels']:>8} | {r['cr']:>6.1f}× | {r['mse']:>10.6f} | "
            f"{r['prd_percent']:>7.2f}% | {r['cosine_sim']:>8.5f} | {r['bits_per_frame']:>10}"
        )

    print("=" * 80)

    # Save results
    out_path = results_dir / "structured_dropout_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
