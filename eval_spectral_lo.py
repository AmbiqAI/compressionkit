#!/usr/bin/env python3
"""Large-N paired evaluation: golden vs spectral_lo for ECG RVQ.

Loads both models for each CR, evaluates on the SAME validation samples,
and runs paired statistical tests.
"""

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from scipy.stats import wilcoxon

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.datasets.ecg import collect_random_samples
from compressionkit.preprocessing.ecg import build_preprocessor, build_augmenter
from compressionkit.trainers.ecg_rvq import build_datasets, build_model

NUM_SAMPLES = 500
SEED = 42
ECG_CRS = ["02x", "04x", "08x", "16x", "32x", "64x"]


def evaluate_model(model: keras.Model, inputs: np.ndarray) -> np.ndarray:
    """Run inference and return per-sample MSE."""
    recon = model.predict(inputs, verbose=0)
    # Shape: (N, 1, T, C) or (N, T, C)
    diff = (inputs - recon).reshape(inputs.shape[0], -1)
    per_sample_mse = np.mean(diff**2, axis=1)
    return per_sample_mse


def evaluate_cosine(model: keras.Model, inputs: np.ndarray) -> np.ndarray:
    """Run inference and return per-sample cosine similarity."""
    recon = model.predict(inputs, verbose=0)
    x = inputs.reshape(inputs.shape[0], -1)
    y = recon.reshape(recon.shape[0], -1)
    dot = np.sum(x * y, axis=1)
    nx = np.linalg.norm(x, axis=1)
    ny = np.linalg.norm(y, axis=1)
    return dot / (nx * ny + 1e-12)


def main():
    results = {}

    for cr in ECG_CRS:
        golden_dir = Path(f"results/ecg_rvq_256hz_{cr}_golden")
        spectral_dir = Path(f"results/ecg_rvq_256hz_{cr}_golden_spectral_lo")

        if not golden_dir.exists() or not spectral_dir.exists():
            print(f"\n{cr}: skipping (missing results dir)")
            continue

        g_cfg_path = golden_dir / "config.json"
        s_cfg_path = spectral_dir / "config.json"
        g_weights = golden_dir / "best_model.weights.h5"
        s_weights = spectral_dir / "best_model.weights.h5"

        if not g_weights.exists() or not s_weights.exists():
            print(f"\n{cr}: skipping (missing weights)")
            continue

        print(f"\n{'='*60}")
        print(f"ECG {cr}")
        print(f"{'='*60}")

        # Load golden config and build val dataset
        g_cfg = EcgRvqConfig.model_validate_json(g_cfg_path.read_text())
        preprocessor = build_preprocessor(
            frame_size=g_cfg.data.frame_size, epsilon=g_cfg.data.epsilon,
        )
        augmenter = build_augmenter(
            aug_cfg=g_cfg.data.augmentation,
            sample_rate=g_cfg.data.effective_sample_rate,
        )

        _, val_ds, _, _ = build_datasets(g_cfg, preprocessor, augmenter)

        # Collect samples
        rng = np.random.default_rng(SEED)
        sample_inputs, sample_targets = collect_random_samples(val_ds, NUM_SAMPLES, rng)
        print(f"Collected {len(sample_inputs)} samples, shape={sample_inputs.shape}")

        # Build & load golden model
        g_model = build_model(g_cfg)
        dummy = np.zeros((1,) + sample_inputs.shape[1:], dtype=np.float32)
        g_model(dummy, training=False)
        g_model.load_weights(str(g_weights))

        # Build & load spectral_lo model
        s_cfg = EcgRvqConfig.model_validate_json(s_cfg_path.read_text())
        s_model = build_model(s_cfg)
        s_model(dummy, training=False)
        s_model.load_weights(str(s_weights))

        # Evaluate both on same samples
        g_mse = evaluate_model(g_model, sample_inputs)
        s_mse = evaluate_model(s_model, sample_inputs)
        g_cos = evaluate_cosine(g_model, sample_inputs)
        s_cos = evaluate_cosine(s_model, sample_inputs)

        # Paired test
        diff_mse = s_mse - g_mse  # negative = spectral better
        wins_mse = int(np.sum(diff_mse < 0))
        stat_mse, p_mse = wilcoxon(g_mse, s_mse)

        diff_cos = s_cos - g_cos  # positive = spectral better
        wins_cos = int(np.sum(diff_cos > 0))
        stat_cos, p_cos = wilcoxon(g_cos, s_cos)

        n = len(g_mse)
        gm = float(np.mean(g_mse))
        sm = float(np.mean(s_mse))
        gc = float(np.mean(g_cos))
        sc = float(np.mean(s_cos))
        delta_mse_pct = (sm - gm) / gm * 100

        print(f"  N = {n}")
        print(f"  Golden  MSE: {gm:.6f}  Cos: {gc:.6f}")
        print(f"  Spec_lo MSE: {sm:.6f}  Cos: {sc:.6f}")
        print(f"  Δ MSE: {delta_mse_pct:+.1f}%")
        print(f"  MSE wins: {wins_mse}/{n}  Wilcoxon p={p_mse:.4f}  {'***' if p_mse < 0.001 else '**' if p_mse < 0.01 else '*' if p_mse < 0.05 else 'ns'}")
        print(f"  Cos wins: {wins_cos}/{n}  Wilcoxon p={p_cos:.4f}  {'***' if p_cos < 0.001 else '**' if p_cos < 0.01 else '*' if p_cos < 0.05 else 'ns'}")

        results[cr] = {
            "n": n,
            "golden_mse": gm, "spectral_mse": sm, "delta_mse_pct": delta_mse_pct,
            "golden_cos": gc, "spectral_cos": sc,
            "mse_wins": wins_mse, "mse_p": p_mse,
            "cos_wins": wins_cos, "cos_p": p_cos,
        }

        # Free memory
        del g_model, s_model
        keras.backend.clear_session()

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY (N={NUM_SAMPLES} paired samples per CR)")
    print(f"{'='*80}")
    print(f"{'CR':<5} | {'G_MSE':>10} | {'S_MSE':>10} | {'Δ MSE':>8} | {'wins':>7} | {'p_mse':>8} | {'G_Cos':>8} | {'S_Cos':>8} | {'p_cos':>8} | {'sig':>4}")
    print("-" * 95)
    for cr, r in results.items():
        sig = "***" if r["mse_p"] < 0.001 else "**" if r["mse_p"] < 0.01 else "*" if r["mse_p"] < 0.05 else "ns"
        print(
            f"{cr:<5} | {r['golden_mse']:>10.6f} | {r['spectral_mse']:>10.6f} | "
            f"{r['delta_mse_pct']:>+7.1f}% | {r['mse_wins']:>3}/{r['n']:<3} | {r['mse_p']:>8.4f} | "
            f"{r['golden_cos']:>8.5f} | {r['spectral_cos']:>8.5f} | {r['cos_p']:>8.4f} | {sig:>4}"
        )


if __name__ == "__main__":
    main()
