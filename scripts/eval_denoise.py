#!/usr/bin/env python3
"""Evaluate denoising capability of a PPG RVQ codec.

Injects controlled noise into clean MESA signals, runs through the codec,
and measures how much noise is removed vs. original signal preserved.

Metrics:
    - Noise Reduction Ratio (NRR): ratio of input noise power to output noise power
    - Signal-to-Noise Improvement (SNI): output SNR - input SNR in dB
    - PRD vs clean: distortion relative to the noise-free original
    - HR preservation: HR MAE between clean original and codec output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

from compressionkit.evaluation.metrics import compute_signal_metrics
from compressionkit.preprocessing.augmentations import (
    add_baseline_wander,
    add_motion_artifact,
    PPGAugmenter,
    build_noise_bank_from_h5,
)


def load_model_from_run(run_dir: Path):
    """Load golden/denoise model from run directory."""
    config_path = run_dir / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    from compressionkit.models.rvq_autoencoder import build_rvq_autoencoder

    mcfg = cfg["model"]
    frame_size = cfg["data"]["frame_size"]

    _, _, _, model = build_rvq_autoencoder(
        frame_size=frame_size,
        embedding_dim=mcfg["embedding_dim"],
        latent_width=mcfg["latent_width"],
        base_filters=mcfg["base_filters"],
        multiplier=mcfg["multiplier"],
        num_levels=mcfg["num_levels"],
        beta=mcfg["beta"],
        num_stages=mcfg["num_stages"],
        encoder_block_norm=mcfg["encoder_block_norm"],
        encoder_head_norm=mcfg["encoder_head_norm"],
        decoder_block_norm=mcfg["decoder_block_norm"],
        decoder_head_norm=mcfg["decoder_head_norm"],
        use_ema=mcfg["use_ema"],
        ema_decay=mcfg["ema_decay"],
    )

    dummy = np.zeros((1, 1, frame_size, 1), dtype=np.float32)
    with tf.device("/CPU:0"):
        model(dummy, training=False)

    for name in ("best_model.weights.h5", "model.weights.h5"):
        path = run_dir / name
        if path.exists():
            model.load_weights(str(path))
            print(f"  Loaded weights: {name}")
            return model

    raise FileNotFoundError(f"No weights in {run_dir}")


def load_clean_windows(run_dir: Path, max_windows: int = 500, seed: int = 42) -> np.ndarray:
    """Load clean MESA validation windows from the model's cache."""
    config_path = run_dir / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    frame_size = cfg["data"]["frame_size"]

    # Load from validation TFRecords or use the sample CSVs
    import glob
    import pandas as pd

    sample_files = sorted(glob.glob(str(run_dir / "sample_*.csv")))
    if sample_files:
        windows = []
        for f in sample_files[:max_windows]:
            df = pd.read_csv(f)
            if "original" in df.columns:
                sig = df["original"].values.astype(np.float32)
                if len(sig) == frame_size:
                    windows.append(sig)
        if windows:
            return np.stack(windows)

    raise RuntimeError("Could not load clean windows from run directory")


def evaluate_denoising(
    model,
    clean_windows: np.ndarray,
    *,
    noise_type: str,
    snr_db: float,
    epsilon: float = 0.001,
    sample_rate: int = 64,
    noise_bank: np.ndarray | None = None,
    seed: int = 42,
) -> dict:
    """Evaluate denoising at a specific noise level.

    Args:
        model: Loaded Keras model.
        clean_windows: Array of shape ``(N, T)`` with clean signals.
        noise_type: One of 'gaussian', 'baseline_wander', 'motion', 'empirical'.
        snr_db: Target input SNR in dB.
        epsilon: LayerNorm epsilon.
        sample_rate: Sampling rate.
        noise_bank: Required for 'empirical' noise type.
        seed: Random seed.

    Returns:
        Dict with denoising metrics.
    """
    rng = np.random.default_rng(seed)
    n_windows, frame_size = clean_windows.shape

    # Generate noisy versions
    noisy_windows = np.empty_like(clean_windows)
    for i in range(n_windows):
        if noise_type == "gaussian":
            sig_power = np.mean(clean_windows[i] ** 2) + 1e-10
            noise_power = sig_power / (10 ** (snr_db / 10))
            noise = rng.standard_normal(frame_size).astype(np.float32)
            noise = noise * np.sqrt(noise_power / (np.mean(noise ** 2) + 1e-10))
            noisy_windows[i] = clean_windows[i] + noise
        elif noise_type == "baseline_wander":
            noisy_windows[i] = add_baseline_wander(
                clean_windows[i],
                sample_rate=sample_rate,
                amplitude_range=(0.1, 0.5),
                rng=rng,
            )
        elif noise_type == "motion":
            noisy_windows[i] = add_motion_artifact(
                clean_windows[i],
                sample_rate=sample_rate,
                snr_range=(snr_db, snr_db),
                rng=rng,
            )
        elif noise_type == "empirical" and noise_bank is not None:
            from compressionkit.preprocessing.augmentations import add_empirical_noise

            noisy_windows[i] = add_empirical_noise(
                clean_windows[i],
                noise_bank,
                snr_range=(snr_db, snr_db),
                rng=rng,
            )
        else:
            noisy_windows[i] = clean_windows[i]

    # Normalize (matching golden LayerNorm)
    noisy_means = noisy_windows.mean(axis=1, keepdims=True)
    noisy_stds = noisy_windows.std(axis=1, keepdims=True) + epsilon
    noisy_normed = (noisy_windows - noisy_means) / noisy_stds

    # Run through codec
    inputs = noisy_normed[:, np.newaxis, :, np.newaxis]
    with tf.device("/CPU:0"):
        recon_normed = np.asarray(model.predict(inputs, batch_size=64, verbose=0))
    recon_normed = recon_normed.reshape(n_windows, frame_size)

    # Denormalize reconstructions
    recon_raw = recon_normed * noisy_stds + noisy_means

    # Also normalize the clean signal for normalized-domain comparison
    clean_means = clean_windows.mean(axis=1, keepdims=True)
    clean_stds = clean_windows.std(axis=1, keepdims=True) + epsilon
    clean_normed = (clean_windows - clean_means) / clean_stds

    # Compute metrics
    # 1. Input noise (noisy vs clean)
    input_noise = noisy_windows - clean_windows
    input_noise_power = np.mean(input_noise ** 2, axis=1)

    # 2. Output noise (reconstruction vs clean)
    output_noise = recon_raw - clean_windows
    output_noise_power = np.mean(output_noise ** 2, axis=1)

    # 3. Noise Reduction Ratio
    valid_mask = (input_noise_power > 1e-8) & (output_noise_power > 1e-8)
    nrr = np.where(valid_mask, input_noise_power / (output_noise_power + 1e-10), 1.0)
    nrr_db = 10 * np.log10(nrr + 1e-10)

    # 4. SNR improvement
    sig_power = np.mean(clean_windows ** 2, axis=1) + 1e-10
    input_snr = 10 * np.log10(sig_power / (input_noise_power + 1e-10))
    output_snr = 10 * np.log10(sig_power / (output_noise_power + 1e-10))
    # Filter out inf/nan
    valid_snr = np.isfinite(input_snr) & np.isfinite(output_snr)
    sni = np.where(valid_snr, output_snr - input_snr, 0.0)

    # 5. Signal metrics (codec output vs clean original)
    sig_metrics = compute_signal_metrics(clean_windows, recon_raw)

    # 6. Passthrough baseline (no codec, just noisy)
    passthrough_metrics = compute_signal_metrics(clean_windows, noisy_windows)

    return {
        "noise_type": noise_type,
        "target_snr_db": float(snr_db),
        "num_windows": int(n_windows),
        "input_prd_percent": float(passthrough_metrics["prd_percent"]),
        "output_prd_percent": float(sig_metrics["prd_percent"]),
        "prd_improvement": float(passthrough_metrics["prd_percent"] - sig_metrics["prd_percent"]),
        "input_snr_db": float(np.nanmean(input_snr[valid_snr])) if valid_snr.any() else 0.0,
        "output_snr_db": float(np.nanmean(output_snr[valid_snr])) if valid_snr.any() else 0.0,
        "snr_improvement_db": float(np.nanmean(sni[valid_snr])) if valid_snr.any() else 0.0,
        "noise_reduction_ratio_db": float(np.nanmean(nrr_db[valid_mask])) if valid_mask.any() else 0.0,
        "output_cosine_similarity": float(sig_metrics["cosine_similarity"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPG denoising codec")
    parser.add_argument("--run-dir", type=Path, required=True, help="Model run directory")
    parser.add_argument("--max-windows", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print(f"Loading model from {args.run_dir}...")
    model = load_model_from_run(args.run_dir)

    with open(args.run_dir / "config.json") as f:
        cfg = json.load(f)
    sample_rate = cfg["data"]["sampling_rate"]
    epsilon = cfg["data"]["epsilon"]
    frame_size = cfg["data"]["frame_size"]

    # Load clean windows
    print("Loading clean windows...")
    clean = load_clean_windows(args.run_dir, max_windows=args.max_windows)
    print(f"  Loaded {len(clean)} clean windows ({frame_size} samples)")

    # Build noise bank
    import glob

    dalia_files = sorted(glob.glob("/home/vscode/datasets/ppg_dalia/*.h5"))
    wesad_files = sorted(glob.glob("/home/vscode/datasets/wesad/*.h5"))
    noise_bank = build_noise_bank_from_h5(
        dalia_files + wesad_files, window_size=frame_size, max_segments=1000,
    )
    print(f"  Noise bank: {len(noise_bank)} segments")

    # Evaluate at multiple noise levels and types
    results: list[dict] = []
    noise_configs = [
        ("gaussian", 5.0),
        ("gaussian", 10.0),
        ("gaussian", 15.0),
        ("gaussian", 20.0),
        ("motion", 5.0),
        ("motion", 10.0),
        ("motion", 15.0),
        ("baseline_wander", 10.0),
        ("empirical", 10.0),
        ("empirical", 15.0),
    ]

    print("\nDenoising evaluation:")
    print(f"{'Type':<16} {'In SNR':>8} {'Out SNR':>8} {'SNI':>8} {'In PRD%':>8} {'Out PRD%':>8} {'Improv':>8}")
    print("-" * 72)

    for noise_type, snr_db in noise_configs:
        r = evaluate_denoising(
            model,
            clean,
            noise_type=noise_type,
            snr_db=snr_db,
            epsilon=epsilon,
            sample_rate=sample_rate,
            noise_bank=noise_bank if noise_type == "empirical" else None,
        )
        results.append(r)
        print(
            f"{noise_type:<16} {r['input_snr_db']:>8.1f} {r['output_snr_db']:>8.1f} "
            f"{r['snr_improvement_db']:>+8.1f} {r['input_prd_percent']:>8.2f} "
            f"{r['output_prd_percent']:>8.2f} {r['prd_improvement']:>+8.2f}"
        )

    # Save results
    output_path = args.output or (args.run_dir / "denoise_eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
