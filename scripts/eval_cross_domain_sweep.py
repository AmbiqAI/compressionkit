#!/usr/bin/env python3
"""Unified cross-domain evaluation for both single-stream RVQ and two-stream codecs.

Evaluates on a fixed set of windows from ALL sources (MESA, BIDMC, DaLiA, WESAD)
for a fair apples-to-apples comparison.

Usage:
    # Single-stream (golden or unified RVQ)
    python scripts/eval_cross_domain_sweep.py \
        --run-dir results/ppg_rvq_64hz_04x_golden \
        --model-type single-stream

    # Two-stream
    python scripts/eval_cross_domain_sweep.py \
        --run-dir results/ppg_two_stream_04x_unified_all \
        --model-type two-stream

    # Batch sweep
    python scripts/eval_cross_domain_sweep.py --sweep
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

with contextlib.suppress(RuntimeError):
    tf.config.set_visible_devices([], "GPU")

from compressionkit.configs.paths import default_datasets_dir
from compressionkit.evaluation.metrics import compute_signal_metrics

# ---------------------------------------------------------------------------
# Fixed eval sources — same windows for every model
# ---------------------------------------------------------------------------

_DATASETS_ROOT = default_datasets_dir()

H5_SOURCES = {
    "bidmc": {
        "root": f"{_DATASETS_ROOT}/bidmc",
        "glob": "*.h5",
        "native_fs": 125,
        "signal_key": "data",
    },
    "ppg_dalia": {
        "root": f"{_DATASETS_ROOT}/ppg_dalia",
        "glob": "*.h5",
        "native_fs": 64,
        "signal_key": "data",
    },
    "wesad": {
        "root": f"{_DATASETS_ROOT}/wesad",
        "glob": "*.h5",
        "native_fs": 64,
        "signal_key": "data",
    },
}

MESA_CACHE = f"{_DATASETS_ROOT}/ppg_cache/mesa/val.tfrecord"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _resample(signal: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return signal.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    gcd = np.gcd(int(fs_in), int(fs_out))
    return resample_poly(signal, fs_out // gcd, fs_in // gcd).astype(np.float32, copy=False)


def collect_h5_windows(
    source_name: str,
    *,
    target_fs: int = 64,
    window_size: int = 320,
    max_windows_per_file: int = 50,
    max_total: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Collect evaluation windows from an H5 source (deterministic)."""
    source_cfg = H5_SOURCES[source_name]
    root = Path(source_cfg["root"])
    files = sorted(root.glob(source_cfg["glob"]))
    if not files:
        return np.empty((0, window_size), dtype=np.float32)

    rng = np.random.default_rng(seed)
    all_windows: list[np.ndarray] = []

    for path in files:
        try:
            with h5py.File(path, "r") as h5:
                if "quality" in h5.attrs and int(h5.attrs["quality"]) == 0:
                    continue
                signal = h5[source_cfg["signal_key"]]
                if signal.ndim == 2:
                    signal = signal[0]
                else:
                    signal = signal[:]
                signal = signal.astype(np.float32)
        except (OSError, KeyError):
            continue

        signal = _resample(signal, source_cfg["native_fs"], target_fs)
        if len(signal) < window_size:
            continue

        hop = window_size
        n_windows = max(0, (len(signal) - window_size) // hop + 1)
        if n_windows == 0:
            continue

        windows = np.stack([signal[i * hop : i * hop + window_size] for i in range(n_windows)]).astype(np.float32)

        # Quality filter
        stds = windows.std(axis=1)
        windows = windows[stds > 1e-4]

        if len(windows) > max_windows_per_file:
            idx = rng.choice(len(windows), max_windows_per_file, replace=False)
            windows = windows[idx]

        if len(windows) > 0:
            all_windows.append(windows)

    if not all_windows:
        return np.empty((0, window_size), dtype=np.float32)

    combined = np.concatenate(all_windows, axis=0)
    if len(combined) > max_total:
        idx = rng.choice(len(combined), max_total, replace=False)
        combined = combined[idx]

    return combined


def collect_mesa_windows(
    *,
    frame_size: int = 320,
    max_total: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Load MESA validation windows from TFRecord cache."""
    spec = {"signal": tf.io.FixedLenFeature([frame_size], tf.float32)}
    ds = tf.data.TFRecordDataset(MESA_CACHE)
    ds = ds.map(lambda raw: tf.io.parse_single_example(raw, spec)["signal"])
    ds = ds.batch(2048)

    windows_list: list[np.ndarray] = []
    for batch in ds:
        windows_list.append(batch.numpy())
    all_windows = np.concatenate(windows_list, axis=0)

    # Subsample deterministically
    rng = np.random.default_rng(seed)
    if len(all_windows) > max_total:
        idx = rng.choice(len(all_windows), max_total, replace=False)
        all_windows = all_windows[idx]

    return all_windows


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_single_stream_model(run_dir: Path) -> tf.keras.Model:
    """Load a single-stream RVQ autoencoder."""
    from compressionkit.models.rvq_autoencoder import build_rvq_autoencoder

    config_path = run_dir / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

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
            return model
    raise FileNotFoundError(f"No weights in {run_dir}")


def load_two_stream_models(run_dir: Path) -> tuple:
    """Load two-stream baseline + pulsatile models."""
    from compressionkit.configs.ppg_two_stream import PpgTwoStreamConfig
    from compressionkit.models.ppg_two_stream import build_baseline_model, build_pulsatile_model

    config_path = run_dir / "config.json"
    with open(config_path) as f:
        cfg_dict = json.load(f)
    cfg = PpgTwoStreamConfig.model_validate(cfg_dict)

    baseline_model, _ = build_baseline_model(cfg)
    pulsatile_model, _ = build_pulsatile_model(cfg)

    # Build with dummy inputs
    bm_cfg = cfg.baseline_model
    baseline_len = cfg.data.frame_size // bm_cfg.downsample_factor
    dummy_bl = np.zeros((1, 1, baseline_len, 1), dtype=np.float32)
    dummy_pl = np.zeros((1, 1, cfg.data.frame_size, 1), dtype=np.float32)
    with tf.device("/CPU:0"):
        baseline_model(dummy_bl, training=False)
        pulsatile_model(dummy_pl, training=False)

    # Load weights
    bl_path = run_dir / "best_baseline.weights.h5"
    pl_path = run_dir / "best_pulsatile.weights.h5"
    if bl_path.exists():
        baseline_model.load_weights(str(bl_path))
    if pl_path.exists():
        pulsatile_model.load_weights(str(pl_path))

    return baseline_model, pulsatile_model, cfg


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_single_stream(model, windows: np.ndarray, epsilon: float = 0.001) -> np.ndarray:
    """Run single-stream model: normalize → predict → denormalize."""
    means = windows.mean(axis=1, keepdims=True)
    stds = np.sqrt(((windows - means) ** 2).mean(axis=1, keepdims=True) + epsilon)
    normed = (windows - means) / stds

    inputs = normed[:, np.newaxis, :, np.newaxis]
    with tf.device("/CPU:0"):
        recon_normed = np.asarray(model.predict(inputs, batch_size=64, verbose=0))
    recon_normed = recon_normed.reshape(windows.shape)

    return recon_normed * stds + means


def predict_two_stream(
    baseline_model,
    pulsatile_model,
    windows: np.ndarray,
    cfg,
) -> np.ndarray:
    """Run two-stream: decompose → encode/decode both → reconstruct."""
    from compressionkit.preprocessing.two_stream import (
        decompose_and_normalize,
        downsample_baseline,
        reconstruct_from_streams,
        upsample_baseline,
    )

    data = cfg.data
    decompose_cfg = data.decompose
    bm = cfg.baseline_model
    recons = np.empty_like(windows)

    # Batch the predictions for efficiency
    batch_size = 64
    n = len(windows)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = windows[start:end]

        baselines_ds = []
        pulsatiles = []
        decomp_params = []

        for sig in batch:
            result = decompose_and_normalize(
                sig,
                sample_rate=data.sampling_rate,
                baseline_cutoff_hz=decompose_cfg.baseline_cutoff_hz,
                order=decompose_cfg.filter_order,
                epsilon=decompose_cfg.epsilon,
            )
            baseline_ds = downsample_baseline(
                result["baseline_norm"],
                factor=bm.downsample_factor,
            )
            baselines_ds.append(baseline_ds)
            pulsatiles.append(result["pulsatile_norm"])
            decomp_params.append(result)

        # Batch predict baseline
        bl_arr = np.array(baselines_ds)[:, np.newaxis, :, np.newaxis]
        with tf.device("/CPU:0"):
            bl_recon = np.asarray(baseline_model.predict(bl_arr, batch_size=batch_size, verbose=0))
        bl_recon = bl_recon.reshape(bl_arr.shape[0], -1)

        # Batch predict pulsatile
        pl_arr = np.array(pulsatiles)[:, np.newaxis, :, np.newaxis]
        with tf.device("/CPU:0"):
            pl_recon = np.asarray(pulsatile_model.predict(pl_arr, batch_size=batch_size, verbose=0))
        pl_recon = pl_recon.reshape(pl_arr.shape[0], -1)

        # Reconstruct each
        for i in range(end - start):
            params = decomp_params[i]
            bl_up = upsample_baseline(
                bl_recon[i],
                factor=bm.downsample_factor,
                target_len=data.frame_size,
            )
            recon = reconstruct_from_streams(
                bl_up,
                pl_recon[i],
                baseline_center=params["baseline_center"],
                baseline_scale=params["baseline_scale"],
                pulsatile_center=params["pulsatile_center"],
                pulsatile_scale=params["pulsatile_scale"],
            )
            recons[start + i] = recon

    return recons


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model_on_windows(
    predict_fn,
    windows: np.ndarray,
) -> dict:
    """Compute metrics given a prediction function and raw windows."""
    if len(windows) == 0:
        return {"error": "no windows", "prd_percent": float("nan"), "cosine_similarity": float("nan")}

    recons = predict_fn(windows)
    metrics = compute_signal_metrics(windows, recons)
    return {
        "num_windows": len(windows),
        "prd_percent": round(metrics["prd_percent"], 4),
        "cosine_similarity": round(metrics["cosine_similarity"], 6),
        "mse": round(metrics["mse"], 8),
        "rmse": round(metrics["rmse"], 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def evaluate_run(
    run_dir: Path,
    model_type: str,
    *,
    max_windows: int = 2000,
    seed: int = 42,
) -> dict:
    """Evaluate a single model run on all sources."""
    run_dir = Path(run_dir)
    print(f"  Loading model from {run_dir.name} ({model_type})...")

    # Load config to get frame_size/epsilon
    with open(run_dir / "config.json") as f:
        cfg_json = json.load(f)

    if model_type == "single-stream":
        frame_size = cfg_json["data"]["frame_size"]
        epsilon = cfg_json["data"].get("epsilon", 0.001)
        model = load_single_stream_model(run_dir)

        def predict_fn(windows):
            return predict_single_stream(model, windows, epsilon=epsilon)

    elif model_type == "two-stream":
        frame_size = cfg_json["data"]["frame_size"]
        baseline_model, pulsatile_model, cfg = load_two_stream_models(run_dir)

        def predict_fn(windows):
            return predict_two_stream(baseline_model, pulsatile_model, windows, cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    results = {"run_name": run_dir.name, "model_type": model_type, "sources": {}}

    # Evaluate on each H5 source
    for source_name in ["bidmc", "ppg_dalia", "wesad"]:
        print(f"    {source_name}...", end=" ", flush=True)
        windows = collect_h5_windows(
            source_name,
            target_fs=64,
            window_size=frame_size,
            max_total=max_windows,
            seed=seed,
        )
        metrics = evaluate_model_on_windows(predict_fn, windows)
        results["sources"][source_name] = metrics
        print(f"PRD={metrics['prd_percent']:.2f}%, cos={metrics['cosine_similarity']:.4f}")

    # Evaluate on MESA
    print("    mesa...", end=" ", flush=True)
    mesa_windows = collect_mesa_windows(
        frame_size=frame_size,
        max_total=max_windows,
        seed=seed,
    )
    mesa_metrics = evaluate_model_on_windows(predict_fn, mesa_windows)
    results["sources"]["mesa"] = mesa_metrics
    print(f"PRD={mesa_metrics['prd_percent']:.2f}%, cos={mesa_metrics['cosine_similarity']:.4f}")

    return results


def run_sweep():
    """Run evaluation on all golden and two-stream models."""
    runs = [
        # Golden single-stream (MESA-only)
        ("results/ppg_rvq_64hz_02x_golden", "single-stream"),
        ("results/ppg_rvq_64hz_04x_golden", "single-stream"),
        ("results/ppg_rvq_64hz_08x_golden", "single-stream"),
        ("results/ppg_rvq_64hz_16x_golden", "single-stream"),
        # Unified single-stream (all sources)
        ("results/ppg_rvq_64hz_02x_unified_all", "single-stream"),
        ("results/ppg_rvq_64hz_04x_unified_all", "single-stream"),
        ("results/ppg_rvq_64hz_08x_unified_all", "single-stream"),
        ("results/ppg_rvq_64hz_16x_unified_all", "single-stream"),
        # Two-stream unified (all sources)
        ("results/ppg_two_stream_02x_unified_all", "two-stream"),
        ("results/ppg_two_stream_04x_unified_all", "two-stream"),
        ("results/ppg_two_stream_08x_unified_all", "two-stream"),
        ("results/ppg_two_stream_16x_unified_all", "two-stream"),
    ]

    all_results = []
    for run_dir, model_type in runs:
        if not Path(run_dir).exists():
            print(f"  SKIP (not found): {run_dir}")
            continue
        print(f"\n{'=' * 60}")
        print(f"  {run_dir}")
        print(f"{'=' * 60}")
        result = evaluate_run(Path(run_dir), model_type)

        # Get CR from config
        with open(Path(run_dir) / "config.json") as f:
            cfg = json.load(f)
        if model_type == "two-stream":
            cr = cfg.get("compression_stats", {}).get("compression_ratio")
            if cr is None:
                # Compute from two-stream report if available
                report_path = Path(run_dir) / "two_stream_report.json"
                if report_path.exists():
                    with open(report_path) as f:
                        report = json.load(f)
                    cr = report.get("compression_stats", {}).get("compression_ratio")
        else:
            # Single-stream: compute from model config
            mcfg = cfg["model"]
            frame_size = cfg["data"]["frame_size"]
            raw_bits = frame_size * 16
            latent_pos = frame_size
            for _ in range(mcfg["num_stages"]):
                latent_pos //= 2
            compressed_bits = latent_pos * mcfg["num_levels"] * 8
            cr = raw_bits / compressed_bits

        result["compression_ratio"] = round(cr, 2) if cr else None
        all_results.append(result)

    # Print summary table
    print(f"\n\n{'=' * 80}")
    print("CROSS-DOMAIN SWEEP SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Model':<42} {'CR':>5} {'BIDMC':>7} {'DaLiA':>7} {'WESAD':>7} {'MESA':>7} {'Avg':>7}")
    print("-" * 80)

    for r in all_results:
        name = r["run_name"]
        cr = r.get("compression_ratio", "?")
        bidmc = r["sources"].get("bidmc", {}).get("prd_percent", float("nan"))
        dalia = r["sources"].get("ppg_dalia", {}).get("prd_percent", float("nan"))
        wesad = r["sources"].get("wesad", {}).get("prd_percent", float("nan"))
        mesa = r["sources"].get("mesa", {}).get("prd_percent", float("nan"))
        avg = np.nanmean([bidmc, dalia, wesad, mesa])
        print(f"{name:<42} {cr:>5} {bidmc:>7.2f} {dalia:>7.2f} {wesad:>7.2f} {mesa:>7.2f} {avg:>7.2f}")

    print("-" * 80)

    # Save JSON
    out_path = Path("results/cross_domain_sweep_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Cross-domain evaluation sweep")
    parser.add_argument("--sweep", action="store_true", help="Run full sweep comparison")
    parser.add_argument("--run-dir", type=Path, help="Single run directory")
    parser.add_argument("--model-type", choices=["single-stream", "two-stream"])
    parser.add_argument("--max-windows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    elif args.run_dir:
        if not args.model_type:
            # Auto-detect
            if (args.run_dir / "best_baseline.weights.h5").exists():
                args.model_type = "two-stream"
            else:
                args.model_type = "single-stream"
        result = evaluate_run(args.run_dir, args.model_type, max_windows=args.max_windows, seed=args.seed)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved to {args.output}")
    else:
        parser.error("Specify --sweep or --run-dir")


if __name__ == "__main__":
    main()
