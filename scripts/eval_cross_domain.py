#!/usr/bin/env python3
"""Cross-domain evaluation: run golden MESA-trained model on DaLiA/BIDMC/WESAD data.

Quantifies the generalization gap by evaluating a model trained on one dataset
(MESA sleep PSG) against windows from other PPG sources (wrist, ICU, lab).
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

from compressionkit.evaluation.metrics import (
    compute_signal_metrics,
    summarize_physiokit_alignment,
)
from compressionkit.models.rvq_autoencoder import build_rvq_autoencoder

# ---------------------------------------------------------------------------
# Data loading from h5 datasets
# ---------------------------------------------------------------------------

H5_SOURCES = {
    "bidmc": {
        "root": "/home/vscode/datasets/bidmc",
        "glob": "*.h5",
        "native_fs": 125,
        "signal_key": "data",
    },
    "ppg_dalia": {
        "root": "/home/vscode/datasets/ppg_dalia",
        "glob": "*.h5",
        "native_fs": 64,
        "signal_key": "data",
    },
    "wesad": {
        "root": "/home/vscode/datasets/wesad",
        "glob": "*.h5",
        "native_fs": 64,
        "signal_key": "data",
    },
}


def _resample(signal: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return signal.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    gcd = np.gcd(int(fs_in), int(fs_out))
    return resample_poly(signal, fs_out // gcd, fs_in // gcd).astype(np.float32, copy=False)


def _load_signal(path: Path, source_cfg: dict, target_fs: int) -> np.ndarray | None:
    """Load and resample a single h5 file's PPG signal."""
    try:
        with h5py.File(path, "r") as h5:
            # BUTPPG quality check
            if "quality" in h5.attrs and int(h5.attrs["quality"]) == 0:
                return None
            signal = h5[source_cfg["signal_key"]]
            if signal.ndim == 2:
                signal = signal[0]
            else:
                signal = signal[:]
            signal = signal.astype(np.float32)
    except (OSError, KeyError):
        return None

    fs_in = source_cfg["native_fs"]
    return _resample(signal, fs_in, target_fs)


def _extract_windows(
    signal: np.ndarray,
    window_size: int,
    hop: int | None = None,
    max_windows: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Extract non-overlapping (or hopped) windows from a signal."""
    if hop is None:
        hop = window_size
    n_windows = max(0, (len(signal) - window_size) // hop + 1)
    if n_windows == 0:
        return np.empty((0, window_size), dtype=np.float32)

    windows = np.stack([signal[i * hop : i * hop + window_size] for i in range(n_windows)]).astype(np.float32)

    # Quality filter: reject flat/saturated/extreme windows
    stds = windows.std(axis=1)
    valid = stds > 1e-4
    windows = windows[valid]

    if max_windows is not None and len(windows) > max_windows:
        if rng is None:
            rng = np.random.default_rng(42)
        idx = rng.choice(len(windows), max_windows, replace=False)
        windows = windows[idx]

    return windows


def collect_source_windows(
    source_name: str,
    *,
    target_fs: int = 64,
    window_size: int = 320,
    max_windows_per_file: int = 50,
    max_total: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Collect evaluation windows from a single h5 source."""
    source_cfg = H5_SOURCES[source_name]
    root = Path(source_cfg["root"])
    files = sorted(root.glob(source_cfg["glob"]))
    if not files:
        print(f"  WARNING: No files found in {root}")
        return np.empty((0, window_size), dtype=np.float32)

    rng = np.random.default_rng(seed)
    all_windows: list[np.ndarray] = []

    for path in files:
        signal = _load_signal(path, source_cfg, target_fs)
        if signal is None or len(signal) < window_size:
            continue
        windows = _extract_windows(signal, window_size, max_windows=max_windows_per_file, rng=rng)
        if len(windows) > 0:
            all_windows.append(windows)

    if not all_windows:
        return np.empty((0, window_size), dtype=np.float32)

    combined = np.concatenate(all_windows, axis=0)
    if len(combined) > max_total:
        idx = rng.choice(len(combined), max_total, replace=False)
        combined = combined[idx]

    return combined


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_golden_model(run_dir: Path) -> tf.keras.Model:
    """Load the golden model from a results directory."""
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

    # Build model with dummy input
    dummy = np.zeros((1, 1, frame_size, 1), dtype=np.float32)
    with tf.device("/CPU:0"):
        model(dummy, training=False)

    # Load weights
    for name in ("best_model.weights.h5", "model.weights.h5"):
        path = run_dir / name
        if path.exists():
            model.load_weights(str(path))
            print(f"  Loaded weights: {name}")
            return model

    raise FileNotFoundError(f"No weights in {run_dir}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def normalize_windows(windows: np.ndarray, epsilon: float = 0.001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-window layer normalization matching golden training."""
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True) + epsilon
    normed = (windows - means) / stds
    return normed.astype(np.float32), means.astype(np.float32), stds.astype(np.float32)


def evaluate_on_source(
    model: tf.keras.Model,
    windows: np.ndarray,
    *,
    epsilon: float = 0.001,
    sample_rate: int = 64,
    batch_size: int = 64,
) -> dict:
    """Run golden model on source windows and compute metrics."""
    if len(windows) == 0:
        return {"error": "no windows available"}

    # Normalize (matching golden LayerNorm preprocessing)
    normed, means, stds = normalize_windows(windows, epsilon=epsilon)

    # Shape for model: (B, 1, T, 1)
    inputs = normed[:, np.newaxis, :, np.newaxis]

    # Predict
    with tf.device("/CPU:0"):
        recon_normed = np.asarray(model.predict(inputs, batch_size=batch_size, verbose=0))

    # Reshape: (B, 1, T, 1) -> (B, T)
    recon_normed = recon_normed.reshape(windows.shape)

    # Denormalize
    recon_raw = recon_normed * stds.squeeze(1)[:, np.newaxis] + means.squeeze(1)[:, np.newaxis]
    # Wait - we need to be careful. The golden model was trained with input=target=normalized.
    # So we should compare normalized targets vs normalized reconstructions AND
    # also denormalized for real-world metrics.

    # Signal metrics on normalized (what the model was trained on)
    targets_normed = normed
    recons_normed = recon_normed

    # Signal metrics on raw (real-world fidelity)
    targets_raw = windows
    recons_raw = recon_raw

    normed_metrics = compute_signal_metrics(targets_normed, recons_normed)
    raw_metrics = compute_signal_metrics(targets_raw, recons_raw)

    # Physiokit alignment (on raw signals - for HR/HRV preservation)
    physiokit, per_sample = summarize_physiokit_alignment(
        targets_raw,
        recons_raw,
        sample_rate=sample_rate,
        low_hz=0.5,
        high_hz=8.0,
        order=3,
        min_peaks=3,
    )

    return {
        "num_windows": len(windows),
        "normalized_metrics": normed_metrics,
        "raw_metrics": raw_metrics,
        "physiokit": physiokit,
    }


def evaluate_long_recording(
    model: tf.keras.Model,
    signal: np.ndarray,
    *,
    frame_size: int = 320,
    hop_ratio: float = 0.5,
    epsilon: float = 0.001,
    sample_rate: int = 64,
    batch_size: int = 64,
) -> dict:
    """Overlap-add reconstruction of a long signal."""
    from compressionkit.evaluation.stitching import _frame_positions, _hann_window

    hop = max(1, int(frame_size * hop_ratio))
    positions = _frame_positions(len(signal), frame_size, hop)
    if not positions:
        return {"error": "signal too short"}

    # Extract and normalize frames
    frames = np.stack([signal[start : start + frame_size] for start in positions]).astype(np.float32)
    means = frames.mean(axis=1, keepdims=True)
    stds = frames.std(axis=1, keepdims=True) + epsilon
    normed = ((frames - means) / stds).astype(np.float32)

    # Predict
    inputs = normed[:, np.newaxis, :, np.newaxis]
    with tf.device("/CPU:0"):
        recon = np.asarray(model.predict(inputs, batch_size=batch_size, verbose=0))
    recon = recon.reshape(normed.shape)

    # Denormalize
    recon_denorm = recon * stds + means

    # Overlap-add
    window = _hann_window(frame_size).astype(np.float64)
    out = np.zeros(len(signal), dtype=np.float64)
    norm = np.zeros(len(signal), dtype=np.float64)
    for idx, start in enumerate(positions):
        out[start : start + frame_size] += window * recon_denorm[idx].astype(np.float64)
        norm[start : start + frame_size] += window
    covered = norm > 1e-8
    out[covered] /= norm[covered]
    out[~covered] = signal[~covered]
    reconstructed = out.astype(np.float32)

    # Metrics
    from compressionkit.evaluation.metrics import compute_ppg_physiokit_metrics

    sig_metrics = compute_signal_metrics(signal, reconstructed)
    orig_pk = compute_ppg_physiokit_metrics(
        signal,
        sample_rate=sample_rate,
        low_hz=0.5,
        high_hz=8.0,
        order=3,
        min_peaks=10,
    )
    recon_pk = compute_ppg_physiokit_metrics(
        reconstructed,
        sample_rate=sample_rate,
        low_hz=0.5,
        high_hz=8.0,
        order=3,
        min_peaks=10,
    )

    result = {"signal_metrics": sig_metrics}
    if orig_pk is not None and recon_pk is not None:
        result["hr_error_bpm"] = abs(recon_pk["hr_bpm"] - orig_pk["hr_bpm"])
        result["sdnn_error_ms"] = abs(recon_pk["sdnn_ms"] - orig_pk["sdnn_ms"])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Cross-domain eval of golden model")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/ppg_rvq_64hz_04x_golden"),
        help="Golden model run directory",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["bidmc", "ppg_dalia", "wesad"],
        help="H5 sources to evaluate on",
    )
    parser.add_argument("--max-windows", type=int, default=2000, help="Max windows per source")
    parser.add_argument("--long-duration-sec", type=float, default=60.0, help="Long recording duration")
    parser.add_argument("--num-long-recordings", type=int, default=10, help="Num long recordings per source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading golden model from {args.run_dir}...")
    model = load_golden_model(args.run_dir)

    # Get frame size from config
    with open(args.run_dir / "config.json") as f:
        cfg = json.load(f)
    frame_size = cfg["data"]["frame_size"]
    sample_rate = cfg["data"]["sampling_rate"]
    epsilon = cfg["data"]["epsilon"]

    print(f"  frame_size={frame_size}, sample_rate={sample_rate}Hz")
    print()

    results: dict[str, dict] = {}

    for source in args.sources:
        if source not in H5_SOURCES:
            print(f"  SKIP: unknown source '{source}'")
            continue

        print(f"{'=' * 60}")
        print(f"Source: {source}")
        print(f"{'=' * 60}")

        # Short-window evaluation
        print(f"  Collecting windows (frame_size={frame_size})...")
        windows = collect_source_windows(
            source,
            target_fs=sample_rate,
            window_size=frame_size,
            max_total=args.max_windows,
            seed=args.seed,
        )
        print(f"  Collected {len(windows)} windows")

        short_metrics = evaluate_on_source(model, windows, epsilon=epsilon, sample_rate=sample_rate)

        # Long-recording evaluation
        source_cfg = H5_SOURCES[source]
        root = Path(source_cfg["root"])
        files = sorted(root.glob(source_cfg["glob"]))
        target_len = int(args.long_duration_sec * sample_rate)
        rng = np.random.default_rng(args.seed)

        long_results: list[dict] = []
        for path in files:
            if len(long_results) >= args.num_long_recordings:
                break
            signal = _load_signal(path, source_cfg, sample_rate)
            if signal is None or len(signal) < target_len:
                continue
            # Random offset
            start = int(rng.integers(0, len(signal) - target_len + 1))
            segment = signal[start : start + target_len]
            # Quality check
            if segment.std() < 1e-4:
                continue
            lr = evaluate_long_recording(
                model,
                segment,
                frame_size=frame_size,
                epsilon=epsilon,
                sample_rate=sample_rate,
            )
            lr["file"] = path.name
            long_results.append(lr)

        # Summarize long recordings
        long_summary = None
        if long_results:
            valid_prd = [r["signal_metrics"]["prd_percent"] for r in long_results if "signal_metrics" in r]
            valid_hr = [r["hr_error_bpm"] for r in long_results if "hr_error_bpm" in r]
            long_summary = {
                "num_recordings": len(long_results),
                "num_with_hr": len(valid_hr),
                "prd_percent_mean": float(np.mean(valid_prd)) if valid_prd else None,
                "prd_percent_median": float(np.median(valid_prd)) if valid_prd else None,
                "hr_mae_bpm": float(np.mean(valid_hr)) if valid_hr else None,
            }

        results[source] = {
            "short_window": short_metrics,
            "long_recording": long_summary,
        }

        # Print summary
        if "error" not in short_metrics:
            rm = short_metrics["raw_metrics"]
            nm = short_metrics["normalized_metrics"]
            print(f"  Short-window (N={short_metrics['num_windows']}):")
            print(f"    Normalized PRD: {nm['prd_percent']:.2f}%")
            print(f"    Raw PRD:        {rm['prd_percent']:.2f}%")
            print(f"    Raw cosine:     {rm['cosine_similarity']:.4f}")
            pk = short_metrics.get("physiokit")
            if pk:
                print(f"    HR MAE:         {pk['hr_mae_bpm']:.2f} bpm")
        if long_summary:
            print(f"  Long-recording ({long_summary['num_recordings']} recs):")
            if long_summary["prd_percent_mean"] is not None:
                print(f"    PRD:    {long_summary['prd_percent_mean']:.2f}%")
            if long_summary["hr_mae_bpm"] is not None:
                print(f"    HR MAE: {long_summary['hr_mae_bpm']:.2f} bpm")
        print()

    # Summary comparison
    print("=" * 60)
    print("CROSS-DOMAIN GENERALIZATION SUMMARY")
    print("=" * 60)
    print(f"{'Source':<12} {'PRD%':>8} {'Cos':>8} {'HR MAE':>8}  {'LR PRD%':>8}")
    print("-" * 52)
    for src, r in results.items():
        sw = r["short_window"]
        lr = r["long_recording"]
        if "error" in sw:
            print(f"{src:<12} {'N/A':>8}")
            continue
        rm = sw["raw_metrics"]
        pk = sw.get("physiokit")
        hr_str = f"{pk['hr_mae_bpm']:.2f}" if pk else "N/A"
        lr_str = f"{lr['prd_percent_mean']:.2f}" if lr and lr.get("prd_percent_mean") else "N/A"
        print(f"{src:<12} {rm['prd_percent']:>8.2f} {rm['cosine_similarity']:>8.4f} {hr_str:>8}  {lr_str:>8}")

    # Compare with golden in-domain
    print("-" * 52)
    summary_path = args.run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            gs = json.load(f)
        gm = gs["metrics"]["best"]
        glr = gs["metrics"].get("long_recording_physiokit", {})
        hr_lr = f"{glr.get('hr_mae_bpm', 'N/A'):.2f}" if isinstance(glr.get("hr_mae_bpm"), (int, float)) else "N/A"
        print(f"{'MESA(in-dom)':<12} {gm['val_prd']:>8.2f} {gm['val_cos']:>8.4f} {'':>8}  {hr_lr:>8}")

    # Save results
    output_path = args.output or (args.run_dir / "cross_domain_eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
