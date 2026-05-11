#!/usr/bin/env python3
"""Evaluate h5-backed mixed PPG RVQ runs with golden-style metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

from compressionkit.configs.ppg_h5_rvq import PpgH5RvqConfig
from compressionkit.datasets.ppg_h5 import (
    PpgH5Source,
    SplitConfig,
    WindowSpec,
    iter_windows,
    patient_split,
)
from compressionkit.evaluation.metrics import (
    compute_ppg_physiokit_metrics,
    compute_signal_metrics,
    summarize_physiokit_alignment,
)
from compressionkit.evaluation.noise import estimate_ppg_noise_floor
from compressionkit.evaluation.spectral_metrics import (
    PPG_DEFAULT_BANDS,
    PPG_DEFAULT_COHERENCE_BAND,
    PPG_DEFAULT_FREQ_WEIGHTS,
    psd_band_error,
    spectral_coherence,
    weighted_freq_prd,
)
from compressionkit.evaluation.stitching import (
    _frame_positions,
    _hann_window,
    seam_discontinuity_ratio,
)
from compressionkit.preprocessing.sanitize import SanitizeConfig, is_clean_window
from compressionkit.recipes.train_ppg_h5_rvq import build_model


def _sanitize_config(cfg: PpgH5RvqConfig) -> SanitizeConfig | None:
    if not cfg.data.sanitize.enabled:
        return None
    return SanitizeConfig(
        min_std=cfg.data.sanitize.min_std,
        max_saturation_frac=cfg.data.sanitize.max_saturation_frac,
        max_abs_z=cfg.data.sanitize.max_abs_z,
        max_outlier_frac=cfg.data.sanitize.max_outlier_frac,
    )


def _sources(cfg: PpgH5RvqConfig) -> list[PpgH5Source]:
    root = Path(cfg.data.root)
    return [
        PpgH5Source(
            slug=src.slug,
            root=root,
            glob=src.glob,
            butppg_quality_only=src.butppg_quality_only,
        )
        for src in cfg.data.sources
    ]


def _resample(signal: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return signal.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    gcd = np.gcd(int(fs_in), int(fs_out))
    return resample_poly(signal, fs_out // gcd, fs_in // gcd).astype(np.float32, copy=False)


def _patient_id(h5: h5py.File, fallback: str) -> str:
    pid = h5.attrs.get("patient_id", fallback)
    if isinstance(pid, bytes):
        pid = pid.decode()
    return str(pid)


def _skip_file(source: PpgH5Source, h5: h5py.File) -> bool:
    if source.slug == "butppg" and source.butppg_quality_only:
        return int(h5.attrs.get("quality", 1)) == 0
    return False


def _load_h5_signal(path: Path, source: PpgH5Source, target_fs: int) -> tuple[np.ndarray, str]:
    with h5py.File(path, "r") as h5:
        if _skip_file(source, h5):
            raise ValueError("quality-skipped file")
        fs_in = int(h5.attrs.get("fs", target_fs))
        pid = _patient_id(h5, path.stem)
        signal = h5["data"][0].astype(np.float32, copy=False)
    return _resample(signal, fs_in, target_fs), pid


def _load_model(cfg: PpgH5RvqConfig, run_dir: Path):
    model = build_model(cfg)
    dummy = np.zeros((1, 1, cfg.data.window_samples, 1), dtype=np.float32)
    with tf.device("/CPU:0"):
        model(dummy, training=False)
    for name in ("best_model.weights.h5", "model.weights.h5"):
        path = run_dir / name
        if path.exists():
            model.load_weights(path)
            return model, name
    raise FileNotFoundError(f"No model weights found under {run_dir}")


def _predict(model, batch: np.ndarray, *, batch_size: int) -> np.ndarray:
    with tf.device("/CPU:0"):
        return np.asarray(model.predict(batch, batch_size=batch_size, verbose=0))


def _compression_context(cfg: PpgH5RvqConfig) -> dict[str, float | int]:
    raw_bits_per_frame = cfg.data.window_samples * cfg.evaluation.input_bit_depth
    downsample_factor = 2 ** cfg.model.num_stages
    latent_positions = cfg.data.window_samples // downsample_factor
    bits_per_index = 8
    bits_per_frame = latent_positions * cfg.model.num_levels * bits_per_index
    return {
        "frame_size": cfg.data.window_samples,
        "sample_rate_hz": cfg.data.target_fs,
        "input_bit_depth": cfg.evaluation.input_bit_depth,
        "latent_positions": latent_positions,
        "downsample_factor": downsample_factor,
        "effective_sample_rate_hz": cfg.data.target_fs / downsample_factor,
        "num_levels": cfg.model.num_levels,
        "bits_per_index_uniform": bits_per_index,
        "uniform_bits_per_frame": bits_per_frame,
        "raw_bits_per_frame": raw_bits_per_frame,
        "uniform_compression_ratio": raw_bits_per_frame / bits_per_frame,
        "raw_bitrate_bps": cfg.data.target_fs * cfg.evaluation.input_bit_depth,
        "uniform_codec_bitrate_bps": bits_per_frame * cfg.data.target_fs / cfg.data.window_samples,
        "uniform_bits_per_sample": bits_per_frame / cfg.data.window_samples,
    }


def _aggregate(values: list[float]) -> dict[str, float | int]:
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if clean.size == 0:
        return {"n": 0}
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "median": float(np.median(clean)),
        "p90": float(np.percentile(clean, 90)),
    }


def _spectral_summary(targets: np.ndarray, recons: np.ndarray, *, sample_rate: int) -> dict[str, Any]:
    band_total: list[float] = []
    weighted_prd: list[float] = []
    coherence: list[float] = []
    per_band: dict[str, list[float]] = {}
    for target, recon in zip(targets, recons):
        band = psd_band_error(target, recon, fs=sample_rate, bands=PPG_DEFAULT_BANDS)
        band_total.append(float(band["band_total_rel_error"]))
        for key, value in band.items():
            if key.endswith("_rel_error"):
                per_band.setdefault(key, []).append(float(value))
        wf = weighted_freq_prd(target, recon, fs=sample_rate, weights=PPG_DEFAULT_FREQ_WEIGHTS)
        weighted_prd.append(float(wf["weighted_freq_prd_percent"]))
        coh = spectral_coherence(target, recon, fs=sample_rate, band=PPG_DEFAULT_COHERENCE_BAND)
        coherence.append(float(next(iter(coh.values()))))
    return {
        "band_total_rel_error": _aggregate(band_total),
        "per_band_rel_error": {key: _aggregate(vals) for key, vals in per_band.items()},
        "weighted_freq_prd_percent": _aggregate(weighted_prd),
        "coherence": _aggregate(coherence),
    }


def _noise_summary(signals: np.ndarray, *, sample_rate: int) -> tuple[list[dict[str, float]], dict[str, Any]]:
    per_signal = [estimate_ppg_noise_floor(sig, fs=sample_rate) for sig in signals]
    return per_signal, {
        "bp_noise_power": _aggregate([float(nf.get("bp_noise_power", 0.0)) for nf in per_signal]),
        "bp_noise_rms": _aggregate([float(nf.get("bp_noise_rms", 0.0)) for nf in per_signal]),
        "hf_noise_power": _aggregate([float(nf.get("hf_noise_power", 0.0)) for nf in per_signal]),
        "hf_noise_rms": _aggregate([float(nf.get("hf_noise_rms", 0.0)) for nf in per_signal]),
    }


def _write_sample_artifacts(
    run_dir: Path,
    targets: np.ndarray,
    recons: np.ndarray,
    *,
    sample_rate: int,
) -> None:
    for idx, (target, recon) in enumerate(zip(targets, recons)):
        path = run_dir / f"sample_{idx:03d}.csv"
        pd.DataFrame({
            "sample": np.arange(target.size, dtype=np.int32),
            "time_sec": np.arange(target.size, dtype=np.float32) / float(sample_rate),
            "original": target.astype(np.float32),
            "reconstructed": recon.astype(np.float32),
        }).to_csv(path, index=False)


def _purge_old_sample_artifacts(run_dir: Path) -> None:
    """Delete stale per-window CSV artifacts before refreshing metrics."""
    for csv_path in run_dir.glob("sample_*.csv"):
        csv_path.unlink()


def _collect_val_windows(
    cfg: PpgH5RvqConfig,
    *,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    spec = WindowSpec(
        target_fs=cfg.data.target_fs,
        window_seconds=cfg.data.window_seconds,
        hop_seconds=cfg.data.hop_seconds,
        sanitize=_sanitize_config(cfg),
        normalize=cfg.data.normalize,
    )
    split = SplitConfig(
        train_frac=cfg.data.split.train_frac,
        val_frac=cfg.data.split.val_frac,
        seed=cfg.data.split.seed,
        split="val",
    )
    rng = np.random.default_rng(seed)
    reservoir: list[np.ndarray] = []
    seen = 0
    for sample in iter_windows(_sources(cfg), spec, split):
        seen += 1
        window = sample["data"].astype(np.float32, copy=False)
        if len(reservoir) < num_samples:
            reservoir.append(window)
            continue
        idx = int(rng.integers(0, seen))
        if idx < num_samples:
            reservoir[idx] = window
    if not reservoir:
        raise RuntimeError("No validation windows available for evaluation")
    return np.stack(reservoir, axis=0)


def _robust_frame_stats(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.median(frames, axis=1).astype(np.float32)
    mad = np.median(np.abs(frames - med[:, None]), axis=1).astype(np.float32)
    scale = (1.4826 * mad + 1e-6).astype(np.float32)
    normed = np.clip((frames - med[:, None]) / scale[:, None], -6.0, 6.0)
    return normed.astype(np.float32), med, scale


def _reconstruct_overlap_add_h5_norm(
    model,
    signal: np.ndarray,
    *,
    frame_size: int,
    hop_ratio: float,
    batch_size: int,
) -> np.ndarray:
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    hop = max(1, int(frame_size * hop_ratio))
    positions = _frame_positions(len(sig), frame_size, hop)
    if not positions:
        return sig.copy()

    frames = np.stack([sig[start : start + frame_size] for start in positions]).astype(np.float32)
    normed, med, scale = _robust_frame_stats(frames)
    batch = normed[:, np.newaxis, :, np.newaxis]
    recon = _predict(model, batch, batch_size=batch_size).reshape(normed.shape)

    window = _hann_window(frame_size).astype(np.float64)
    out = np.zeros(len(sig), dtype=np.float64)
    norm = np.zeros(len(sig), dtype=np.float64)
    for idx, start in enumerate(positions):
        denorm = recon[idx].astype(np.float64) * float(scale[idx]) + float(med[idx])
        out[start : start + frame_size] += window * denorm
        norm[start : start + frame_size] += window
    covered = norm > 1e-8
    out[covered] /= norm[covered]
    out[~covered] = sig[~covered]
    return out.astype(np.float32)


def _evaluate_short_windows(
    model,
    cfg: PpgH5RvqConfig,
    *,
    run_dir: Path,
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    windows = _collect_val_windows(cfg, num_samples=num_samples, seed=seed)
    inputs = windows[..., np.newaxis]
    recon = _predict(model, inputs, batch_size=cfg.data.batch_size)
    targets = windows.reshape(windows.shape[0], -1)
    recons = recon.reshape(recon.shape[0], -1)
    noise_per_sample, noise = _noise_summary(targets, sample_rate=cfg.data.target_fs)
    mean_noise_power = float(np.mean([nf.get("bp_noise_power", 0.0) for nf in noise_per_sample]))
    signal_metrics = compute_signal_metrics(targets, recons, noise_power=mean_noise_power)
    _write_sample_artifacts(run_dir, targets, recons, sample_rate=cfg.data.target_fs)
    physiokit, per_sample = summarize_physiokit_alignment(
        targets,
        recons,
        sample_rate=cfg.data.target_fs,
        low_hz=0.5,
        high_hz=8.0,
        order=3,
        min_peaks=5,
    )
    return {
        "num_requested": num_samples,
        "num_collected": int(windows.shape[0]),
        "signal": signal_metrics,
        "noise": noise,
        "spectral": _spectral_summary(targets, recons, sample_rate=cfg.data.target_fs),
        "physiokit": physiokit,
        "physiokit_valid_pairs": 0 if physiokit is None else physiokit["num_valid_pairs"],
        "physiokit_total_pairs": len(per_sample),
    }


def _iter_val_recordings(cfg: PpgH5RvqConfig) -> list[tuple[PpgH5Source, Path, str]]:
    recordings: list[tuple[PpgH5Source, Path, str]] = []
    for src in _sources(cfg):
        for path in src.files():
            try:
                with h5py.File(path, "r") as h5:
                    if _skip_file(src, h5):
                        continue
                    pid = _patient_id(h5, path.stem)
            except OSError:
                continue
            bucket = patient_split(
                pid,
                source=src.slug,
                train_frac=cfg.data.split.train_frac,
                val_frac=cfg.data.split.val_frac,
                seed=cfg.data.split.seed,
            )
            if bucket == "val":
                recordings.append((src, path, pid))
    return recordings


def _evaluate_long_recordings(
    model,
    cfg: PpgH5RvqConfig,
    *,
    duration_sec: float,
    hop_ratio: float,
    num_recordings: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    recordings = _iter_val_recordings(cfg)
    rng.shuffle(recordings)
    target_len = int(round(duration_sec * cfg.data.target_fs))
    min_peaks = max(5, int(duration_sec * 0.5))
    san_cfg = _sanitize_config(cfg)

    per_recording: list[dict[str, Any]] = []
    hr_abs: list[float] = []
    hr_bias: list[float] = []
    sdnn_abs: list[float] = []
    rmssd_abs: list[float] = []
    mse_vals: list[float] = []
    prd_vals: list[float] = []
    cos_vals: list[float] = []
    prdn_vals: list[float] = []
    seam_ratios: list[float] = []
    seam_rms_vals: list[float] = []
    non_seam_rms_vals: list[float] = []
    noise_power_vals: list[float] = []
    noise_rms_vals: list[float] = []
    spectral_band_total_vals: list[float] = []
    spectral_weighted_prd_vals: list[float] = []
    spectral_coherence_vals: list[float] = []

    for src, path, _pid in recordings:
        if len(per_recording) >= num_recordings:
            break
        try:
            signal, pid = _load_h5_signal(path, src, cfg.data.target_fs)
        except Exception as exc:
            per_recording.append({"source": src.slug, "file": path.name, "error": str(exc)})
            continue
        if signal.size < target_len:
            continue
        start = int(rng.integers(0, signal.size - target_len + 1))
        raw = signal[start : start + target_len]
        if san_cfg is not None:
            clean = is_clean_window(raw[np.newaxis, :], san_cfg)
            if not clean.ok:
                per_recording.append({
                    "source": src.slug,
                    "file": path.name,
                    "patient_id": pid,
                    "quality_rejected": True,
                    "reject_reason": clean.reason,
                })
                continue
        recon = _reconstruct_overlap_add_h5_norm(
            model,
            raw,
            frame_size=cfg.data.window_samples,
            hop_ratio=hop_ratio,
            batch_size=batch_size,
        )
        noise_floor = estimate_ppg_noise_floor(raw, fs=cfg.data.target_fs)
        sig_metrics = compute_signal_metrics(
            raw, recon, noise_power=float(noise_floor.get("bp_noise_power", 0.0))
        )
        spectral = _spectral_summary(
            raw[np.newaxis, :], recon[np.newaxis, :], sample_rate=cfg.data.target_fs,
        )
        seam = seam_discontinuity_ratio(
            recon,
            frame_size=cfg.data.window_samples,
            hop_ratio=hop_ratio,
            radius=4,
        )
        orig_pk = compute_ppg_physiokit_metrics(
            raw, sample_rate=cfg.data.target_fs, low_hz=0.5, high_hz=8.0, order=3,
            min_peaks=min_peaks,
        )
        recon_pk = compute_ppg_physiokit_metrics(
            recon, sample_rate=cfg.data.target_fs, low_hz=0.5, high_hz=8.0, order=3,
            min_peaks=min_peaks,
        )
        entry: dict[str, Any] = {
            "source": src.slug,
            "file": path.name,
            "patient_id": pid,
            "start_sample": start,
            "quality_rejected": False,
            "signal_metrics": sig_metrics,
            "noise_floor": noise_floor,
            "spectral": spectral,
            "seam_metrics": seam,
            "original_physiokit": orig_pk,
            "reconstructed_physiokit": recon_pk,
            "delta": None,
        }
        if orig_pk is not None and recon_pk is not None and orig_pk["sdnn_ms"] <= 300.0:
            hr_diff = recon_pk["hr_bpm"] - orig_pk["hr_bpm"]
            sdnn_diff = recon_pk["sdnn_ms"] - orig_pk["sdnn_ms"]
            rmssd_diff = recon_pk["rmssd_ms"] - orig_pk["rmssd_ms"]
            entry["delta"] = {
                "hr_bpm": float(hr_diff),
                "sdnn_ms": float(sdnn_diff),
                "rmssd_ms": float(rmssd_diff),
            }
            hr_abs.append(abs(hr_diff))
            hr_bias.append(hr_diff)
            sdnn_abs.append(abs(sdnn_diff))
            rmssd_abs.append(abs(rmssd_diff))
            mse_vals.append(sig_metrics["mse"])
            prd_vals.append(sig_metrics["prd_percent"])
            prdn_vals.append(sig_metrics.get("prdn_noise_percent", float("nan")))
            cos_vals.append(sig_metrics["cosine_similarity"])
            noise_power_vals.append(float(noise_floor.get("bp_noise_power", 0.0)))
            noise_rms_vals.append(float(noise_floor.get("bp_noise_rms", 0.0)))
            seam_ratios.append(float(seam.get("ratio", float("nan"))))
            seam_rms_vals.append(float(seam.get("seam_rms", float("nan"))))
            non_seam_rms_vals.append(float(seam.get("non_seam_rms", float("nan"))))
            spectral_band_total_vals.append(
                float(spectral["band_total_rel_error"].get("mean", float("nan")))
            )
            spectral_weighted_prd_vals.append(
                float(spectral["weighted_freq_prd_percent"].get("mean", float("nan")))
            )
            spectral_coherence_vals.append(
                float(spectral["coherence"].get("mean", float("nan")))
            )
        per_recording.append(entry)

    valid = len(hr_abs)
    summary = None
    if valid:
        summary = {
            "duration_sec": duration_sec,
            "hop_ratio": hop_ratio,
            "num_total_recordings": len(per_recording),
            "num_valid_recordings": valid,
            "hr_mae_bpm": float(np.mean(hr_abs)),
            "hr_median_ae_bpm": float(np.median(hr_abs)),
            "hr_bias_bpm": float(np.mean(hr_bias)),
            "sdnn_mae_ms": float(np.mean(sdnn_abs)),
            "sdnn_median_ae_ms": float(np.median(sdnn_abs)),
            "rmssd_mae_ms": float(np.mean(rmssd_abs)),
            "rmssd_median_ae_ms": float(np.median(rmssd_abs)),
            "signal_mse": float(np.mean(mse_vals)),
            "signal_prd_percent": float(np.mean(prd_vals)),
            "signal_prdn_noise_percent": float(np.nanmean(prdn_vals)),
            "signal_cosine_similarity": float(np.mean(cos_vals)),
            "noise_power": _aggregate(noise_power_vals),
            "noise_rms": _aggregate(noise_rms_vals),
            "spectral": {
                "band_total_rel_error": _aggregate(spectral_band_total_vals),
                "weighted_freq_prd_percent": _aggregate(spectral_weighted_prd_vals),
                "coherence": _aggregate(spectral_coherence_vals),
            },
            "stability": {
                "seam_ratio": _aggregate(seam_ratios),
                "seam_rms": _aggregate(seam_rms_vals),
                "non_seam_rms": _aggregate(non_seam_rms_vals),
            },
        }
    return {"summary": summary, "per_recording": per_recording}


def evaluate_run(
    config_path: Path,
    *,
    num_samples: int | None,
    num_recordings: int,
    seed: int,
    weights_run_dir: Path | None = None,
    output_run_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = PpgH5RvqConfig.from_yaml(str(config_path))
    default_run_dir = Path(cfg.output.results_root) / cfg.run_name
    run_dir = output_run_dir or default_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = weights_run_dir or default_run_dir
    model, weights_name = _load_model(cfg, weights_dir)
    compression = _compression_context(cfg)
    metric_samples = cfg.evaluation.num_samples if num_samples is None else num_samples
    _purge_old_sample_artifacts(run_dir)
    short = _evaluate_short_windows(
        model, cfg, run_dir=run_dir, num_samples=metric_samples, seed=seed,
    )
    long = _evaluate_long_recordings(
        model,
        cfg,
        duration_sec=60.0,
        hop_ratio=0.5,
        num_recordings=num_recordings,
        batch_size=32,
        seed=seed,
    )
    out = {
        "run_name": cfg.run_name,
        "config_path": str(config_path),
        "weights": weights_name,
        "weights_run_dir": str(weights_dir),
        "output_run_dir": str(run_dir),
        "compression": compression,
        "short_window": short,
        "long_recording": long["summary"],
        "long_recording_per_recording": long["per_recording"],
    }
    out_path = run_dir / "h5_eval_metrics.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=str)

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {
            "run_name": run_dir.name,
            "source_config": str(config_path),
            "weights_run_dir": str(weights_dir),
        }
    summary["h5_eval_metrics"] = {
        "compression": compression,
        "short_window": short,
        "long_recording": long["summary"],
        "metrics_file": out_path.name,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Metric windows to evaluate. Defaults to evaluation.num_samples from each config.",
    )
    parser.add_argument("--num-recordings", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights-run-dir",
        type=Path,
        default=None,
        help="Load weights from this run directory instead of each config's run_name.",
    )
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        default=None,
        help="Write metrics/sample artifacts here instead of each config's run_name. Only valid with one config.",
    )
    args = parser.parse_args()

    if args.output_run_dir is not None and len(args.configs) != 1:
        raise ValueError("--output-run-dir can only be used with a single config path")

    results = [
        evaluate_run(
            path,
            num_samples=args.num_samples,
            num_recordings=args.num_recordings,
            seed=args.seed,
            weights_run_dir=args.weights_run_dir,
            output_run_dir=args.output_run_dir,
        )
        for path in args.configs
    ]
    for result in results:
        short_pk = result["short_window"]["physiokit"] or {}
        long_pk = result["long_recording"] or {}
        print(
            result["run_name"],
            "short_hr_mae=", short_pk.get("hr_mae_bpm"),
            "short_sdnn_mae=", short_pk.get("sdnn_mae_ms"),
            "short_rmssd_mae=", short_pk.get("rmssd_mae_ms"),
            "long_hr_mae=", long_pk.get("hr_mae_bpm"),
            "long_sdnn_mae=", long_pk.get("sdnn_mae_ms"),
            "long_rmssd_mae=", long_pk.get("rmssd_mae_ms"),
        )


if __name__ == "__main__":
    main()
