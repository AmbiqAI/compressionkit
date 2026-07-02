#!/usr/bin/env python3
"""Unified PPG v1 scorecard runner.

Produces ONE consolidated scorecard for a trained PPG codec run, composed from
three sections:

1. Clean fidelity      — PRD/cosine + HR/HRV alignment on held-out clean windows.
2. Robustness battery  — fixed corruptions (gaussian/motion/baseline/empirical/
                         cutout) reporting input floor, denoising delta, HR MAE,
                         and (cutout only) masked PRD over the visible span.
3. Synthetic triplets  — truth-aware prd_vs_clean vs prd_vs_noisy.

All sections run on a single fixed window set (paired, fixed seed) so candidates
are directly comparable. This script is self-contained: it depends only on the
tracked ``compressionkit`` package, not on local research scratch under
``experiments/`` (gitignored, not portable across environments).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from compressionkit.configs.ppg_rvq import PpgRvqConfig
from compressionkit.datasets.ppg_cache import SourceWeight, load_cached_raw_windows
from compressionkit.evaluation.empirical_regime import encode_decode_batch, prd
from compressionkit.evaluation.metrics import compute_ppg_physiokit_metrics
from compressionkit.evaluation.rvq_codec import RvqCodec
from compressionkit.preprocessing.augmentations import (
    add_baseline_wander,
    add_empirical_noise,
    add_motion_artifact,
)
from compressionkit.preprocessing.ppg import generate_synthetic_ppg_batch
from compressionkit.synthetic import NoiseSpec, add_noise, ppg_dynamical
from compressionkit.trainers.ppg_rvq import _build_ppg_noise_bank

BATTERY_ARTIFACTS = ["gaussian", "motion", "baseline_wander", "empirical_noise", "cutout"]
SYNTHETIC_ARTIFACTS = ["gaussian", "baseline_wander", "motion"]
SYNTHETIC_LEVELS = [0.25, 0.5]
SYNTHETIC_DROP_WARN_FRACTION = 0.5


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=-1)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    return dot / (na * nb + 1e-12)


def _flatten_frames(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def _default_levels(artifact: str) -> list[str]:
    if artifact in {"motion", "empirical_noise"}:
        return ["clean", "15", "12", "9", "6"]
    if artifact == "baseline_wander":
        return ["clean", "0.20", "0.35", "0.50", "0.60"]
    if artifact == "gaussian":
        return ["clean", "0.10", "0.15", "0.20", "0.25"]
    if artifact == "cutout":
        return ["clean", "0.10", "0.20", "0.30", "0.40"]
    raise ValueError(f"Unsupported artifact: {artifact}")


def _layer_norm_windows(windows: np.ndarray, epsilon: float) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float32)
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.mean(np.square(x - mean), axis=1, keepdims=True)
    return ((x - mean) / np.sqrt(var + epsilon)).astype(np.float32)


def _apply_cutout(batch: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = batch.copy()
    width = max(1, round(fraction * batch.shape[1]))
    width = min(width, batch.shape[1])
    for idx in range(batch.shape[0]):
        start = int(rng.integers(0, batch.shape[1] - width + 1))
        out[idx, start : start + width] = 0.0
    return out


def corrupt_raw_windows(
    clean_raw: np.ndarray,
    *,
    artifact: str,
    level: str,
    sample_rate: int,
    seed: int,
    noise_bank: np.ndarray | None,
) -> np.ndarray:
    """Apply one fixed-severity corruption to a batch of raw (un-normalized) windows."""
    if level == "clean":
        return clean_raw.copy()

    value = float(level)
    rng = np.random.default_rng(seed)
    out = clean_raw.copy().astype(np.float32)

    if artifact == "gaussian":
        noise = rng.standard_normal(out.shape).astype(np.float32)
        sig_std = np.std(out, axis=1, keepdims=True) + 1e-8
        return out + noise * (value * sig_std)

    if artifact == "baseline_wander":
        for idx in range(out.shape[0]):
            out[idx] = add_baseline_wander(
                out[idx],
                sample_rate=sample_rate,
                amplitude_range=(value, value),
                rng=rng,
            )
        return out

    if artifact == "motion":
        for idx in range(out.shape[0]):
            out[idx] = add_motion_artifact(
                out[idx],
                sample_rate=sample_rate,
                snr_range=(value, value),
                rng=rng,
            )
        return out

    if artifact == "empirical_noise":
        if noise_bank is None or len(noise_bank) == 0:
            raise ValueError("empirical_noise requested but no noise bank is available")
        for idx in range(out.shape[0]):
            out[idx] = add_empirical_noise(
                out[idx],
                noise_bank,
                snr_range=(value, value),
                rng=rng,
            )
        return out

    if artifact == "cutout":
        return _apply_cutout(out, value, seed)

    raise ValueError(f"Unsupported artifact: {artifact}")


def codec_metrics(clean: np.ndarray, noisy: np.ndarray, recon: np.ndarray) -> dict[str, np.ndarray | float]:
    clean_f = _flatten_frames(clean)
    noisy_f = _flatten_frames(noisy)
    recon_f = _flatten_frames(recon)
    prd_clean = prd(clean_f, recon_f)
    prd_noisy = prd(noisy_f, recon_f)
    cos_clean = _cos(clean_f, recon_f)
    cos_noisy = _cos(noisy_f, recon_f)
    prd_input = prd(clean_f, noisy_f)
    return {
        "prd_clean": prd_clean,
        "prd_noisy": prd_noisy,
        "cos_clean": cos_clean,
        "cos_noisy": cos_noisy,
        "prd_clean_mean": float(np.mean(prd_clean)),
        "prd_noisy_mean": float(np.mean(prd_noisy)),
        "cos_clean_mean": float(np.mean(cos_clean)),
        "cos_noisy_mean": float(np.mean(cos_noisy)),
        "denoising_delta_prd": float(np.mean(prd_input) - np.mean(prd_clean)),
        "noisy_vs_clean_prd": float(np.mean(prd_input)),
        "noisy_vs_clean_cos": float(np.mean(_cos(clean_f, noisy_f))),
    }


def _build_clean_windows(
    *,
    generator: str,
    n_windows: int,
    frame_size: int,
    sample_rate: int,
    seed: int,
) -> np.ndarray:
    if generator == "physiokit":
        return _layer_norm_windows(
            generate_synthetic_ppg_batch(
                num_segments=n_windows,
                signal_length=frame_size,
                sample_rate=sample_rate,
                seed=seed,
            ),
            1e-9,
        )

    rng = np.random.default_rng(seed)
    out = np.empty((n_windows, frame_size), dtype=np.float32)
    pre_roll_s = 1.0
    window_s = frame_size / sample_rate
    n_pre = int(pre_roll_s * sample_rate)
    for idx in range(n_windows):
        hr = float(rng.uniform(55.0, 95.0))
        hr_std = float(rng.uniform(0.5, 3.0))
        resp_hz = float(rng.uniform(0.18, 0.32))
        sig = ppg_dynamical(
            duration_s=pre_roll_s + window_s,
            sample_rate=sample_rate,
            hr_mean=hr,
            hr_std=hr_std,
            respiration_hz=resp_hz,
            respiration_amplitude_mod=float(rng.uniform(0.02, 0.08)),
            respiration_baseline_amplitude=float(rng.uniform(0.0, 0.03)),
            seed=seed + 2000 + idx,
        )
        sig = np.asarray(sig[n_pre : n_pre + frame_size], dtype=np.float32)
        out[idx] = sig
    return _layer_norm_windows(out, 1e-9)


def _corrupt_windows(
    clean: np.ndarray,
    *,
    artifact: str,
    level: float,
    sample_rate: int,
    seed: int,
) -> np.ndarray:
    if artifact == "none" or level <= 0.0:
        return clean.copy()

    rng = np.random.default_rng(seed)
    out = np.empty_like(clean)
    for idx, frame in enumerate(clean):
        if artifact == "gaussian":
            noise = rng.standard_normal(frame.shape).astype(np.float32) * (level * frame.std() + 1e-9)
            x = frame + noise
        else:
            if artifact == "baseline_wander":
                spec = NoiseSpec(weights={"baseline_wander": 1.0})
            elif artifact == "motion":
                spec = NoiseSpec(weights={"motion": 1.0})
            elif artifact == "wearable_mix":
                spec = NoiseSpec(weights={"baseline_wander": 1.0, "motion": 1.5, "gauss": 0.5})
            else:
                raise ValueError(f"Unsupported artifact: {artifact}")
            snr_db = float(30.0 - 30.0 * level)
            x, _ = add_noise(
                frame,
                sample_rate=float(sample_rate),
                snr_db=snr_db,
                spec=spec,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        out[idx] = x.astype(np.float32)
    return _layer_norm_windows(out, 1e-9)


def _synthetic_metrics(clean: np.ndarray, noisy: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    clean_f = _flatten_frames(clean)
    noisy_f = _flatten_frames(noisy)
    recon_f = _flatten_frames(recon)
    return {
        "prd_vs_clean": float(np.mean(prd(clean_f, recon_f))),
        "prd_vs_noisy": float(np.mean(prd(noisy_f, recon_f))),
        "cos_vs_clean": float(np.mean(_cos(clean_f, recon_f))),
        "cos_vs_noisy": float(np.mean(_cos(noisy_f, recon_f))),
        "input_prd_vs_clean": float(np.mean(prd(clean_f, noisy_f))),
        "denoising_delta_prd": float(np.mean(prd(clean_f, noisy_f)) - np.mean(prd(clean_f, recon_f))),
    }


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def _hr_series(windows: np.ndarray, *, sample_rate: int, phys: dict) -> np.ndarray:
    """Per-window HR (bpm); NaN where physiokit cannot resolve a stable HR."""
    out = np.full(windows.shape[0], np.nan, dtype=np.float64)
    for i, w in enumerate(windows):
        m = compute_ppg_physiokit_metrics(
            w,
            sample_rate=sample_rate,
            low_hz=phys["low_hz"],
            high_hz=phys["high_hz"],
            order=phys["order"],
            min_peaks=phys["min_peaks"],
        )
        if m is not None:
            out[i] = m["hr_bpm"]
    return out


def _hr_mae(hr_ref: np.ndarray, hr_est: np.ndarray) -> dict[str, float]:
    valid = ~np.isnan(hr_ref) & ~np.isnan(hr_est)
    n = int(valid.sum())
    if n == 0:
        return {"hr_mae_bpm": float("nan"), "coverage": 0.0, "n": 0}
    return {
        "hr_mae_bpm": float(np.mean(np.abs(hr_ref[valid] - hr_est[valid]))),
        "coverage": float(n / len(hr_ref)),
        "n": n,
    }


def _clean_fidelity(codec, clean_norm, *, sample_rate, phys, hr_clean):
    recon = encode_decode_batch(codec, clean_norm)
    cf = codec_metrics(clean_norm, clean_norm, recon)  # noisy==clean -> floor 0
    hr_recon = _hr_series(recon, sample_rate=sample_rate, phys=phys)
    return {
        "prd_median": _pct(cf["prd_clean"], 50),
        "prd_p90": _pct(cf["prd_clean"], 90),
        "cos_median": _pct(cf["cos_clean"], 50),
        "hr": _hr_mae(hr_clean, hr_recon),
    }, recon


def _cutout_mask(clean_raw, noisy_raw):
    """Boolean mask of the zeroed (dropout) span for cutout windows."""
    return noisy_raw == 0.0


def _masked_prd(clean_norm, recon, mask_visible):
    """PRD computed only over the visible (non-dropout) samples, per window."""
    vals = []
    for c, r, m in zip(clean_norm, recon, mask_visible):
        if m.sum() < 2:
            continue
        cf, rf = c.reshape(-1)[m], r.reshape(-1)[m]
        num = np.linalg.norm(cf - rf)
        den = np.linalg.norm(cf) + 1e-12
        vals.append(100.0 * num / den)
    return float(np.median(vals)) if vals else float("nan")


def _filter_finite_rows(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Drop any window rows that contain non-finite values in any paired array."""
    if not arrays:
        return ()
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr.reshape(arr.shape[0], -1)).all(axis=1)
    return tuple(arr[mask] for arr in arrays)


def _drop_stats(before: int, after: int) -> dict[str, float | int]:
    dropped = max(0, before - after)
    fraction = (dropped / before) if before else 0.0
    return {
        "input_windows": int(before),
        "retained_windows": int(after),
        "dropped_windows": int(dropped),
        "dropped_fraction": float(fraction),
        "warn_threshold_fraction": float(SYNTHETIC_DROP_WARN_FRACTION),
        "warn_threshold_exceeded": bool(fraction > SYNTHETIC_DROP_WARN_FRACTION),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--num-windows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir
    cfg = PpgRvqConfig.model_validate_json((run_dir / "config.json").read_text())
    sr = int(cfg.data.sampling_rate)
    frame = int(cfg.data.frame_size)
    pm = cfg.evaluation.physiokit_metrics
    phys = {"low_hz": pm.low_hz, "high_hz": pm.high_hz, "order": pm.order, "min_peaks": max(4, pm.min_peaks)}

    print(f"Loading codec from {run_dir}")
    codec = RvqCodec.from_run_dir(run_dir, modality="ppg")

    # Fixed shared real window set from the unified strict-sanitized val split.
    sources = [SourceWeight(slug=s.slug, weight=s.weight) for s in cfg.data.unified_cache.sources]
    clean_raw = load_cached_raw_windows(
        sources,
        cache_root=Path(cfg.data.unified_cache.cache_root),
        frame_size=frame,
        split="val",
        max_windows=args.num_windows,
        seed=args.seed,
    )
    clean_norm = _layer_norm_windows(clean_raw, cfg.data.epsilon)
    hr_clean = _hr_series(clean_norm, sample_rate=sr, phys=phys)
    print(f"Window set: {len(clean_raw)} | HR coverage on clean: {np.mean(~np.isnan(hr_clean)):.0%}")

    noise_bank = _build_ppg_noise_bank(cfg)

    report: dict = {
        "run_dir": str(run_dir),
        "num_windows": len(clean_raw),
        "seed": args.seed,
        "compression_ratio": float(getattr(codec, "target_cr", 0.0)),
    }

    # --- Section 1: clean fidelity -------------------------------------------
    print("\n[1/3] Clean fidelity")
    clean_section, _ = _clean_fidelity(codec, clean_norm, sample_rate=sr, phys=phys, hr_clean=hr_clean)
    report["clean_fidelity"] = clean_section
    print(
        f"  PRD median={clean_section['prd_median']:.2f} p90={clean_section['prd_p90']:.2f} "
        f"| HR MAE={clean_section['hr']['hr_mae_bpm']:.2f} bpm (cov {clean_section['hr']['coverage']:.0%})"
    )

    # --- Section 2: robustness battery ---------------------------------------
    print("\n[2/3] Robustness battery (floor / delta / HR MAE)")
    battery: dict = {}
    for art in BATTERY_ARTIFACTS:
        if art == "empirical_noise" and (noise_bank is None or len(noise_bank) == 0):
            continue
        battery[art] = {}
        for level in _default_levels(art):
            noisy_raw = corrupt_raw_windows(
                clean_raw,
                artifact=art,
                level=level,
                sample_rate=sr,
                seed=args.seed,
                noise_bank=noise_bank,
            )
            noisy_norm = _layer_norm_windows(noisy_raw, cfg.data.epsilon)
            recon = encode_decode_batch(codec, noisy_norm)
            m = codec_metrics(clean_norm, noisy_norm, recon)
            hr_recon = _hr_series(recon, sample_rate=sr, phys=phys)
            entry = {
                "input_floor_prd": m["noisy_vs_clean_prd"],
                "denoising_delta_prd": m["denoising_delta_prd"],
                "prd_vs_clean_median": _pct(m["prd_clean"], 50),
                "hr": _hr_mae(hr_clean, hr_recon),
            }
            if art == "cutout":
                mask_visible = ~_cutout_mask(clean_raw, noisy_raw)
                entry["masked_prd_median"] = _masked_prd(clean_norm, recon, mask_visible)
            battery[art][level] = entry
            extra = f" maskedPRD={entry.get('masked_prd_median', float('nan')):.1f}" if art == "cutout" else ""
            print(
                f"  {art:>16} @ {level:>6}: floor={entry['input_floor_prd']:6.1f}  "
                f"delta={entry['denoising_delta_prd']:+6.2f}  HRmae={entry['hr']['hr_mae_bpm']:5.2f}{extra}"
            )
    report["robustness_battery"] = battery

    # --- Section 3: synthetic truth-aware triplets ---------------------------
    print("\n[3/3] Synthetic triplets (prd_vs_clean vs prd_vs_noisy)")
    synth_clean = _build_clean_windows(
        generator="physiokit",
        n_windows=min(args.num_windows, 500),
        frame_size=frame,
        sample_rate=sr,
        seed=args.seed,
    )
    synth_clean_total = int(synth_clean.shape[0])
    (synth_clean,) = _filter_finite_rows(synth_clean)
    synth_clean_stats = _drop_stats(synth_clean_total, int(synth_clean.shape[0]))
    if synth_clean_stats["dropped_windows"]:
        warn = " WARNING" if synth_clean_stats["warn_threshold_exceeded"] else ""
        print(
            "  synthetic clean finite filter: "
            f"dropped={synth_clean_stats['dropped_windows']} / {synth_clean_stats['input_windows']} "
            f"({100.0 * synth_clean_stats['dropped_fraction']:.1f}%){warn}"
        )
    triplets: dict = {}
    triplet_drop_stats: dict = {"clean": synth_clean_stats}
    for art in SYNTHETIC_ARTIFACTS:
        triplets[art] = {}
        triplet_drop_stats[art] = {}
        for level in SYNTHETIC_LEVELS:
            noisy = _corrupt_windows(synth_clean, artifact=art, level=level, sample_rate=sr, seed=args.seed)
            recon = encode_decode_batch(codec, noisy)
            pair_total = int(synth_clean.shape[0])
            clean_f, noisy_f, recon_f = _filter_finite_rows(synth_clean, noisy, recon)
            drop_stats = _drop_stats(pair_total, int(clean_f.shape[0]))
            triplet_drop_stats[art][str(level)] = drop_stats
            triplets[art][str(level)] = _synthetic_metrics(clean_f, noisy_f, recon_f)
            t = triplets[art][str(level)]
            drop_note = ""
            if drop_stats["dropped_windows"]:
                drop_note = (
                    f" drop={drop_stats['dropped_windows']}/{drop_stats['input_windows']}"
                    f" ({100.0 * drop_stats['dropped_fraction']:.1f}%)"
                )
                if drop_stats["warn_threshold_exceeded"]:
                    drop_note += " WARNING"
            print(
                f"  {art:>16} @ {level}: prd_vs_clean={t['prd_vs_clean']:6.2f}  "
                f"prd_vs_noisy={t['prd_vs_noisy']:6.2f}  delta={t['denoising_delta_prd']:+6.2f}{drop_note}"
            )
    report["synthetic_triplets"] = triplets
    report["synthetic_triplet_drop_stats"] = triplet_drop_stats

    out_path = args.out or (run_dir / "v1_scorecard.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
