#!/usr/bin/env python3
"""Unified PPG v1 scorecard runner.

Produces ONE consolidated scorecard for a trained PPG codec run, composed from
the pieces the v1 design requires (experiments/ppg_v1_augmentation_eval_design.md):

1. Clean fidelity      — PRD/cosine + HR/HRV alignment on held-out clean windows.
2. Robustness battery  — fixed corruptions (gaussian/motion/baseline/empirical/
                         cutout) reporting input floor, denoising delta, HR MAE,
                         and (cutout only) masked PRD over the visible span.
3. Synthetic triplets  — truth-aware prd_vs_clean vs prd_vs_noisy.

All sections run on a single fixed window set (paired, fixed seed) so candidates
are directly comparable. Reuses the audited eval primitives rather than
reimplementing corruption/metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Ensure the repo root is importable so the audited eval primitives under
# experiments/ can be reused when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from compressionkit.configs.ppg_rvq import PpgRvqConfig
from compressionkit.datasets.ppg_cache import SourceWeight, load_cached_raw_windows
from compressionkit.evaluation.metrics import compute_ppg_physiokit_metrics
from compressionkit.evaluation.rvq_codec import RvqCodec
from compressionkit.trainers.ppg_rvq import _build_ppg_noise_bank

# Reuse audited primitives.
from experiments.eval_ppg_artifact_paired import (
    codec_metrics,
    corrupt_raw_windows,
    encode_decode_batch,
    _default_levels,
    _layer_norm_windows,
)
from experiments.eval_ppg_synthetic_triplet import (
    _build_clean_windows,
    _corrupt_windows,
    _metrics as _synthetic_metrics,
)

BATTERY_ARTIFACTS = ["gaussian", "motion", "baseline_wander", "empirical_noise", "cutout"]
SYNTHETIC_ARTIFACTS = ["gaussian", "baseline_wander", "motion"]
SYNTHETIC_LEVELS = [0.25, 0.5]
SYNTHETIC_DROP_WARN_FRACTION = 0.5


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def _hr_series(windows: np.ndarray, *, sample_rate: int, phys: dict) -> np.ndarray:
    """Per-window HR (bpm); NaN where physiokit cannot resolve a stable HR."""
    out = np.full(windows.shape[0], np.nan, dtype=np.float64)
    for i, w in enumerate(windows):
        m = compute_ppg_physiokit_metrics(
            w, sample_rate=sample_rate,
            low_hz=phys["low_hz"], high_hz=phys["high_hz"],
            order=phys["order"], min_peaks=phys["min_peaks"],
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
        return tuple()
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
    phys = {"low_hz": pm.low_hz, "high_hz": pm.high_hz, "order": pm.order,
            "min_peaks": max(4, pm.min_peaks)}

    print(f"Loading codec from {run_dir}")
    codec = RvqCodec.from_run_dir(run_dir, modality="ppg")

    # Fixed shared real window set from the unified strict-sanitized val split.
    sources = [SourceWeight(slug=s.slug, weight=s.weight) for s in cfg.data.unified_cache.sources]
    clean_raw = load_cached_raw_windows(
        sources, cache_root=Path(cfg.data.unified_cache.cache_root),
        frame_size=frame, split="val", max_windows=args.num_windows, seed=args.seed,
    )
    clean_norm = _layer_norm_windows(clean_raw, cfg.data.epsilon)
    hr_clean = _hr_series(clean_norm, sample_rate=sr, phys=phys)
    print(f"Window set: {len(clean_raw)} | HR coverage on clean: {np.mean(~np.isnan(hr_clean)):.0%}")

    noise_bank = _build_ppg_noise_bank(cfg)

    report: dict = {
        "run_dir": str(run_dir),
        "num_windows": int(len(clean_raw)),
        "seed": args.seed,
        "compression_ratio": float(getattr(codec, "target_cr", 0.0)),
    }

    # --- Section 1: clean fidelity -------------------------------------------
    print("\n[1/3] Clean fidelity")
    clean_section, _ = _clean_fidelity(codec, clean_norm, sample_rate=sr, phys=phys, hr_clean=hr_clean)
    report["clean_fidelity"] = clean_section
    print(f"  PRD median={clean_section['prd_median']:.2f} p90={clean_section['prd_p90']:.2f} "
          f"| HR MAE={clean_section['hr']['hr_mae_bpm']:.2f} bpm (cov {clean_section['hr']['coverage']:.0%})")

    # --- Section 2: robustness battery ---------------------------------------
    print("\n[2/3] Robustness battery (floor / delta / HR MAE)")
    battery: dict = {}
    for art in BATTERY_ARTIFACTS:
        if art == "empirical_noise" and (noise_bank is None or len(noise_bank) == 0):
            continue
        battery[art] = {}
        for level in _default_levels(art):
            noisy_raw = corrupt_raw_windows(
                clean_raw, artifact=art, level=level, sample_rate=sr,
                seed=args.seed, noise_bank=noise_bank,
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
            print(f"  {art:>16} @ {level:>6}: floor={entry['input_floor_prd']:6.1f}  "
                  f"delta={entry['denoising_delta_prd']:+6.2f}  HRmae={entry['hr']['hr_mae_bpm']:5.2f}{extra}")
    report["robustness_battery"] = battery

    # --- Section 3: synthetic truth-aware triplets ---------------------------
    print("\n[3/3] Synthetic triplets (prd_vs_clean vs prd_vs_noisy)")
    synth_clean = _build_clean_windows(
        generator="physiokit", n_windows=min(args.num_windows, 500),
        frame_size=frame, sample_rate=sr, seed=args.seed,
    )
    synth_clean_total = int(synth_clean.shape[0])
    synth_clean, = _filter_finite_rows(synth_clean)
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
            print(f"  {art:>16} @ {level}: prd_vs_clean={t['prd_vs_clean']:6.2f}  "
                  f"prd_vs_noisy={t['prd_vs_noisy']:6.2f}  delta={t['denoising_delta_prd']:+6.2f}{drop_note}")
    report["synthetic_triplets"] = triplets
    report["synthetic_triplet_drop_stats"] = triplet_drop_stats

    out_path = args.out or (run_dir / "v1_scorecard.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
