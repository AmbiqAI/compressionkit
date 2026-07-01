"""Self-validation of synthetic generators against the morphology evaluators.

Three checks per modality:

1. **Identity** -- pair (clean, clean). Every paired metric Δ must be ~0.
   This isolates evaluator-internal noise (peak match jitter, etc.).
2. **Repeatability** -- pair (clean, clean + 1 LSB-equivalent noise). Sets
   the measurement floor below which codec differences are not meaningful.
3. **Noise-only baseline** -- pair (clean, noisy at SNR = 20 / 10 / 0 dB).
   Quantifies how much morphology metrics drift purely from noise on the
   "recon" side, with no codec involved. This is the floor any codec must
   beat to claim it is preserving morphology *better than passing noise
   straight through*.

Outputs ``results/_synthetic_validation.json`` summarising all three checks
for both ECG and PPG.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from compressionkit.evaluation.ecg_morphology import evaluate_ecg_morphology
from compressionkit.evaluation.ppg_morphology import evaluate_ppg_morphology
from compressionkit.synthetic import (
    NoiseSpec,
    add_noise,
    ecg_mcsharry,
    ppg_dynamical,
)


def _build_ecg_batch(
    n_windows: int,
    window_s: float,
    sample_rate: float,
    hr_jitter: tuple[float, float] = (55.0, 95.0),
) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = np.empty((n_windows, int(window_s * sample_rate)), dtype=np.float32)
    for i in range(n_windows):
        hr = float(rng.uniform(*hr_jitter))
        sig = ecg_mcsharry(
            duration_s=window_s,
            sample_rate=sample_rate,
            hr_mean=hr,
            hr_std=1.5,
            seed=1000 + i,
        )
        # Scale to ~mV range so the evaluator's bandpass / peak detector is
        # comfortable, and z-normalize per window to match scorecard input.
        sig = sig * 100.0
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        out[i] = sig.astype(np.float32)
    return out


def _build_ppg_batch(
    n_windows: int,
    window_s: float,
    sample_rate: float,
    hr_jitter: tuple[float, float] = (55.0, 95.0),
) -> np.ndarray:
    rng = np.random.default_rng(1)
    out = np.empty((n_windows, int(window_s * sample_rate)), dtype=np.float32)
    for i in range(n_windows):
        hr = float(rng.uniform(*hr_jitter))
        sig = ppg_dynamical(
            duration_s=window_s,
            sample_rate=sample_rate,
            hr_mean=hr,
            hr_std=1.0,
            seed=2000 + i,
        )
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        out[i] = sig.astype(np.float32)
    return out


def _add_noise_batch(
    clean: np.ndarray,
    sample_rate: float,
    snr_db: float,
    spec: NoiseSpec,
    seed_offset: int,
) -> np.ndarray:
    noisy = np.empty_like(clean)
    for i, c in enumerate(clean):
        nz, _ = add_noise(
            c.astype(np.float64),
            sample_rate=sample_rate,
            snr_db=snr_db,
            spec=spec,
            seed=seed_offset + i,
        )
        nz = (nz - nz.mean()) / (nz.std() + 1e-9)
        noisy[i] = nz.astype(np.float32)
    return noisy


def _summarise_morph(report: dict[str, Any], metric_keys: list[str]) -> dict[str, Any]:
    """Pick the headline numbers from a morphology report."""
    out: dict[str, Any] = {
        "windows": report.get("num_windows"),
        "windows_with_beats": report.get("num_windows_with_beats"),
        "matched": report.get("num_beats_matched", report.get("num_pulses_matched")),
    }
    for k in metric_keys:
        block = report.get(k)
        if not isinstance(block, dict):
            continue
        if "abs_delta" in block and isinstance(block["abs_delta"], dict):
            out[k] = {
                "abs_delta_mean": block["abs_delta"].get("mean"),
                "abs_delta_p90": block["abs_delta"].get("p90"),
                "orig_mean": block.get("orig", {}).get("mean"),
                "recon_mean": block.get("recon", {}).get("mean"),
            }
        elif "mean" in block:
            out[k] = {"mean": block.get("mean"), "p10": block.get("p10")}
    return out


def _run_ecg_checks(n_windows: int = 20, window_s: float = 8.0, sample_rate: float = 256.0) -> dict[str, Any]:
    clean = _build_ecg_batch(n_windows, window_s, sample_rate)
    print(f"[ECG] built {clean.shape[0]} clean windows of {clean.shape[1]} samples")

    # ECG-style noise mixture: BW + EMG + powerline + sensor floor.
    ecg_noise = NoiseSpec(
        weights={
            "baseline_wander": 1.0,
            "emg": 1.0,
            "powerline": 0.3,
            "gauss": 0.3,
        },
        powerline_hz=60.0,
    )

    headline = [
        "r_amplitude",
        "qrs_width_ms",
        "st_deviation",
        "t_amplitude",
        "baseline_drift",
        "beat_shape_correlation",
    ]

    results: dict[str, Any] = {}

    print("[ECG] identity (clean vs clean) ...")
    rep = evaluate_ecg_morphology(clean, clean, sample_rate=int(sample_rate))
    results["identity"] = _summarise_morph(rep, headline)

    print("[ECG] repeatability (clean vs clean + tiny noise) ...")
    tiny = clean + (1.0 / 32768.0) * np.random.default_rng(7).standard_normal(clean.shape).astype(np.float32)
    rep = evaluate_ecg_morphology(clean, tiny, sample_rate=int(sample_rate))
    results["repeatability_1lsb"] = _summarise_morph(rep, headline)

    for snr in (20.0, 10.0, 0.0):
        print(f"[ECG] noise-only baseline @ SNR={snr:>4.1f} dB ...")
        noisy = _add_noise_batch(clean, sample_rate, snr, ecg_noise, seed_offset=5000)
        rep = evaluate_ecg_morphology(clean, noisy, sample_rate=int(sample_rate))
        results[f"noise_only_snr_{int(snr):02d}db"] = _summarise_morph(rep, headline)

    return results


def _run_ppg_checks(n_windows: int = 20, window_s: float = 10.0, sample_rate: float = 64.0) -> dict[str, Any]:
    clean = _build_ppg_batch(n_windows, window_s, sample_rate)
    print(f"[PPG] built {clean.shape[0]} clean windows of {clean.shape[1]} samples")

    # PPG noise mixture: dominated by motion + baseline wander.
    ppg_noise = NoiseSpec(
        weights={
            "baseline_wander": 1.0,
            "motion": 1.5,
            "gauss": 0.5,
        },
    )

    headline = [
        "ac_amplitude_norm",
        "upstroke_slope",
        "pulse_width_sec",
        "rise_time_sec",
        "fall_time_sec",
        "pulse_shape_correlation",
    ]

    results: dict[str, Any] = {}

    print("[PPG] identity ...")
    rep = evaluate_ppg_morphology(clean, clean, sample_rate=int(sample_rate))
    results["identity"] = _summarise_morph(rep, headline)

    print("[PPG] repeatability ...")
    tiny = clean + (1.0 / 32768.0) * np.random.default_rng(8).standard_normal(clean.shape).astype(np.float32)
    rep = evaluate_ppg_morphology(clean, tiny, sample_rate=int(sample_rate))
    results["repeatability_1lsb"] = _summarise_morph(rep, headline)

    for snr in (20.0, 10.0, 0.0):
        print(f"[PPG] noise-only baseline @ SNR={snr:>4.1f} dB ...")
        noisy = _add_noise_batch(clean, sample_rate, snr, ppg_noise, seed_offset=6000)
        rep = evaluate_ppg_morphology(clean, noisy, sample_rate=int(sample_rate))
        results[f"noise_only_snr_{int(snr):02d}db"] = _summarise_morph(rep, headline)

    return results


def main() -> None:
    out_path = Path("results/_synthetic_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "ecg": _run_ecg_checks(),
        "ppg": _run_ppg_checks(),
    }
    out_path.write_text(
        json.dumps(summary, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o)
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
