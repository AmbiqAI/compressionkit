"""Compare real PPG codecs on a shared synthetic-truth bundle.

This script generates clean synthetic PPG windows, corrupts them with a fixed
Gaussian noise level, runs both a real RVQ golden and a real SPIHT codec over
the exact same noisy inputs, and builds truth-aware quality scorecards for each.

Outputs under ``results/_synthetic_scorecards/ppg_<cr>x_noiseXX/``:

* ``clean_truth.npz`` -- aligned clean-reference bundle used by both codecs.
* ``rvq/quality_scorecard.json``
* ``spiht/quality_scorecard.json``
* ``comparison.json`` -- concise side-by-side summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from compressionkit.evaluation.codec import SpihtAcCodec
from compressionkit.evaluation.rvq_codec import RvqCodec
from compressionkit.evaluation.scorecard import write_quality_scorecard
from compressionkit.synthetic import ppg_dynamical


def _build_clean_windows(n_windows: int, frame_size: int, sample_rate: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = np.empty((n_windows, frame_size), dtype=np.float32)
    pre_roll_s = 1.0
    window_s = frame_size / sample_rate
    n_pre = int(pre_roll_s * sample_rate)
    for i in range(n_windows):
        hr = float(rng.uniform(55.0, 95.0))
        sig = ppg_dynamical(
            duration_s=pre_roll_s + window_s,
            sample_rate=sample_rate,
            hr_mean=hr,
            hr_std=1.0,
            seed=2000 + i,
        )
        sig = sig[n_pre : n_pre + frame_size].astype(np.float32)
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        out[i] = sig
    return out


def _add_gaussian(clean: np.ndarray, noise_pct: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty_like(clean)
    for i, c in enumerate(clean):
        noise = rng.standard_normal(c.shape).astype(np.float32) * (noise_pct * c.std() + 1e-9)
        x = c + noise
        x = (x - x.mean()) / (x.std() + 1e-9)
        out[i] = x
    return out


def _encode_decode_batch(codec: Any, frames: np.ndarray) -> np.ndarray:
    out = np.empty_like(frames)
    for i, frame in enumerate(frames):
        encoded = codec.encode(frame)
        recon = codec.decode(encoded)
        recon = np.asarray(recon, dtype=np.float32).reshape(-1)[: frames.shape[1]]
        recon = (recon - recon.mean()) / (recon.std() + 1e-9)
        out[i] = recon
    return out


def _write_run_dir(run_dir: Path, noisy: np.ndarray, recon: np.ndarray) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text("{}")
    for i, (orig, out) in enumerate(zip(noisy, recon)):
        pd.DataFrame({
            "original": orig.astype(np.float32),
            "reconstructed": out.astype(np.float32),
        }).to_csv(run_dir / f"sample_{i:03d}.csv", index=False)


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(float(den)) < 1e-12:
        return None
    return float(num) / float(den)


def _extract_headline(card: dict[str, Any]) -> dict[str, float | None]:
    clean_ref = card.get("clean_reference", {})
    base_td = clean_ref.get("input_baseline", {}).get("time_domain", {})
    out_td = clean_ref.get("reconstruction", {}).get("time_domain", {})
    den_td = clean_ref.get("denoising", {}).get("time_domain", {})
    out_sp = clean_ref.get("reconstruction", {}).get("spectral", {})
    input_prd = base_td.get("prd_percent", {}).get("mean")
    recon_prd = out_td.get("prd_percent", {}).get("mean")
    denoise_prd = den_td.get("prd_percent_improvement")
    input_cosine = base_td.get("cosine_similarity", {}).get("mean")
    recon_cosine = out_td.get("cosine_similarity", {}).get("mean")
    return {
        "top_level_prd_vs_noisy": card.get("time_domain", {}).get("prd_percent", {}).get("mean"),
        "top_level_cosine_vs_noisy": card.get("time_domain", {}).get("cosine_similarity", {}).get("mean"),
        "input_prd_vs_clean": input_prd,
        "recon_prd_vs_clean": recon_prd,
        "recon_error_ratio_vs_input": _safe_div(recon_prd, input_prd),
        "relative_prd_reduction_pct": None if _safe_div(denoise_prd, input_prd) is None else 100.0 * _safe_div(denoise_prd, input_prd),
        "denoise_delta_prd": denoise_prd,
        "input_cosine_vs_clean": input_cosine,
        "recon_cosine_vs_clean": recon_cosine,
        "cosine_delta_vs_input": None if input_cosine is None or recon_cosine is None else float(recon_cosine - input_cosine),
        "recon_coherence_vs_clean": out_sp.get("coherence", {}).get("mean"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-cr", type=int, choices=(4, 8), default=8)
    ap.add_argument("--noise-pct", type=float, default=0.8)
    ap.add_argument("--n-windows", type=int, default=24)
    args = ap.parse_args()

    sample_rate = 64
    frame_size = 320
    clean = _build_clean_windows(args.n_windows, frame_size, sample_rate)
    noisy = clean.copy() if args.noise_pct == 0.0 else _add_gaussian(clean, args.noise_pct, seed=17)

    if args.target_cr == 4:
        rvq_run = Path("results/ppg_rvq_64hz_04x_golden")
    else:
        rvq_run = Path("results/ppg_rvq_64hz_08x_golden")

    spiht = SpihtAcCodec(
        name=f"spiht_{args.target_cr}x",
        modality="ppg",
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=float(args.target_cr),
    )
    rvq = RvqCodec.from_run_dir(rvq_run, modality="ppg")

    print(f"[setup] Generated {args.n_windows} clean windows and noisy copies at noise_pct={args.noise_pct:.2f}")
    print(f"[setup] Running SPIHT {args.target_cr}x")
    spiht_recon = _encode_decode_batch(spiht, noisy)
    print(f"[setup] Running RVQ {args.target_cr}x from {rvq_run}")
    rvq_recon = _encode_decode_batch(rvq, noisy)

    out_root = Path("results/_synthetic_scorecards") / f"ppg_{args.target_cr:02d}x_noise{int(round(args.noise_pct * 100)):03d}"
    out_root.mkdir(parents=True, exist_ok=True)
    clean_path = out_root / "clean_truth.npz"
    np.savez(clean_path, clean_truth=clean)

    spiht_dir = out_root / "spiht"
    rvq_dir = out_root / "rvq"
    _write_run_dir(spiht_dir, noisy, spiht_recon)
    _write_run_dir(rvq_dir, noisy, rvq_recon)

    spiht_scorecard = write_quality_scorecard(
        spiht_dir,
        modality="ppg",
        sample_rate=sample_rate,
        clean_reference_path=clean_path,
    )
    rvq_scorecard = write_quality_scorecard(
        rvq_dir,
        modality="ppg",
        sample_rate=sample_rate,
        clean_reference_path=clean_path,
    )

    spiht_card = json.loads(spiht_scorecard.read_text())
    rvq_card = json.loads(rvq_scorecard.read_text())
    baseline_prd = spiht_card["clean_reference"]["input_baseline"]["time_domain"]["prd_percent"]["mean"]
    baseline_cosine = spiht_card["clean_reference"]["input_baseline"]["time_domain"]["cosine_similarity"]["mean"]

    comparison = {
        "target_cr": args.target_cr,
        "noise_pct": args.noise_pct,
        "n_windows": args.n_windows,
        "input_baseline": {
            "prd_vs_clean": baseline_prd,
            "cosine_vs_clean": baseline_cosine,
        },
        "spiht": _extract_headline(spiht_card),
        "rvq": _extract_headline(rvq_card),
        "delta_rvq_minus_spiht": {
            "prd_vs_noisy": rvq_card["time_domain"]["prd_percent"]["mean"] - spiht_card["time_domain"]["prd_percent"]["mean"],
            "prd_vs_clean": rvq_card["clean_reference"]["reconstruction"]["time_domain"]["prd_percent"]["mean"] - spiht_card["clean_reference"]["reconstruction"]["time_domain"]["prd_percent"]["mean"],
            "denoise_delta_prd": rvq_card["clean_reference"]["denoising"]["time_domain"]["prd_percent_improvement"] - spiht_card["clean_reference"]["denoising"]["time_domain"]["prd_percent_improvement"],
            "cosine_vs_clean": rvq_card["clean_reference"]["reconstruction"]["time_domain"]["cosine_similarity"]["mean"] - spiht_card["clean_reference"]["reconstruction"]["time_domain"]["cosine_similarity"]["mean"],
        },
    }
    cmp_path = out_root / "comparison.json"
    cmp_path.write_text(json.dumps(comparison, indent=2))

    print(f"[done] Wrote clean truth to {clean_path}")
    print(f"[done] SPIHT scorecard: {spiht_scorecard}")
    print(f"[done] RVQ scorecard  : {rvq_scorecard}")
    print(f"[done] Comparison     : {cmp_path}")
    print(
        "[summary] "
        f"baseline PRD(clean)={comparison['input_baseline']['prd_vs_clean']:.2f} | "
        f"SPIHT PRD(clean)={comparison['spiht']['recon_prd_vs_clean']:.2f}, "
        f"denoise={comparison['spiht']['denoise_delta_prd']:.2f} "
        f"({comparison['spiht']['relative_prd_reduction_pct']:.1f}% rel) | "
        f"RVQ PRD(clean)={comparison['rvq']['recon_prd_vs_clean']:.2f}, "
        f"denoise={comparison['rvq']['denoise_delta_prd']:.2f} "
        f"({comparison['rvq']['relative_prd_reduction_pct']:.1f}% rel)"
    )


if __name__ == "__main__":
    main()
