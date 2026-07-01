"""ECG codec noise sweep — counterpart to ``sweep_codec_noise_ppg.py``.

Same methodology: synthetic clean ECG (McSharry) → add Gaussian noise at
``noise_std / clean_std`` ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} → run through
SPIHT 4×/8× and RVQ 4×/8× → score against both noisy input (faithfulness)
and clean truth (denoising). Outputs a 4-panel PNG and a summary JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compressionkit.evaluation.codec import SpihtAcCodec
from compressionkit.evaluation.empirical_regime import corr as _corr
from compressionkit.evaluation.empirical_regime import encode_decode_batch
from compressionkit.evaluation.empirical_regime import prd as _prd
from compressionkit.evaluation.rvq_codec import RvqCodec
from compressionkit.synthetic import ecg_mcsharry

# ---------------------------------------------------------------------------
# Helpers (same shape as the PPG sweep)
# ---------------------------------------------------------------------------


def build_clean_windows(n_windows: int, frame_size: int, sample_rate: float) -> np.ndarray:
    """Generate ``n_windows`` clean ECG windows with HR jitter and a pre-roll."""
    rng = np.random.default_rng(0)
    out = np.empty((n_windows, frame_size), dtype=np.float32)
    pre_roll_s = 1.5
    window_s = frame_size / sample_rate
    n_pre = int(pre_roll_s * sample_rate)
    for i in range(n_windows):
        hr = float(rng.uniform(55.0, 95.0))
        sig = ecg_mcsharry(
            duration_s=pre_roll_s + window_s,
            sample_rate=sample_rate,
            hr_mean=hr,
            hr_std=1.5,
            seed=3000 + i,
        )
        sig = sig[n_pre : n_pre + frame_size].astype(np.float32)
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        out[i] = sig
    return out


def add_gaussian(clean: np.ndarray, noise_pct: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty_like(clean)
    for i, c in enumerate(clean):
        noise = rng.standard_normal(c.shape).astype(np.float32) * (noise_pct * c.std() + 1e-9)
        x = c + noise
        x = (x - x.mean()) / (x.std() + 1e-9)
        out[i] = x
    return out


def codec_metrics(clean: np.ndarray, noisy: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    prd_n = _prd(noisy, recon)
    prd_c = _prd(clean, recon)
    corr_n = _corr(noisy, recon)
    corr_c = _corr(clean, recon)
    prd_nvc = _prd(clean, noisy)
    corr_nvc = _corr(clean, noisy)
    return {
        "prd_vs_noisy_mean": float(prd_n.mean()),
        "prd_vs_clean_mean": float(prd_c.mean()),
        "corr_vs_noisy_mean": float(corr_n.mean()),
        "corr_vs_clean_mean": float(corr_c.mean()),
        "denoising_delta_prd": float(prd_nvc.mean() - prd_c.mean()),
        "_noisy_vs_clean_prd": float(prd_nvc.mean()),
        "_noisy_vs_clean_corr": float(corr_nvc.mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_noise_pcts(text: str) -> list[float]:
    vals = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not vals:
        raise ValueError("--noise-pcts must provide at least one comma-separated value")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-windows", type=int, default=24)
    ap.add_argument(
        "--noise-pcts",
        type=_parse_noise_pcts,
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Comma-separated Gaussian noise levels as noise_std / signal_std.",
    )
    ap.add_argument(
        "--rvq-4x-run-dir",
        type=Path,
        default=Path("results/ecg_rvq_256hz_04x_golden"),
        help="RVQ 4x run dir to load instead of the faithful default.",
    )
    ap.add_argument(
        "--rvq-8x-run-dir",
        type=Path,
        default=Path("results/ecg_rvq_256hz_08x_golden"),
        help="RVQ 8x run dir to load instead of the faithful default.",
    )
    ap.add_argument(
        "--output-stem",
        type=str,
        default="summary_ecg",
        help="Output stem for <stem>.json and <stem>.png under results/_codec_noise_sweep/.",
    )
    args = ap.parse_args()

    out_dir = Path("results/_codec_noise_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 256.0
    frame_size = 512  # matches ECG RVQ goldens
    n_windows = args.n_windows
    noise_pcts = args.noise_pcts

    print(
        f"[setup] Building {n_windows} clean ECG windows ({frame_size} samples @ {sample_rate} Hz, ~{frame_size / sample_rate:.1f} s)"
    )
    clean = build_clean_windows(n_windows, frame_size, sample_rate)

    print("[setup] Building SPIHT codecs (4x, 8x)")
    spiht_4 = SpihtAcCodec(
        name="spiht_4x", modality="ecg", sample_rate=int(sample_rate), frame_size=frame_size, target_cr=4.0
    )
    spiht_8 = SpihtAcCodec(
        name="spiht_8x", modality="ecg", sample_rate=int(sample_rate), frame_size=frame_size, target_cr=8.0
    )

    if not args.rvq_4x_run_dir.exists():
        raise FileNotFoundError(f"RVQ 4x run dir does not exist: {args.rvq_4x_run_dir}")
    if not args.rvq_8x_run_dir.exists():
        raise FileNotFoundError(f"RVQ 8x run dir does not exist: {args.rvq_8x_run_dir}")

    print(f"[setup] Loading ECG RVQ 4x from {args.rvq_4x_run_dir} ...")
    rvq_4 = RvqCodec.from_run_dir(args.rvq_4x_run_dir, modality="ecg")
    print(f"[setup] Loading ECG RVQ 8x from {args.rvq_8x_run_dir} ...")
    rvq_8 = RvqCodec.from_run_dir(args.rvq_8x_run_dir, modality="ecg")

    codecs = [
        ("SPIHT 4x", spiht_4, "#1f77b4", "-"),
        ("SPIHT 8x", spiht_8, "#1f77b4", "--"),
        ("RVQ 4x", rvq_4, "#d62728", "-"),
        ("RVQ 8x", rvq_8, "#d62728", "--"),
    ]

    summary: dict = {"noise_pcts": noise_pcts, "codecs": {}, "baselines": []}
    for label, *_ in codecs:
        summary["codecs"][label] = {"noise_pct": [], "metrics": []}

    for npct in noise_pcts:
        print(f"\n[noise_pct = {npct:.2f}]")
        noisy = clean.copy() if npct == 0.0 else add_gaussian(clean, npct, seed=int(npct * 1000) + 17)

        base = {
            "noisy_vs_clean_prd": float(_prd(clean, noisy).mean()),
            "noisy_vs_clean_corr": float(_corr(clean, noisy).mean()),
        }
        summary["baselines"].append({"noise_pct": npct, **base})
        print(
            f"  baseline (noisy vs clean): PRD={base['noisy_vs_clean_prd']:.1f}%  "
            f"corr={base['noisy_vs_clean_corr']:.3f}"
        )

        for label, codec, _, _ in codecs:
            recon = encode_decode_batch(codec, noisy)
            m = codec_metrics(clean, noisy, recon)
            summary["codecs"][label]["noise_pct"].append(npct)
            summary["codecs"][label]["metrics"].append(m)
            print(
                f"  {label:8s}: "
                f"PRDn={m['prd_vs_noisy_mean']:5.1f}%  "
                f"PRDc={m['prd_vs_clean_mean']:5.1f}%  "
                f"corr_c={m['corr_vs_clean_mean']:.3f}  "
                f"denoise_delta={m['denoising_delta_prd']:+5.1f}%"
            )

    json_path = out_dir / f"{args.output_stem}.json"
    json_path.write_text(
        json.dumps(summary, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o)
    )
    print(f"\nWrote {json_path}")

    metric_keys = [
        ("prd_vs_noisy_mean", "PRD vs noisy input (%)", "faithfulness — lower = preserves what codec saw"),
        ("prd_vs_clean_mean", "PRD vs CLEAN truth (%)", "truth — lower = closer to underlying signal"),
        ("corr_vs_clean_mean", "corr vs CLEAN truth", "truth — higher = closer to underlying signal"),
        ("denoising_delta_prd", "denoising Δ PRD (%)", "positive = codec denoises; negative = codec adds distortion"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    axes = axes.flatten()
    for ax, (key, ylabel, subtitle) in zip(axes, metric_keys):
        for label, _codec, color, ls in codecs:
            ys = [m[key] for m in summary["codecs"][label]["metrics"]]
            ax.plot(noise_pcts, ys, color=color, linestyle=ls, marker="o", lw=1.8, label=label)
        if key == "prd_vs_clean_mean":
            ys_base = [b["noisy_vs_clean_prd"] for b in summary["baselines"]]
            ax.plot(noise_pcts, ys_base, color="#888", lw=1.2, ls=":", marker="x", label="noisy input (no codec)")
        if key == "corr_vs_clean_mean":
            ys_base = [b["noisy_vs_clean_corr"] for b in summary["baselines"]]
            ax.plot(noise_pcts, ys_base, color="#888", lw=1.2, ls=":", marker="x", label="noisy input (no codec)")
        if key == "denoising_delta_prd":
            ax.axhline(0, color="#888", lw=0.7, ls=":")
        ax.set_title(subtitle, fontsize=10, loc="left")
        ax.set_xlabel("Gaussian noise (noise_std / signal_std)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
    fig.suptitle("ECG codec behaviour under increasing Gaussian noise — synthetic ground truth", fontsize=11)
    fig.tight_layout()
    png_path = out_dir / f"{args.output_stem}.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
