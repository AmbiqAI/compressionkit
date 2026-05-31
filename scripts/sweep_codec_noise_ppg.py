"""Codec noise sweep: PPG SPIHT vs PPG RVQ across 0-100% Gaussian noise.

Generates clean synthetic PPG windows, adds Gaussian noise at increasing
levels, runs each window through SPIHT and RVQ codecs at 4x and 8x, and
reports two parallel metric tracks:

* **Faithfulness** -- recon vs the noisy input the codec received.
* **Denoising / truth** -- recon vs the clean ground truth.

The headline plot stacks both tracks so you can see, per noise level:

* which codec preserves the input most faithfully (top row), and
* which codec is closest to the underlying truth (bottom row).

Noise is parameterised as ``noise_pct = noise_std / clean_std`` so
``0%`` = clean (inf SNR) and ``100%`` = SNR 0 dB. The script also dumps
``results/_codec_noise_sweep.json`` with all per-noise-level numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Silence TF noise.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from compressionkit.evaluation.codec import SpihtAcCodec
from compressionkit.evaluation.rvq_codec import RvqCodec
from compressionkit.synthetic import ppg_dynamical


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def build_clean_windows(n_windows: int, frame_size: int, sample_rate: float) -> np.ndarray:
    """Generate ``n_windows`` clean PPG windows with HR jitter and a pre-roll."""
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
        # Per-window z-norm to match scorecard convention.
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        out[i] = sig
    return out


def add_gaussian(clean: np.ndarray, noise_pct: float, seed: int) -> np.ndarray:
    """Add white Gaussian at ``std = noise_pct * clean.std()``, then re-znorm."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(clean)
    for i, c in enumerate(clean):
        noise = rng.standard_normal(c.shape).astype(np.float32) * (noise_pct * c.std() + 1e-9)
        x = c + noise
        x = (x - x.mean()) / (x.std() + 1e-9)
        out[i] = x
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _prd(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Percent root-mean-square diff per row. PRD = 100 * ||ref-est|| / ||ref||."""
    num = np.linalg.norm(ref - est, axis=-1)
    den = np.linalg.norm(ref, axis=-1) + 1e-12
    return 100.0 * num / den


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation per row."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1)) + 1e-12
    return num / den


def codec_metrics(
    clean: np.ndarray, noisy: np.ndarray, recon: np.ndarray
) -> dict[str, float]:
    """Compute both tracks of metrics for one (codec, noise-level) cell."""
    prd_noisy = _prd(noisy, recon)
    prd_clean = _prd(clean, recon)
    corr_noisy = _corr(noisy, recon)
    corr_clean = _corr(clean, recon)
    # How well does the *noisy input* itself compare to truth? Useful baseline
    # ("no codec at all").
    prd_noisy_vs_clean = _prd(clean, noisy)
    corr_noisy_vs_clean = _corr(clean, noisy)
    return {
        "prd_vs_noisy_mean": float(prd_noisy.mean()),
        "prd_vs_clean_mean": float(prd_clean.mean()),
        "corr_vs_noisy_mean": float(corr_noisy.mean()),
        "corr_vs_clean_mean": float(corr_clean.mean()),
        # Denoising delta: how much closer to truth is the recon than the noisy
        # input? Positive => codec denoises.
        "denoising_delta_prd": float(prd_noisy_vs_clean.mean() - prd_clean.mean()),
        # Identity baselines for the same cell
        "_noisy_vs_clean_prd": float(prd_noisy_vs_clean.mean()),
        "_noisy_vs_clean_corr": float(corr_noisy_vs_clean.mean()),
    }


# ---------------------------------------------------------------------------
# Codec runners
# ---------------------------------------------------------------------------


def encode_decode_batch(codec, frames: np.ndarray) -> np.ndarray:
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        enc = codec.encode(f)
        rec = codec.decode(enc)
        # Some codecs return (T,) or (T,1) -- flatten
        rec = np.asarray(rec, dtype=np.float32).reshape(-1)[: frames.shape[1]]
        # Match per-window z-norm of the input
        rec = (rec - rec.mean()) / (rec.std() + 1e-9)
        out[i] = rec
    return out


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path("results/_codec_noise_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 64.0
    frame_size = 320
    n_windows = 24
    noise_pcts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    print(f"[setup] Building {n_windows} clean PPG windows ({frame_size} samples @ {sample_rate} Hz)")
    clean = build_clean_windows(n_windows, frame_size, sample_rate)

    print("[setup] Building SPIHT codecs (4x, 8x)")
    spiht_4 = SpihtAcCodec(name="spiht_4x", modality="ppg", sample_rate=64,
                           frame_size=frame_size, target_cr=4.0)
    spiht_8 = SpihtAcCodec(name="spiht_8x", modality="ppg", sample_rate=64,
                           frame_size=frame_size, target_cr=8.0)

    print("[setup] Loading RVQ 4x golden ...")
    rvq_4 = RvqCodec.from_run_dir("results/ppg_rvq_64hz_04x_golden", modality="ppg")
    print("[setup] Loading RVQ 8x golden ...")
    rvq_8 = RvqCodec.from_run_dir("results/ppg_rvq_64hz_08x_golden", modality="ppg")

    codecs = [
        ("SPIHT 4x", spiht_4, "#1f77b4", "-"),
        ("SPIHT 8x", spiht_8, "#1f77b4", "--"),
        ("RVQ 4x",   rvq_4,   "#d62728", "-"),
        ("RVQ 8x",   rvq_8,   "#d62728", "--"),
    ]

    summary: dict = {"noise_pcts": noise_pcts, "codecs": {}, "baselines": []}

    for label, codec, _, _ in codecs:
        summary["codecs"][label] = {"noise_pct": [], "metrics": []}

    for npct in noise_pcts:
        print(f"\n[noise_pct = {npct:.2f}]")
        noisy = clean.copy() if npct == 0.0 else add_gaussian(clean, npct, seed=int(npct * 1000))

        # Baseline (no codec): how close is noisy to clean?
        base = {
            "noisy_vs_clean_prd": float(_prd(clean, noisy).mean()),
            "noisy_vs_clean_corr": float(_corr(clean, noisy).mean()),
        }
        summary["baselines"].append({"noise_pct": npct, **base})
        print(f"  baseline (noisy vs clean): PRD={base['noisy_vs_clean_prd']:.1f}%  "
              f"corr={base['noisy_vs_clean_corr']:.3f}")

        for label, codec, _, _ in codecs:
            recon = encode_decode_batch(codec, noisy)
            m = codec_metrics(clean, noisy, recon)
            summary["codecs"][label]["noise_pct"].append(npct)
            summary["codecs"][label]["metrics"].append(m)
            print(f"  {label:8s}: "
                  f"PRDn={m['prd_vs_noisy_mean']:5.1f}%  "
                  f"PRDc={m['prd_vs_clean_mean']:5.1f}%  "
                  f"corr_c={m['corr_vs_clean_mean']:.3f}  "
                  f"denoise_delta={m['denoising_delta_prd']:+5.1f}%")

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2,
                                    default=lambda o: None if isinstance(o, float) and math.isnan(o) else o))
    print(f"\nWrote {json_path}")

    # --------------------------- plotting ---------------------------
    metric_keys = [
        ("prd_vs_noisy_mean",  "PRD vs noisy input (%)",   "faithfulness — lower = preserves what codec saw"),
        ("prd_vs_clean_mean",  "PRD vs CLEAN truth (%)",   "truth — lower = closer to underlying signal"),
        ("corr_vs_clean_mean", "corr vs CLEAN truth",       "truth — higher = closer to underlying signal"),
        ("denoising_delta_prd", "denoising Δ PRD (%)",      "positive = codec denoises; negative = codec adds distortion"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    axes = axes.flatten()
    for ax, (key, ylabel, subtitle) in zip(axes, metric_keys):
        for label, codec, color, ls in codecs:
            ys = [m[key] for m in summary["codecs"][label]["metrics"]]
            ax.plot(noise_pcts, ys, color=color, linestyle=ls, marker="o", lw=1.8,
                    label=label)
        # Baseline = no codec at all (when relevant)
        if key == "prd_vs_clean_mean":
            ys_base = [b["noisy_vs_clean_prd"] for b in summary["baselines"]]
            ax.plot(noise_pcts, ys_base, color="#888", lw=1.2, ls=":", marker="x",
                    label="noisy input (no codec)")
        if key == "corr_vs_clean_mean":
            ys_base = [b["noisy_vs_clean_corr"] for b in summary["baselines"]]
            ax.plot(noise_pcts, ys_base, color="#888", lw=1.2, ls=":", marker="x",
                    label="noisy input (no codec)")
        if key == "denoising_delta_prd":
            ax.axhline(0, color="#888", lw=0.7, ls=":")
        ax.set_title(subtitle, fontsize=10, loc="left")
        ax.set_xlabel("Gaussian noise (noise_std / signal_std)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
    fig.suptitle("PPG codec behaviour under increasing Gaussian noise — synthetic ground truth",
                 fontsize=11)
    fig.tight_layout()
    png_path = out_dir / "ppg_codec_noise_sweep.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
