"""Codec noise sweep: PPG DSP vs RVQ across noise and artifact levels.

Generates clean synthetic PPG windows, adds Gaussian noise at increasing
levels, runs each window through configured DSP and RVQ codecs, and reports
two parallel metric tracks:

* **Faithfulness** -- recon vs the noisy input the codec received.
* **Denoising / truth** -- recon vs the clean ground truth.

The headline plot stacks both tracks so you can see, per noise level:

* which codec preserves the input most faithfully (top row), and
* which codec is closest to the underlying truth (bottom row).

For ``gaussian``, the level is parameterised as ``noise_pct = noise_std / clean_std``
so ``0`` = clean (inf SNR) and ``1.0`` = SNR 0 dB. For structured artefacts,
the same ``--noise-pcts`` values are interpreted as a severity control.

Usage::

    uv run python scripts/sweep_codec_noise_ppg.py
    uv run python scripts/sweep_codec_noise_ppg.py --n-windows 64 \
        --noise-pcts 0,0.1,0.2,0.4,0.8,1.2 --crs 4,8 \
        --dsp-modes default,tuned,tuned_ac
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Silence TF noise.
import os

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from compressionkit.evaluation.codec import SpihtAcCodec
from compressionkit.evaluation.empirical_regime import corr as _corr
from compressionkit.evaluation.empirical_regime import encode_decode_batch
from compressionkit.evaluation.empirical_regime import prd as _prd
from compressionkit.evaluation.rvq_codec import RvqCodec
from compressionkit.synthetic import ppg_dynamical
from compressionkit.synthetic.noise import NoiseSpec, add_noise

PPG_DEFAULT_WAVELET = "coif5"
PPG_DEFAULT_LEVELS = 6
PPG_TUNED_WAVELET = "bior4.4"
PPG_TUNED_LEVELS = 6


def _parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one float value")
    return values


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


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


def _severity_to_snr_db(severity: float) -> float:
    return float(30.0 - 30.0 * severity)


def add_structured_noise(
    clean: np.ndarray,
    severity: float,
    seed: int,
    *,
    sample_rate: float,
    spec: NoiseSpec,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty_like(clean)
    snr_db = _severity_to_snr_db(severity)
    for i, c in enumerate(clean):
        noisy, _noise = add_noise(
            c,
            sample_rate=sample_rate,
            snr_db=snr_db,
            spec=spec,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        x = noisy.astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-9)
        out[i] = x
    return out


def add_dropout(clean: np.ndarray, severity: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty_like(clean)
    for i, c in enumerate(clean):
        x = c.copy()
        frac = min(max(0.05 + 0.30 * severity, 0.02), 0.5)
        width = max(4, round(frac * x.shape[0]))
        start = int(rng.integers(0, max(1, x.shape[0] - width + 1)))
        x[start : start + width] = 0.0
        x = (x - x.mean()) / (x.std() + 1e-9)
        out[i] = x
    return out


def add_clipping(clean: np.ndarray, severity: float, seed: int) -> np.ndarray:
    del seed
    out = np.empty_like(clean)
    threshold = max(0.35, 2.5 - 1.8 * severity)
    for i, c in enumerate(clean):
        x = np.clip(c, -threshold, threshold)
        x = (x - x.mean()) / (x.std() + 1e-9)
        out[i] = x.astype(np.float32)
    return out


def corrupt_windows(
    clean: np.ndarray,
    *,
    corruption: str,
    level: float,
    seed: int,
    sample_rate: float,
) -> np.ndarray:
    if level <= 0.0:
        return clean.copy()
    if corruption == "gaussian":
        return add_gaussian(clean, level, seed=seed)
    if corruption == "baseline_wander":
        return add_structured_noise(
            clean,
            level,
            seed,
            sample_rate=sample_rate,
            spec=NoiseSpec(weights={"baseline_wander": 1.0}),
        )
    if corruption == "motion":
        return add_structured_noise(
            clean,
            level,
            seed,
            sample_rate=sample_rate,
            spec=NoiseSpec(weights={"motion": 1.0}),
        )
    if corruption == "wearable_mix":
        return add_structured_noise(
            clean,
            level,
            seed,
            sample_rate=sample_rate,
            spec=NoiseSpec(weights={"baseline_wander": 1.0, "motion": 1.5, "gauss": 0.5}),
        )
    if corruption == "dropout":
        return add_dropout(clean, level, seed=seed)
    if corruption == "clipping":
        return add_clipping(clean, level, seed=seed)
    raise ValueError(f"Unsupported corruption: {corruption}")


def _level_label(corruption: str) -> str:
    if corruption == "gaussian":
        return "Gaussian noise (noise_std / signal_std)"
    if corruption in {"baseline_wander", "motion", "wearable_mix"}:
        return "Artifact severity (mapped to lower input SNR)"
    return "Artifact severity"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def codec_metrics(clean: np.ndarray, noisy: np.ndarray, recon: np.ndarray) -> dict[str, float]:
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


def _build_dsp_codecs(
    crs: list[int], dsp_modes: list[str], frame_size: int, sample_rate: int
) -> list[tuple[str, object, str, str]]:
    mode_styles = {"default": ("#1f77b4", "-"), "tuned": ("#2ca02c", "--"), "tuned_ac": ("#9467bd", "-.")}
    codecs: list[tuple[str, object, str, str]] = []
    for cr in crs:
        for mode in dsp_modes:
            if mode == "default":
                codec = SpihtAcCodec(
                    name=f"spiht_default_{cr}x",
                    modality="ppg",
                    sample_rate=sample_rate,
                    frame_size=frame_size,
                    target_cr=float(cr),
                    wavelet=PPG_DEFAULT_WAVELET,
                    levels=PPG_DEFAULT_LEVELS,
                    use_ac=True,
                )
                label = f"SPIHT default {cr}x"
            elif mode == "tuned":
                codec = SpihtAcCodec(
                    name=f"spiht_tuned_{cr}x",
                    modality="ppg",
                    sample_rate=sample_rate,
                    frame_size=frame_size,
                    target_cr=float(cr),
                    wavelet=PPG_TUNED_WAVELET,
                    levels=PPG_TUNED_LEVELS,
                    use_ac=False,
                )
                label = f"SPIHT tuned {cr}x"
            elif mode == "tuned_ac":
                codec = SpihtAcCodec(
                    name=f"spiht_tuned_ac_{cr}x",
                    modality="ppg",
                    sample_rate=sample_rate,
                    frame_size=frame_size,
                    target_cr=float(cr),
                    wavelet=PPG_TUNED_WAVELET,
                    levels=PPG_TUNED_LEVELS,
                    use_ac=True,
                )
                label = f"SPIHT tuned+AC {cr}x"
            else:
                raise ValueError(f"Unsupported DSP mode: {mode}")
            color, line_style = mode_styles[mode]
            codecs.append((label, codec, color, line_style))
    return codecs


def _load_rvq_codecs(crs: list[int], rvq_root: Path) -> list[tuple[str, object, str, str]]:
    codecs: list[tuple[str, object, str, str]] = []
    styles = {2: ("#d62728", ":"), 4: ("#d62728", "-"), 8: ("#d62728", "--"), 16: ("#d62728", "-.")}
    for cr in crs:
        run_dir = rvq_root / f"ppg_rvq_64hz_{cr:02d}x_golden"
        if not run_dir.exists():
            print(f"[warn] Skipping RVQ {cr}x; missing {run_dir}")
            continue
        print(f"[setup] Loading RVQ {cr}x golden ...")
        codec = RvqCodec.from_run_dir(run_dir, modality="ppg")
        color, line_style = styles.get(cr, ("#d62728", "-"))
        codecs.append((f"RVQ {cr}x", codec, color, line_style))
    return codecs


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/_codec_noise_sweep")
    parser.add_argument("--n-windows", type=int, default=24)
    parser.add_argument("--sample-rate", type=float, default=64.0)
    parser.add_argument("--frame-size", type=int, default=320)
    parser.add_argument("--noise-pcts", default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument(
        "--corruption",
        default="gaussian",
        choices=["gaussian", "baseline_wander", "motion", "wearable_mix", "dropout", "clipping"],
    )
    parser.add_argument("--crs", default="4,8")
    parser.add_argument(
        "--dsp-modes",
        default="default,tuned,tuned_ac",
        help="Comma-separated DSP baselines: default,tuned,tuned_ac",
    )
    parser.add_argument("--rvq-root", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = float(args.sample_rate)
    frame_size = int(args.frame_size)
    n_windows = int(args.n_windows)
    noise_pcts = _parse_float_list(args.noise_pcts)
    corruption = str(args.corruption)
    crs = _parse_int_list(args.crs)
    dsp_modes = [part.strip() for part in args.dsp_modes.split(",") if part.strip()]

    print(f"[setup] Building {n_windows} clean PPG windows ({frame_size} samples @ {sample_rate} Hz)")
    clean = build_clean_windows(n_windows, frame_size, sample_rate)

    print(f"[setup] Building DSP codecs for CRs {crs} with modes {dsp_modes}")
    codecs = _build_dsp_codecs(crs, dsp_modes, frame_size, int(sample_rate))
    codecs.extend(_load_rvq_codecs(crs, Path(args.rvq_root)))
    if not codecs:
        raise SystemExit("No codecs available for sweep")

    summary: dict = {
        "corruption": corruption,
        "noise_pcts": noise_pcts,
        "crs": crs,
        "dsp_modes": dsp_modes,
        "codecs": {},
        "baselines": [],
    }

    for label, _codec, _, _ in codecs:
        summary["codecs"][label] = {"noise_pct": [], "metrics": []}

    for npct in noise_pcts:
        print(f"\n[{corruption} severity = {npct:.2f}]")
        noisy = corrupt_windows(
            clean,
            corruption=corruption,
            level=npct,
            seed=int(npct * 1000) + 17,
            sample_rate=sample_rate,
        )

        # Baseline (no codec): how close is noisy to clean?
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

    json_path = out_dir / f"summary_{corruption}.json"
    json_path.write_text(
        json.dumps(summary, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o)
    )
    print(f"\nWrote {json_path}")

    # --------------------------- plotting ---------------------------
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
        # Baseline = no codec at all (when relevant)
        if key == "prd_vs_clean_mean":
            ys_base = [b["noisy_vs_clean_prd"] for b in summary["baselines"]]
            ax.plot(noise_pcts, ys_base, color="#888", lw=1.2, ls=":", marker="x", label="noisy input (no codec)")
        if key == "corr_vs_clean_mean":
            ys_base = [b["noisy_vs_clean_corr"] for b in summary["baselines"]]
            ax.plot(noise_pcts, ys_base, color="#888", lw=1.2, ls=":", marker="x", label="noisy input (no codec)")
        if key == "denoising_delta_prd":
            ax.axhline(0, color="#888", lw=0.7, ls=":")
        ax.set_title(subtitle, fontsize=10, loc="left")
        ax.set_xlabel(_level_label(corruption))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
    fig.suptitle(f"PPG codec behaviour under increasing {corruption} severity — DSP vs RVQ", fontsize=11)
    fig.tight_layout()
    png_path = out_dir / f"ppg_codec_{corruption}_sweep.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
