"""CLI for running the codec evaluation harness end-to-end.

Wires together the building blocks added in PRs 1-6:

    * :class:`compressionkit.evaluation.Codec` adapters
      (:class:`SpihtAcCodec`, :class:`RvqCodec`)
    * Fidelity / spectral metrics
    * Adversarial input battery
    * Stitching A/B
    * RVQ QoS calibration

Outputs a JSON manifest + markdown summary suitable for paste into the
scorecard or customer-meeting slides.

Example
-------
SPIHT-AC PPG @ 4x on synthetic data::

    python scripts/eval_codec.py \\
        --codec spiht_ac --modality ppg --cr 4 \\
        --frame-size 320 --sample-rate 64 \\
        --tiers fidelity adversarial stitching \\
        --out results/evals/spiht_ppg_04x

RVQ PPG golden 4x with QoS::

    python scripts/eval_codec.py \\
        --codec rvq --modality ppg \\
        --rvq-run results/ppg_rvq_64hz_04x_golden \\
        --tiers fidelity adversarial stitching qos \\
        --out results/evals/rvq_ppg_04x_golden

The CLI deliberately keeps the data source synthetic (or user-supplied
``.npy``) so the harness has no dataset-specific dependencies. Customers can
plug in their own waveforms via ``--signal-npy``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from compressionkit.evaluation import (
    Codec,
    RvqCodec,
    RvqQoSCalibrator,
    SpihtAcCodec,
    compare_stitching_methods,
    compute_rvq_qos,
    compute_signal_metrics,
    run_adversarial_battery,
)
from compressionkit.evaluation.spectral_metrics import (
    ECG_DEFAULT_BANDS,
    ECG_DEFAULT_COHERENCE_BAND,
    PPG_DEFAULT_BANDS,
    PPG_DEFAULT_COHERENCE_BAND,
    psd_band_error,
    spectral_coherence,
)

TIERS = ("fidelity", "spectral", "adversarial", "stitching", "qos")


# ---------------------------------------------------------------------------
# Signal source
# ---------------------------------------------------------------------------


def _synthetic_signal(modality: str, fs: int, n_samples: int, seed: int) -> np.ndarray:
    """Cheap, dataset-free signal generator for harness smoke runs.

    PPG: 1.2 Hz fundamental + 2.4 Hz overtone + light noise.
    ECG: 1.0 Hz QRS-like train (sinusoid + brief Gaussian bumps) + noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    if modality == "ppg":
        sig = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t) + 0.05 * rng.standard_normal(n_samples)
    elif modality == "ecg":
        # Brief Gaussian bumps to fake QRS complexes at 1 Hz.
        period = round(fs / 1.0)
        bump_width = max(2, round(0.04 * fs))
        sig = 0.1 * rng.standard_normal(n_samples)
        centres = np.arange(period // 2, n_samples, period)
        x = np.arange(-bump_width, bump_width + 1)
        bump = np.exp(-(x**2) / (2 * (bump_width / 3.0) ** 2)).astype(np.float32)
        for c in centres:
            lo, hi = max(0, c - bump_width), min(n_samples, c + bump_width + 1)
            sig[lo:hi] += bump[: hi - lo]
    else:
        raise ValueError(f"Unknown modality {modality!r}")
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    return sig.astype(np.float32)


def _load_signal(args: argparse.Namespace) -> np.ndarray:
    if args.signal_npy:
        sig = np.load(args.signal_npy).astype(np.float32).reshape(-1)
        if sig.size < args.frame_size * 2:
            raise ValueError(f"--signal-npy too short ({sig.size}); need >= 2 * frame_size = {2 * args.frame_size}")
        return sig
    n_samples = max(args.frame_size * 16, args.frame_size * 2)
    return _synthetic_signal(args.modality, args.sample_rate, n_samples, seed=args.seed)


def _frame_signal(signal: np.ndarray, frame_size: int) -> np.ndarray:
    """Slice a long signal into non-overlapping frames."""
    n_frames = signal.size // frame_size
    return signal[: n_frames * frame_size].reshape(n_frames, frame_size)


# ---------------------------------------------------------------------------
# Codec construction
# ---------------------------------------------------------------------------


def _build_codec(args: argparse.Namespace) -> Codec:
    if args.codec == "spiht_ac":
        return SpihtAcCodec(
            name=f"spiht_ac_{args.modality}_{int(args.cr):02d}x",
            modality=args.modality,
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
            target_cr=float(args.cr),
        )
    if args.codec == "rvq":
        if not args.rvq_run:
            raise SystemExit("--rvq-run is required when --codec rvq")
        return RvqCodec.from_run_dir(Path(args.rvq_run), modality=args.modality)
    raise SystemExit(f"Unknown codec {args.codec!r}")


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------


def tier_fidelity(codec: Codec, frames: np.ndarray) -> dict[str, Any]:
    """Per-frame fidelity metrics aggregated to mean / std / median / p10 / p90."""
    rows = []
    for f in frames:
        dec = np.asarray(codec.decode(codec.encode(f)), dtype=np.float32).reshape(-1)
        rows.append(compute_signal_metrics(f, dec))
    keys = rows[0].keys()
    summary: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=np.float64)
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90)),
        }
    return {"n_frames": len(rows), "metrics": summary}


def tier_spectral(codec: Codec, frames: np.ndarray, fs: int, modality: str) -> dict[str, Any]:
    bands = PPG_DEFAULT_BANDS if modality == "ppg" else ECG_DEFAULT_BANDS
    coh_band = PPG_DEFAULT_COHERENCE_BAND if modality == "ppg" else ECG_DEFAULT_COHERENCE_BAND
    band_rows, coh_rows = [], []
    for f in frames:
        dec = np.asarray(codec.decode(codec.encode(f)), dtype=np.float32).reshape(-1)
        band_rows.append(psd_band_error(f, dec, fs=fs, bands=bands))
        coh_rows.append(spectral_coherence(f, dec, fs=fs, band=coh_band))
    band_keys = band_rows[0].keys()
    band_summary = {k: float(np.mean([r[k] for r in band_rows])) for k in band_keys}
    coh_keys = coh_rows[0].keys()
    coh_summary = {k: float(np.mean([r[k] for r in coh_rows])) for k in coh_keys}
    return {"band_error": band_summary, "coherence": coh_summary}


def tier_adversarial(codec: Codec, signal: np.ndarray) -> dict[str, Any]:
    # Use a chunk of the signal as the "valid signal" bank for inject_* scenarios.
    n = codec.frame_size
    n_frames = min(8, max(2, signal.size // n))
    signal_frames = _frame_signal(signal[: n * n_frames], n)
    results = run_adversarial_battery(codec, n_frames=8, signal_frames=signal_frames)
    return {r.test_name: r.to_dict() for r in results}


def tier_stitching(codec: Codec, signal: np.ndarray) -> dict[str, Any]:
    df = compare_stitching_methods(codec, signal, hop_ratios=(0.25, 0.5, 0.75))
    return {"rows": df.to_dict(orient="records")}


def tier_qos(codec: Codec, frames: np.ndarray) -> dict[str, Any]:
    if not isinstance(codec, RvqCodec):
        return {"skipped": True, "reason": "qos tier requires --codec rvq"}
    # Split frames 50/50 into calibration and scoring.
    half = max(1, len(frames) // 2)
    cal_frames, score_frames = frames[:half], frames[half:]
    if len(cal_frames) == 0 or len(score_frames) == 0:
        return {"skipped": True, "reason": "not enough frames for qos split"}
    cal_qos = [compute_rvq_qos(codec.encode(f)) for f in cal_frames]
    cal = RvqQoSCalibrator().fit(cal_qos)
    score_qos = [compute_rvq_qos(codec.encode(f)) for f in score_frames]
    confidences = [cal.score(q) for q in score_qos]
    arr = np.asarray(confidences, dtype=np.float64)
    return {
        "n_calibration": len(cal_frames),
        "n_scored": len(score_frames),
        "confidence": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        },
    }


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Codec eval — {report['codec']['name']}")
    lines.append("")
    lines.append(f"- modality: `{report['codec']['modality']}`")
    lines.append(f"- sample_rate: `{report['codec']['sample_rate']} Hz`")
    lines.append(f"- frame_size: `{report['codec']['frame_size']}`")
    lines.append(f"- target_cr: `{report['codec']['target_cr']}`")
    lines.append(f"- n_frames: `{report['n_frames']}`")
    lines.append("")
    if "fidelity" in report["tiers"]:
        m = report["tiers"]["fidelity"]["metrics"]
        lines.append("## Fidelity")
        lines.append("")
        lines.append("| metric | mean | std | median | p10 | p90 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for k, v in m.items():
            lines.append(
                f"| {k} | {v['mean']:.4f} | {v['std']:.4f} | {v['median']:.4f} | {v['p10']:.4f} | {v['p90']:.4f} |"
            )
        lines.append("")
    if "adversarial" in report["tiers"]:
        lines.append("## Adversarial battery")
        lines.append("")
        lines.append("| test | n | energy_ratio | output_l2_zero | halluc_peaks | prd_mean |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, row in report["tiers"]["adversarial"].items():
            er = row.get("energy_ratio")
            er_s = "inf" if er is None or (isinstance(er, float) and np.isinf(er)) else f"{er:.3f}"

            def _f(x, fmt=".3f"):
                if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else "inf"
                return format(x, fmt)

            lines.append(
                f"| {name} | {row['n_frames']} | {er_s} | "
                f"{_f(row['output_l2_when_input_zero'])} | "
                f"{_f(row['hallucinated_peaks'], '.2f')} | "
                f"{_f(row['reconstruction_prd_mean'], '.2f')} |"
            )
        lines.append("")
    if "stitching" in report["tiers"]:
        lines.append("## Stitching A/B")
        lines.append("")
        lines.append("| method | hop_ratio | prd | seam_ratio |")
        lines.append("|---|---:|---:|---:|")
        for row in report["tiers"]["stitching"]["rows"]:
            hop = row["hop_ratio"]
            hop_s = "—" if hop != hop else f"{hop:.2f}"  # NaN check
            lines.append(f"| {row['method']} | {hop_s} | {row['prd']:.3f} | {row['seam_ratio']:.3f} |")
        lines.append("")
    if "qos" in report["tiers"]:
        q = report["tiers"]["qos"]
        lines.append("## RVQ QoS")
        lines.append("")
        if q.get("skipped"):
            lines.append(f"_Skipped: {q['reason']}_")
        else:
            c = q["confidence"]
            lines.append(f"- n_calibration: {q['n_calibration']}")
            lines.append(f"- n_scored: {q['n_scored']}")
            lines.append(
                f"- confidence: mean={c['mean']:.3f} median={c['median']:.3f} p10={c['p10']:.3f} p90={c['p90']:.3f}"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codec", required=True, choices=["spiht_ac", "rvq"])
    p.add_argument("--modality", required=True, choices=["ppg", "ecg"])
    p.add_argument("--cr", type=float, default=4.0, help="Target compression ratio (SPIHT). Ignored for RVQ.")
    p.add_argument(
        "--frame-size",
        type=int,
        default=None,
        help="Frame size in samples. Default: 320 (PPG) / 512 (ECG); inferred from RVQ run when --codec rvq.",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Sample rate Hz. Default: 64 (PPG) / 256 (ECG); inferred from RVQ run when --codec rvq.",
    )
    p.add_argument("--rvq-run", type=str, default=None, help="Path to RVQ golden run directory.")
    p.add_argument(
        "--tiers",
        nargs="+",
        default=["fidelity", "adversarial", "stitching"],
        choices=list(TIERS),
        help="Which evaluation tiers to run. Default: fidelity adversarial stitching.",
    )
    p.add_argument(
        "--signal-npy", type=str, default=None, help="Optional .npy file with a 1-D signal. Default: synthetic."
    )
    p.add_argument("--n-frames", type=int, default=16, help="Frames for fidelity / spectral / qos tiers.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True, help="Output directory (created if missing).")
    return p


def _apply_defaults(args: argparse.Namespace) -> None:
    if args.frame_size is None:
        args.frame_size = 320 if args.modality == "ppg" else 512
    if args.sample_rate is None:
        args.sample_rate = 64 if args.modality == "ppg" else 256


def run(args: argparse.Namespace) -> dict[str, Any]:
    _apply_defaults(args)
    codec = _build_codec(args)

    # Override frame_size / sample_rate from the codec when it knows better
    # (e.g. RvqCodec from a saved config).
    args.frame_size = codec.frame_size
    args.sample_rate = codec.sample_rate

    signal = _load_signal(args)
    frames = _frame_signal(signal, codec.frame_size)
    if frames.shape[0] > args.n_frames:
        frames = frames[: args.n_frames]

    report: dict[str, Any] = {
        "codec": {
            "name": codec.name,
            "modality": codec.modality,
            "sample_rate": codec.sample_rate,
            "frame_size": codec.frame_size,
            "target_cr": codec.target_cr,
        },
        "n_frames": int(frames.shape[0]),
        "signal_samples": int(signal.size),
        "tiers": {},
        "elapsed_s": {},
    }

    tier_fns: dict[str, Any] = {
        "fidelity": lambda: tier_fidelity(codec, frames),
        "spectral": lambda: tier_spectral(codec, frames, codec.sample_rate, codec.modality),
        "adversarial": lambda: tier_adversarial(codec, signal),
        "stitching": lambda: tier_stitching(codec, signal),
        "qos": lambda: tier_qos(codec, frames),
    }
    for tier in args.tiers:
        t0 = time.time()
        report["tiers"][tier] = tier_fns[tier]()
        report["elapsed_s"][tier] = round(time.time() - t0, 3)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "report.md").write_text(_render_markdown(report))
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(_render_markdown(report))
    print(f"\nWrote {args.out}/report.json and report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
