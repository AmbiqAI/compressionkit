"""Sanity check: PRD vs PRDN-noise on clean synthetic ECG.

Generates clean synthetic ECG (``noise_multiplier=0.0``), runs it through
each golden codec, and compares raw PRD against PRDN-noise. On clean
inputs the bandpass-residual noise estimate should be ~0, so the two
metrics are expected to agree closely. If PRDN-noise is dramatically
lower than PRD on clean data, the metric would be over-discounting and
the customer claim would be unsafe.

Example:
    python scripts/sanity_clean_ecg.py \\
        --runs results/ecg_rvq_256hz_02x_golden \\
                results/ecg_rvq_256hz_04x_golden \\
                results/ecg_rvq_256hz_08x_golden \\
                results/ecg_rvq_256hz_16x_golden \\
                results/ecg_rvq_256hz_32x_golden \\
                results/ecg_rvq_256hz_64x_golden \\
        --num-segments 64 --sample-rate 256
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Some golden encoders use DepthwiseConv2D with stride (1, 2), which is not
# implemented on the cuDNN GPU kernel ("equal length strides" requirement).
# Force CPU so all CRs work without retraining or special handling.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.evaluation.metrics import compute_signal_metrics
from compressionkit.evaluation.noise import estimate_ecg_noise_floor
from compressionkit.preprocessing.ecg import generate_synthetic_ecg_batch


def _load_model(run_dir: Path):
    import keras

    from compressionkit.trainers.ecg_rvq import build_model

    keras.backend.clear_session()
    cfg = EcgRvqConfig.model_validate_json((run_dir / "config.json").read_text())
    model = build_model(cfg)
    dummy = np.zeros(
        (1, 1, cfg.data.frame_size, max(1, cfg.data.num_leads or 1)),
        dtype=np.float32,
    )
    model(dummy, training=False)
    for name in ("best_model.weights.h5", "model.weights.h5"):
        p = run_dir / name
        if p.exists():
            model.load_weights(p)
            return model, cfg
    raise FileNotFoundError(f"No weights in {run_dir}")


def _reconstruct(model, signals: np.ndarray) -> np.ndarray:
    """signals: (N, frame_size). Returns recon (N, frame_size)."""
    n, fs = signals.shape
    batch = signals.reshape(n, 1, fs, 1).astype(np.float32)
    recon = model.predict(batch, batch_size=32, verbose=0)
    return np.asarray(recon).reshape(n, fs)


def _agg(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def evaluate_run(
    run_dir: Path,
    *,
    sample_rate: int,
    num_segments: int,
    seed: int,
    noise_multipliers: list[float],
) -> dict:
    model, cfg = _load_model(run_dir)
    frame_size = cfg.data.frame_size
    out: dict = {"run": str(run_dir), "frame_size": frame_size, "buckets": {}}

    for nm in noise_multipliers:
        signals = generate_synthetic_ecg_batch(
            num_segments=num_segments,
            signal_length=frame_size,
            sample_rate=sample_rate,
            lead_index=1,
            heart_rate_bpm=[50.0, 100.0],
            noise_multiplier=[float(nm), float(nm)],
            impedance=[1.0, 1.0],
            seed=seed,
        )
        recons = _reconstruct(model, signals)

        prd_vals: list[float] = []
        prdn_vals: list[float] = []
        np_est_vals: list[float] = []
        bp_rms_vals: list[float] = []
        for orig, recon in zip(signals, recons):
            nf = estimate_ecg_noise_floor(orig, fs=sample_rate)
            np_est = float(nf.get("bp_noise_power", 0.0))
            m = compute_signal_metrics(orig, recon, noise_power=np_est)
            prd_vals.append(m["prd_percent"])
            prdn_vals.append(m.get("prdn_noise_percent", float("nan")))
            np_est_vals.append(np_est)
            bp_rms_vals.append(float(nf.get("bp_noise_rms", 0.0)))

        out["buckets"][f"noise_mult_{nm:g}"] = {
            "noise_multiplier": float(nm),
            "prd_percent": _agg(prd_vals),
            "prdn_noise_percent": _agg(
                [v for v in prdn_vals if not np.isnan(v)]
            ),
            "bp_noise_rms": _agg(bp_rms_vals),
            "bp_noise_power": _agg(np_est_vals),
        }
    return out


def _print_summary(report: dict, runs_order: list[Path]) -> None:
    print()
    print(
        f"{'run':<48} {'noise':>8} {'noise_rms':>10} "
        f"{'PRD%':>8} {'PRDN%':>8} {'gap':>8}"
    )
    print("-" * 100)
    for run_dir in runs_order:
        run_report = report["runs"][str(run_dir)]
        for _key, b in run_report["buckets"].items():
            prd = b["prd_percent"]["mean"]
            prdn = b["prdn_noise_percent"]["mean"]
            print(
                f"{run_dir.name:<48} {b['noise_multiplier']:>8.2f} "
                f"{b['bp_noise_rms']['mean']:>10.4f} "
                f"{prd:>8.2f} {prdn:>8.2f} {prd - prdn:>8.2f}"
            )
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--sample-rate", type=int, default=256)
    ap.add_argument("--num-segments", type=int, default=64)
    ap.add_argument(
        "--noise-multipliers",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="0.0 = clean synthetic; >0 = injected noise (physiokit synth scale).",
    )
    ap.add_argument("--seed", type=int, default=20260430)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/clean_ecg_sanity.json"),
    )
    args = ap.parse_args()

    report: dict = {
        "sample_rate": args.sample_rate,
        "num_segments": args.num_segments,
        "noise_multipliers": list(args.noise_multipliers),
        "runs": {},
    }
    for run_dir in args.runs:
        run_dir = run_dir.resolve()
        print(f"==> {run_dir.name}")
        report["runs"][str(run_dir)] = evaluate_run(
            run_dir,
            sample_rate=args.sample_rate,
            num_segments=args.num_segments,
            seed=args.seed,
            noise_multipliers=list(args.noise_multipliers),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote: {args.output}")
    _print_summary(report, [r.resolve() for r in args.runs])


if __name__ == "__main__":
    main()
