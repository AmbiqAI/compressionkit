"""Evaluate stitching quality on a trained ECG RVQ model.

Loads a model from a results directory and runs every configured
stitching strategy over a handful of validation recordings, printing a
per-method table and writing a ``stitching_report.json`` next to the
model. Intended as a quick post-hoc analysis tool that does not require
retraining.

Example::

    uv run python scripts/eval_ecg_stitching.py \\
        --run-dir results/ecg_rvq_256hz_32x_golden \\
        --duration-sec 30 \\
        --num-recordings 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import keras

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.evaluation.ecg_stitching import evaluate_stitching
from compressionkit.evaluation.stitching import STITCH_METHODS

DEFAULT_METHODS: list[str] = ["hard_concat", "overlap_add", "linear_crossfade", "tukey_overlap_add"]


def _load_model(run_dir: Path) -> tuple[keras.Model, EcgRvqConfig]:
    """Reload the full model by restoring best weights into a fresh build."""
    import numpy as np

    from compressionkit.trainers.ecg_rvq import build_model

    cfg_path = run_dir / "config.json"
    cfg = EcgRvqConfig.model_validate_json(cfg_path.read_text())
    model = build_model(cfg)
    # Trigger weight allocation by running one dummy frame through the model
    dummy = np.zeros((1, 1, cfg.data.frame_size, max(1, cfg.data.num_leads or 1)), dtype=np.float32)
    model(dummy, training=False)

    best = run_dir / "best_model.weights.h5"
    final = run_dir / "model.weights.h5"
    if best.exists():
        model.load_weights(best)
    elif final.exists():
        model.load_weights(final)
    else:
        raise FileNotFoundError(f"No model weights found in {run_dir}")
    return model, cfg


def _format_table(report: dict) -> str:
    header = (
        f"{'method':<20} {'n_rec':>6} {'PRD%':>8} {'cos':>8} {'seam_ratio':>12} {'seam_rms':>12} {'non_seam_rms':>14}"
    )
    lines = [header, "-" * len(header)]
    for name, stats in report["methods"].items():
        lines.append(
            f"{name:<20} {stats['num_recordings']:>6d} "
            f"{stats['mean_prd_percent']:>8.2f} "
            f"{stats['mean_cosine_similarity']:>8.4f} "
            f"{stats['mean_seam_ratio']:>12.3f} "
            f"{stats['mean_seam_rms']:>12.5f} "
            f"{stats['mean_non_seam_rms']:>14.5f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Trained run directory (contains config.json + weights)."
    )
    parser.add_argument(
        "--duration-sec", type=float, default=30.0, help="Per-recording reconstruction length in seconds."
    )
    parser.add_argument("--num-recordings", type=int, default=10)
    parser.add_argument("--hop-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lead-index", type=int, default=1)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=sorted(STITCH_METHODS),
        help="Stitching methods to evaluate.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Report JSON path (default: <run-dir>/stitching_report.json)."
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.exists():
        parser.error(f"Run directory not found: {run_dir}")

    print(f"Loading model from {run_dir}...", file=sys.stderr)
    model, cfg = _load_model(run_dir)
    data = cfg.data

    print(
        f"Evaluating stitching: methods={args.methods} "
        f"duration={args.duration_sec}s num_recordings={args.num_recordings}",
        file=sys.stderr,
    )
    report = evaluate_stitching(
        model,
        datasets_dir=Path(data.datasets_dir),
        dataset_glob=data.dataset_glob,
        frame_size=data.frame_size,
        sample_rate=data.effective_sample_rate,
        duration_sec=args.duration_sec,
        epsilon=data.epsilon,
        methods=args.methods,
        hop_ratio=args.hop_ratio,
        num_recordings=args.num_recordings,
        batch_size=args.batch_size,
        seed=data.shuffle_seed,
        lead_index=args.lead_index,
    )

    print(_format_table(report))

    out_path = args.output or (run_dir / "stitching_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
