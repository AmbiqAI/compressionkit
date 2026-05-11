"""Re-run post-training evaluation on an existing trained golden PPG run.

This mirrors ``refresh_eval_ecg.py`` for PPG: it loads ``config.json`` and
``best_model.weights.h5`` from a run directory, overrides
``evaluation.num_samples`` / ``num_plot_samples``, and rewrites the
``sample_<NNN>.csv`` metrics pool while keeping visual plots capped.

Example:
    python scripts/refresh_eval_ppg.py results/ppg_rvq_64hz_08x_golden \
        --num-samples 1000 --num-plot-samples 25 --rebuild-scorecard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

if not os.environ.get("REFRESH_EVAL_USE_GPU"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

logger = logging.getLogger("refresh-eval-ppg")


def _purge_old_sample_artifacts(run_dir: Path) -> None:
    """Delete sample CSVs and plots so the refreshed scorecard is clean."""
    for csv_path in run_dir.glob("sample_*.csv"):
        csv_path.unlink()
    plots_dir = run_dir / "plots"
    if plots_dir.is_dir():
        for png_path in plots_dir.glob("sample_*.png"):
            png_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-plot-samples", type=int, default=25)
    parser.add_argument(
        "--keep-long-recording",
        action="store_true",
        help="By default long-recording eval is skipped and the existing report stays.",
    )
    parser.add_argument(
        "--rebuild-scorecard",
        action="store_true",
        help="Also rebuild the PPG quality scorecard after refreshing samples.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Override scorecard sample rate. Defaults to data.sampling_rate from config.",
    )
    parser.add_argument(
        "--min-signal-std",
        type=float,
        default=1.0e-4,
        help="Reject near-flat/corrupted windows below this std when rebuilding scorecards.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_dir = args.run_dir.resolve()
    cfg_path = run_dir / "config.json"
    ckpt_path = run_dir / "best_model.weights.h5"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    from compressionkit.configs.ppg_rvq import PpgRvqConfig
    from compressionkit.preprocessing.ppg import build_augmenter, build_preprocessor
    from compressionkit.trainers.ppg_rvq import (
        build_datasets,
        build_model,
        compile_model,
        run_evaluation,
    )

    cfg_dump = json.loads(cfg_path.read_text())
    eval_block = cfg_dump.setdefault("evaluation", {})
    eval_block["num_samples"] = int(args.num_samples)
    eval_block["num_plot_samples"] = int(args.num_plot_samples)
    if not args.keep_long_recording:
        eval_block.setdefault("long_recording", {})["enabled"] = False

    cfg = PpgRvqConfig.model_validate(cfg_dump)
    sample_rate = args.sample_rate or cfg.data.sampling_rate

    logger.info("Run dir              : %s", run_dir)
    logger.info("num_samples          : %d", cfg.evaluation.num_samples)
    logger.info("num_plot_samples     : %d", cfg.evaluation.num_plot_samples)
    logger.info("long-recording eval  : %s", cfg.evaluation.long_recording.enabled)

    data = cfg.data
    preprocessor = build_preprocessor(frame_size=data.frame_size, epsilon=data.epsilon)
    augmenter = build_augmenter(tuple(data.gaussian_noise))
    _, val_ds, validation_steps, ds_info = build_datasets(cfg, preprocessor, augmenter)
    logger.info("Dataset mode         : %s", ds_info["mode"])

    model = build_model(cfg)
    compile_model(model, cfg, learning_rate=1e-3)
    for x_batch, _y in val_ds.take(1):
        model(x_batch, training=False)
        break
    model.load_weights(ckpt_path)
    logger.info("Loaded weights       : %s", ckpt_path.name)

    _purge_old_sample_artifacts(run_dir)
    run_evaluation(
        cfg,
        model=model,
        val_ds=val_ds,
        run_dir=run_dir,
        validation_steps=validation_steps,
    )
    n_csv = len(list(run_dir.glob("sample_*.csv")))
    n_png = len(list((run_dir / "plots").glob("sample_*.png"))) if (run_dir / "plots").is_dir() else 0
    logger.info("Wrote %d CSVs, %d PNGs to %s", n_csv, n_png, run_dir)

    if args.rebuild_scorecard:
        from compressionkit.evaluation.scorecard import write_quality_scorecard

        out = write_quality_scorecard(
            run_dir,
            modality="ppg",
            sample_rate=sample_rate,
            noise_estimator="bp",
            min_signal_std=args.min_signal_std,
        )
        logger.info("Scorecard            : %s", out)


if __name__ == "__main__":
    main()
