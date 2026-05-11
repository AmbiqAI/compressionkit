"""Re-run post-training evaluation on an existing trained ECG run.

Loads ``config.json`` + ``best_model.weights.h5`` from a run directory,
overrides ``evaluation.num_samples`` / ``num_plot_samples`` (and disables
stitching by default), and re-emits the per-sample CSV/PNG artifacts so that
``build_quality_scorecard.py`` can compute statistics on a much larger sample
pool (e.g. 1000 instead of the original 50).

Existing ``sample_<NNN>.csv`` and ``plots/sample_<NNN>.png`` files for indices
< new ``num_samples`` are overwritten; higher indices left over from a previous
larger run are removed so the scorecard sees a clean window.

Example:
    python scripts/refresh_eval_ecg.py results/ecg_rvq_256hz_32x_golden \
        --num-samples 1000 --num-plot-samples 50 --rebuild-scorecard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

# Some encoder configs use DepthwiseConv2D with asymmetric strides (1, 2) which
# cuDNN refuses outside XLA (model.fit enables XLA, model.predict does not). The
# safe portable path is to force CPU for the refresh, which is fast enough for
# the few hundred to few thousand samples this script targets. Set
# REFRESH_EVAL_USE_GPU=1 to opt back into the GPU.
if not os.environ.get("REFRESH_EVAL_USE_GPU"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

logger = logging.getLogger("refresh-eval-ecg")


def _purge_old_sample_artifacts(run_dir: Path) -> None:
    """Delete sample_*.csv and plots/sample_*.png so we can write a fresh set."""
    for csv in run_dir.glob("sample_*.csv"):
        csv.unlink()
    plots_dir = run_dir / "plots"
    if plots_dir.is_dir():
        for png in plots_dir.glob("sample_*.png"):
            png.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--num-plot-samples", type=int, default=50)
    ap.add_argument(
        "--keep-stitching",
        action="store_true",
        help="By default stitching eval is skipped (the existing report stays). "
             "Pass to re-run stitching too.",
    )
    ap.add_argument(
        "--rebuild-scorecard",
        action="store_true",
        help="Also re-run build_quality_scorecard with the standard ECG settings.",
    )
    ap.add_argument("--sample-rate", type=int, default=None,
                    help="Override scorecard sample-rate (default: data.target_sample_rate from config).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_dir: Path = args.run_dir.resolve()
    cfg_path = run_dir / "config.json"
    ckpt_path = run_dir / "best_model.weights.h5"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    # Imports are deferred so --help works without TF installed.
    from compressionkit.configs.ecg_rvq import EcgRvqConfig
    from compressionkit.preprocessing.ecg import build_augmenter, build_preprocessor
    from compressionkit.trainers.ecg_rvq import (
        build_datasets,
        build_model,
        compile_model,
        run_evaluation,
    )

    cfg_dump = json.loads(cfg_path.read_text())
    # Override eval block before validation so the cfg has the new defaults.
    eval_block = cfg_dump.setdefault("evaluation", {})
    eval_block["num_samples"] = int(args.num_samples)
    eval_block["num_plot_samples"] = int(args.num_plot_samples)
    if not args.keep_stitching:
        eval_block.setdefault("stitching", {})["enabled"] = False

    cfg = EcgRvqConfig.model_validate(cfg_dump)
    sample_rate = args.sample_rate or cfg.data.effective_sample_rate

    logger.info("Run dir          : %s", run_dir)
    logger.info("num_samples      : %d", cfg.evaluation.num_samples)
    logger.info("num_plot_samples : %d", cfg.evaluation.num_plot_samples)
    logger.info("stitching enabled: %s", cfg.evaluation.stitching.enabled)

    # 1. Datasets (training augmenter is fine; eval uses val_ds which is shuffle=False).
    data = cfg.data
    preprocessor = build_preprocessor(frame_size=data.frame_size, epsilon=data.epsilon)
    augmenter = build_augmenter(aug_cfg=data.augmentation, sample_rate=data.effective_sample_rate)
    _, val_ds, validation_steps, ds_info = build_datasets(cfg, preprocessor, augmenter)
    logger.info("Dataset mode     : %s", ds_info["mode"])

    # 2. Build + load best weights.
    model = build_model(cfg)
    compile_model(model, cfg, learning_rate=1e-3)  # value irrelevant for eval-only.
    # Build model variables by calling once on a tiny batch.
    for x_batch, _y in val_ds.take(1):
        _ = model(x_batch, training=False)
        break
    model.load_weights(ckpt_path)
    logger.info("Loaded weights   : %s", ckpt_path.name)

    # 3. Purge old sample artifacts so the scorecard sees a clean window.
    _purge_old_sample_artifacts(run_dir)

    # 4. Run evaluation (writes new sample_<NNN>.csv files + first N plots).
    run_evaluation(
        cfg, model=model, val_ds=val_ds, run_dir=run_dir, validation_steps=validation_steps,
    )
    n_csv = len(list(run_dir.glob("sample_*.csv")))
    n_png = len(list((run_dir / "plots").glob("sample_*.png"))) if (run_dir / "plots").is_dir() else 0
    logger.info("Wrote %d CSVs, %d PNGs to %s", n_csv, n_png, run_dir)

    # 5. Optional: rebuild the scorecard.
    if args.rebuild_scorecard:
        from compressionkit.evaluation.scorecard import write_quality_scorecard

        out = write_quality_scorecard(
            run_dir, modality="ecg", sample_rate=sample_rate, noise_estimator="bp",
        )
        logger.info("Scorecard        : %s", out)


if __name__ == "__main__":
    main()
