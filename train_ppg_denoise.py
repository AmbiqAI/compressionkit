#!/usr/bin/env python3
"""Train denoising PPG RVQ codec using Tier-1 augmentations.

This script trains a codec that simultaneously compresses and denoises PPG.
The key mechanism: augmented (corrupted) input → encoder → RVQ → decoder → clean target.

Usage:
    python train_ppg_denoise.py configs/ppg_rvq_64hz_04x_denoise.yaml
    python train_ppg_denoise.py configs/ppg_rvq_64hz_04x_denoise.yaml --cross-domain-eval
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train denoising PPG RVQ codec")
    parser.add_argument("config", type=Path, help="YAML config path")
    parser.add_argument(
        "--cross-domain-eval",
        action="store_true",
        help="Run cross-domain eval after training",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Import after arg parse to speed up --help
    import yaml

    from compressionkit.configs.ppg_rvq import PpgRvqConfig
    from compressionkit.recipes.train_ppg_rvq import train as run_training

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    cfg = PpgRvqConfig.model_validate(raw)
    logger = logging.getLogger("denoise-train")

    logger.info("Training denoising codec: %s", cfg.run_name)
    logger.info("  Augmentation enabled: %s", cfg.data.augmentation.enabled)
    if cfg.data.augmentation.enabled:
        logger.info("  Noise sources: %s", cfg.data.augmentation.noise_bank_sources)
        logger.info("  Baseline wander prob: %.2f", cfg.data.augmentation.baseline_wander_prob)
        logger.info("  Motion artifact prob: %.2f", cfg.data.augmentation.motion_artifact_prob)

    # Run training
    run_training(cfg)

    # Optional cross-domain evaluation
    if args.cross_domain_eval:
        run_dir = Path(cfg.output.results_root) / cfg.run_name
        logger.info("Running cross-domain evaluation...")
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "scripts/eval_cross_domain.py",
                "--run-dir",
                str(run_dir),
                "--sources",
                "bidmc",
                "ppg_dalia",
                "wesad",
                "--max-windows",
                "2000",
                "--num-long-recordings",
                "10",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
