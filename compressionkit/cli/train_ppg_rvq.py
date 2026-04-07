"""CLI for PPG RVQ compression training.

Usage::

    python -m compressionkit.cli.train_ppg_rvq --config configs/ppg_rvq_08x_ds8_l2.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from compressionkit.configs.ppg_rvq import PpgRvqConfig
from compressionkit.trainers.ppg_rvq import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a PPG RVQ compression model from a YAML configuration.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args(argv)

    cfg = PpgRvqConfig.from_yaml(str(args.config))
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
