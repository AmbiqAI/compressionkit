#!/usr/bin/env python
"""Run PPG RVQ wavelet experiments."""

from __future__ import annotations

import sys
from pathlib import Path

from compressionkit.configs.ppg_rvq import PpgRvqConfig
from compressionkit.recipes._registry import get_recipe


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config.yaml> [config2.yaml ...]")
        sys.exit(1)

    spec = get_recipe("train-ppg-rvq")
    for config_path in sys.argv[1:]:
        print(f"\n{'=' * 60}")
        print(f"Running: {config_path}")
        print(f"{'=' * 60}\n")
        cfg = PpgRvqConfig.from_yaml(config_path)
        result = spec.train_fn(cfg)
        prd = result.get("primary_metrics", {}).get("prd_percent", "N/A")
        print(f"\n>>> {Path(config_path).stem}: PRD={prd}")


if __name__ == "__main__":
    main()
