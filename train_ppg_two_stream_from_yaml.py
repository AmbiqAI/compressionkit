"""Train a two-stream PPG codec from a YAML config.

Usage::

    uv run python train_ppg_two_stream_from_yaml.py configs/ppg_two_stream_08x.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ppg-two-stream")


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: train_ppg_two_stream_from_yaml.py <config.yaml>", file=sys.stderr)
        return 1

    config_path = Path(argv[0])
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    from compressionkit.configs.ppg_two_stream import PpgTwoStreamConfig
    from compressionkit.trainers.ppg_two_stream import train_two_stream

    cfg = PpgTwoStreamConfig.from_yaml(str(config_path))
    logger.info("Loaded config: %s (run_name=%s)", config_path, cfg.run_name)

    report = train_two_stream(cfg)
    cr = report["compression_stats"]["compression_ratio"]
    prd = report["primary_metrics"]["prd_percent"]
    cos = report["primary_metrics"]["cosine_similarity"]
    logger.info("Done. CR=%.2fx PRD=%.2f%% cos=%.4f", cr, prd, cos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
