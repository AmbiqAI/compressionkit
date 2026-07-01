"""Re-export a golden run directory as a complete deployment package.

Usage:
    python scripts/package_golden_release.py \\
        --golden-dir results/ppg_rvq_64hz_04x_golden \\
        --modality ppg \\
        --sample-rate 64 \\
        --compression-ratio 4

Produces a self-contained `deploy/` subdirectory with all artifacts needed
for HuggingFace release, including float32 decoder TFLite, model card, and
synthetic stimulus data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a golden run for release.")
    parser.add_argument("--golden-dir", type=Path, required=True, help="Path to golden run directory.")
    parser.add_argument("--modality", choices=["ecg", "ppg"], required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--compression-ratio", type=int, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override output dir (default: golden-dir/deploy)."
    )
    parser.add_argument("--num-stimulus", type=int, default=10, help="Number of synthetic stimulus samples.")
    parser.add_argument(
        "--export-decoder-int8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export INT8 decoder TFLite alongside the float32 decoder.",
    )
    parser.add_argument(
        "--scorecard",
        type=Path,
        default=None,
        help="Path to quality_scorecard.json. Defaults to <golden-dir>/quality_scorecard.json when present.",
    )
    args = parser.parse_args()

    golden_dir = args.golden_dir
    if not golden_dir.is_dir():
        logger.error("Golden directory not found: %s", golden_dir)
        sys.exit(1)

    output_dir = args.output_dir or golden_dir / "deploy"
    from compressionkit.experiments.registry import GoldenExperiment
    from compressionkit.experiments.repackage import repackage_rvq_golden

    experiment = GoldenExperiment(
        experiment_id=f"{args.modality}-rvq-{args.compression_ratio}x",
        modality=args.modality,
        family="codec",
        method="rvq",
        recipe=None,
        config_path=None,
        run_name=golden_dir.name,
        sample_rate=args.sample_rate,
        compression_ratio=args.compression_ratio,
        hf_repo_id=f"Ambiq/compressionkit-{args.modality}-{args.compression_ratio}x",
        dataset_id="manual",
    )
    summary = repackage_rvq_golden(
        experiment,
        run_dir=golden_dir,
        output_dir=output_dir,
        num_stimulus=args.num_stimulus,
        export_decoder_int8=args.export_decoder_int8,
        scorecard_path=args.scorecard,
    )

    logger.info("Package complete: %s", output_dir)
    logger.info("Artifacts: %s", json.dumps(summary["artifacts"], indent=2))


if __name__ == "__main__":
    main()
