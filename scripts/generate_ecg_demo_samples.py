"""Generate quality-gated, real MIT-BIH ECG CSV files for the web demo.

Examples:
    uv run python scripts/generate_ecg_demo_samples.py
    uv run python scripts/generate_ecg_demo_samples.py --duration-seconds 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

from compressionkit.configs.paths import default_datasets_dir
from compressionkit.export.demo_ecg import export_ecg_demo_csvs


def main() -> None:
    """Parse arguments and write ECG demo CSV artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/demo_ecg_real_mitdb_256hz_30s"),
        help="Destination directory for CSV files and manifest.json.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(default_datasets_dir()) / "mitdb",
        help="Canonical MIT-BIH H5 directory.",
    )
    parser.add_argument("--num-clips", type=int, default=10, help="Number of clips to generate (1-10).")
    parser.add_argument("--duration-seconds", type=float, default=30.0, help="Length of each clip (minimum 10).")
    parser.add_argument("--sample-rate", type=int, default=256, help="Output sample rate in Hz.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    args = parser.parse_args()

    exported = export_ecg_demo_csvs(
        args.output_dir,
        dataset_dir=args.dataset_dir,
        num_clips=args.num_clips,
        duration_seconds=args.duration_seconds,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )
    print(f"Wrote {len(exported.csv_paths)} ECG CSV files to {args.output_dir}")
    print(f"Quality manifest: {exported.manifest_path}")


if __name__ == "__main__":
    main()
