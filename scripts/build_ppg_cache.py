#!/usr/bin/env python3
"""Build unified PPG TFRecord caches for one or more dataset sources.

Each source is cached independently under ``<cache_root>/<slug>/`` with
``train.tfrecord``, ``val.tfrecord``, and ``metadata.json``.  Re-running
skips sources whose cache already exists (use ``--force`` to rebuild).

Usage::

    # Build all known sources
    python scripts/build_ppg_cache.py --sources mesa bidmc ppg_dalia wesad butppg

    # Build just MESA with a window cap
    python scripts/build_ppg_cache.py --sources mesa --max-windows-per-file 512

    # Custom paths
    python scripts/build_ppg_cache.py --sources mesa \\
        --datasets-root /data/datasets \\
        --cache-root /data/ppg_cache
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build-ppg-cache")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unified PPG TFRecord caches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source slugs to build (e.g. mesa bidmc ppg_dalia wesad butppg)",
    )
    parser.add_argument(
        "--datasets-root",
        default="/home/vscode/datasets",
        help="Root directory containing raw dataset folders",
    )
    parser.add_argument(
        "--cache-root",
        default="datasets/ppg_cache",
        help="Output directory for TFRecord caches",
    )
    parser.add_argument("--target-fs", type=int, default=64, help="Target sample rate (Hz)")
    parser.add_argument("--frame-size", type=int, default=320, help="Window size in samples")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=None,
        help="Cap windows per source file (useful for large datasets like MESA)",
    )
    parser.add_argument("--force", action="store_true", help="Force rebuild existing caches")
    parser.add_argument("--no-sanitize", action="store_true", help="Disable window sanitization")

    args = parser.parse_args()

    from compressionkit.datasets.ppg_cache import (
        KNOWN_SOURCES,
        CacheBuildConfig,
        build_source_cache,
    )

    # Validate source slugs
    unknown = set(args.sources) - set(KNOWN_SOURCES)
    if unknown:
        logger.error("Unknown sources: %s. Known: %s", unknown, sorted(KNOWN_SOURCES))
        sys.exit(1)

    total_start = time.time()
    results: dict[str, dict] = {}

    for slug in args.sources:
        logger.info("=" * 60)
        logger.info("Building cache: %s", slug)
        logger.info("=" * 60)

        cfg = CacheBuildConfig(
            slug=slug,
            datasets_root=args.datasets_root,
            cache_root=args.cache_root,
            target_fs=args.target_fs,
            frame_size=args.frame_size,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            split_seed=args.split_seed,
            max_windows_per_file=args.max_windows_per_file,
            force_rebuild=args.force,
            sanitize=not args.no_sanitize,
        )

        t0 = time.time()
        cache_dir, metadata = build_source_cache(cfg)
        elapsed = time.time() - t0

        results[slug] = {
            "cache_dir": str(cache_dir),
            "train": metadata["train_examples"],
            "val": metadata["val_examples"],
            "time_s": round(elapsed, 1),
        }
        logger.info(
            "  %s done: %d train, %d val windows in %.1fs",
            slug, metadata["train_examples"], metadata["val_examples"], elapsed,
        )

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info("Summary (%.1fs total):", total_elapsed)
    logger.info("%-15s %10s %10s %8s", "Source", "Train", "Val", "Time(s)")
    logger.info("-" * 45)
    for slug, r in results.items():
        logger.info("%-15s %10d %10d %8.1f", slug, r["train"], r["val"], r["time_s"])


if __name__ == "__main__":
    main()
