"""Smoke-test ingest: pull the LUDB bundle from Ambiq S3 and verify h5 layout.

LUDB = Lobachevsky University Database. 200 subjects × 10 s × 12-lead @ 500 Hz
with manual P/QRS/T wave delineation. Already published as a single zip on the
Ambiq AI public S3 bucket via heartkit, so this script is the fastest way to
validate that the download/extract/verify pipeline works end-to-end.

Usage::

    # Smoke test (downloads, extracts, verifies first 5 patients)
    python scripts/datasets/download_ludb.py --root datasets_test --limit 5

    # Full pull
    python scripts/datasets/download_ludb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import make_parser, maybe_upload_s3, resolve_dirs, setup_logging

from compressionkit.datasets._download import download_ambiq_s3_zip

SLUG = "ludb"


def verify(canonical_dir: Path, *, limit: int | None, logger) -> None:
    """Sanity-check a few h5 files and emit one-line stats."""
    h5_files = sorted(canonical_dir.glob("*.h5"))
    if not h5_files:
        raise RuntimeError(f"No h5 files found under {canonical_dir}")
    logger.info("Found %d h5 files in %s", len(h5_files), canonical_dir)
    for fp in h5_files[: limit or 3]:
        with h5py.File(fp, "r") as h:
            data = h["data"]
            seg = h.get("segmentations", None)
            shape = tuple(data.shape)
            seg_shape = tuple(seg.shape) if seg is not None else None
            logger.info(
                "  %s data=%s dtype=%s seg=%s",
                fp.name, shape, data.dtype, seg_shape,
            )
            arr = np.asarray(data[:])
            if not np.isfinite(arr).all():
                raise RuntimeError(f"Non-finite values in {fp.name}")


def main() -> None:
    parser = make_parser(slug=SLUG, description=__doc__)
    args = parser.parse_args()
    logger = setup_logging(args)

    canonical, _raw = resolve_dirs(args, SLUG)
    logger.info("Canonical dir: %s", canonical)

    if not args.skip_download:
        download_ambiq_s3_zip(
            slug=SLUG,
            dst_dir=canonical,
            extract=True,
            delete_zip=False,
            force=args.force,
        )

    verify(canonical, limit=args.limit, logger=logger)
    maybe_upload_s3(canonical, args.upload_s3, slug=SLUG)


if __name__ == "__main__":
    main()
