"""Ingest the QT Database (qtdb) bundle from Ambiq S3.

QTDB: 105 fifteen-minute 2-lead excerpts from MIT-BIH and Holter recordings,
with manual beat-by-beat fiducial annotations (P onset/peak/end, QRS,
T peak/end). The reference set for QT-interval and waveform-fiducial
evaluation. Already published as a single zip at
``s3://ambiq-ai-datasets/qtdb/qtdb.zip``.

NOTE: Unlike LUDB/LSAD, the Ambiq ``qtdb.zip`` ships *raw WFDB* files
(``selXXX.dat/.hea/.atr``), not pre-converted h5. If you need fresh canonical
h5 files you'll need to convert them with ``wfdb.rdrecord`` similar to MITDB.
At the time of writing the workspace already contained 1,212 ``*.h5`` files
under ``datasets/qtdb`` from a prior heartkit run, so this script is a
placeholder and skips re-download by default.

Usage::

    python scripts/datasets/download_qtdb.py --force   # re-pull raw zip into a separate dir
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import make_parser, maybe_upload_s3, resolve_dirs, setup_logging  # noqa: E402

from compressionkit.datasets._download import download_ambiq_s3_zip  # noqa: E402

SLUG = "qtdb"


def verify(canonical_dir: Path, *, limit: int | None, logger) -> None:
    h5_files = sorted(canonical_dir.glob("*.h5"))
    if not h5_files:
        raise RuntimeError(f"No h5 files found under {canonical_dir}")
    logger.info("Found %d h5 files in %s", len(h5_files), canonical_dir)
    for fp in h5_files[: limit or 3]:
        with h5py.File(fp, "r") as h:
            keys = list(h.keys())
            data = h.get("data")
            shape = tuple(data.shape) if data is not None else None
            logger.info("  %s data=%s keys=%s", fp.name, shape, keys)
            if data is not None:
                arr = np.asarray(data[:])
                if not np.isfinite(arr).all():
                    raise RuntimeError(f"Non-finite values in {fp.name}")


def main() -> None:
    parser = make_parser(slug=SLUG, description=__doc__)
    args = parser.parse_args()
    logger = setup_logging(args)

    canonical, raw = resolve_dirs(args, SLUG)
    logger.info("Canonical dir: %s", canonical)

    existing = list(canonical.glob("*.h5"))
    if existing and not args.force:
        logger.info(
            "Found %d existing h5 files under %s — skipping re-download. "
            "Pass --force to re-pull the raw bundle (it contains WFDB files, "
            "not h5; you'll need a converter).",
            len(existing), canonical,
        )
    elif not args.skip_download:
        # Pulls raw WFDB files into <raw> rather than the canonical dir.
        download_ambiq_s3_zip(
            slug=SLUG,
            dst_dir=raw,
            extract=True,
            delete_zip=False,
            force=args.force,
        )
        logger.warning(
            "Raw WFDB files extracted to %s — you'll need a wfdb→h5 converter "
            "(see download_mitdb.py for the pattern).", raw,
        )

    verify(canonical, limit=args.limit, logger=logger)
    maybe_upload_s3(canonical, args.upload_s3, slug=SLUG)


if __name__ == "__main__":
    main()
