"""Ingest the Icentia11k 2-week single-lead patch ECG dataset from Ambiq S3.

Icentia11k = 11,000 ambulatory patch records, single-lead, 250 Hz, ~2 weeks
each. Published as object prefix ``s3://ambiq-ai-datasets/icentia11k/p*****.h5``
where each h5 already follows the canonical layout. Total ~330 GB if pulled
in full, so this script supports ``--limit N`` to grab the first N patients
for early experimentation.

Usage::

    # First 50 patients (~1.5 GB)
    python scripts/datasets/download_icentia11k.py --limit 50

    # Full pull (do this on a beefy disk)
    python scripts/datasets/download_icentia11k.py
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

from compressionkit.datasets._download import download_ambiq_s3_prefix  # noqa: E402

SLUG = "icentia11k"


def verify(canonical_dir: Path, *, limit: int | None, logger) -> None:
    """Validate the per-segment layout used by the Icentia11k bundle.

    Icentia11k h5 files follow a nested layout (one group per ~1 hr segment),
    not the flat PTB-XL layout::

        pXXXXX/sNN/data   shape (~1.05M, 1) float16
        pXXXXX/sNN/blabels (B, 2) int32   – beat labels
        pXXXXX/sNN/rlabels (R, 2) int32   – rhythm labels
    """
    h5_files = sorted(canonical_dir.glob("*.h5"))
    if not h5_files:
        raise RuntimeError(f"No h5 files found under {canonical_dir}")
    logger.info("Found %d h5 files in %s", len(h5_files), canonical_dir)

    for fp in h5_files[: limit or 3]:
        with h5py.File(fp, "r") as h:
            patient_keys = list(h.keys())
            if not patient_keys:
                raise RuntimeError(f"Empty patient group in {fp.name}")
            pkey = patient_keys[0]
            seg_keys = list(h[pkey].keys())
            n_seg = len(seg_keys)
            sample = h[f"{pkey}/{seg_keys[0]}/data"]
            logger.info(
                "  %s patient=%s segments=%d sample=%s dtype=%s",
                fp.name, pkey, n_seg, tuple(sample.shape), sample.dtype,
            )
            sl = np.asarray(sample[: min(1024, sample.shape[0])])
            if not np.isfinite(sl).all():
                raise RuntimeError(f"Non-finite values in {fp.name}")


def main() -> None:
    parser = make_parser(slug=SLUG, description=__doc__)
    args = parser.parse_args()
    logger = setup_logging(args)

    canonical, _raw = resolve_dirs(args, SLUG)
    logger.info("Canonical dir: %s", canonical)

    if not args.skip_download:
        download_ambiq_s3_prefix(
            slug=SLUG,
            dst_dir=canonical,
            max_objects=args.limit,
        )

    verify(canonical, limit=args.limit, logger=logger)
    maybe_upload_s3(canonical, args.upload_s3, slug=SLUG)


if __name__ == "__main__":
    main()
