"""Ingest the St-Petersburg INCART 12-lead Arrhythmia Database from PhysioNet.

INCARTDB: 75 records, 12 leads, 257 Hz, 30 minutes each. Annotated with
beat-level labels (>175 k beats). Useful for stitching/long-record evaluation
in addition to MITDB's beat-detection ground truth.

Usage::

    python scripts/datasets/download_incartdb.py --limit 2  # smoke
    python scripts/datasets/download_incartdb.py            # full (~75 records, ~3 GB raw)
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import wfdb

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import make_parser, maybe_upload_s3, resolve_dirs, setup_logging  # noqa: E402

from compressionkit.datasets._download import download_physionet_files, http_download  # noqa: E402

SLUG = "incartdb"
VERSION = "1.0.0"
ACQUISITION = "clinic-12lead-30min"

# Filename suffixes for one record. INCART uses .dat/.hea + an .atr beat file.
EXTS = (".dat", ".hea", ".atr")


def _record_list(raw_dir: Path) -> list[str]:
    """Return INCART record names by fetching ``RECORDS`` once."""
    rel = "RECORDS"
    url = f"https://physionet.org/files/{SLUG}/{VERSION}/{rel}"
    local = raw_dir / rel
    if not local.exists():
        http_download(url, local, show_progress=False)
    return [ln.strip() for ln in local.read_text().splitlines() if ln.strip()]


def _fetch_record(record: str, raw_dir: Path) -> None:
    rels = [f"{record}{ext}" for ext in EXTS]
    download_physionet_files(SLUG, VERSION, rels, raw_dir)


def _convert_record(raw_dir: Path, record: str, out_dir: Path, *, logger) -> Path:
    rec = wfdb.rdrecord(str(raw_dir / record))
    ann = wfdb.rdann(str(raw_dir / record), "atr")
    sig = np.asarray(rec.p_signal, dtype=np.float32).T
    out = out_dir / f"{record}.h5"
    with h5py.File(out, "w") as h:
        h.create_dataset("data", data=sig)
        h.create_dataset("r_peaks", data=np.asarray(ann.sample, dtype=np.int64))
        h.create_dataset(
            "beat_symbols",
            data=np.asarray([s.encode("utf-8") for s in ann.symbol]),
        )
        h.attrs["fs"] = int(rec.fs)
        h.attrs["lead_names"] = ",".join(rec.sig_name)
        h.attrs["source"] = SLUG
        h.attrs["acquisition"] = ACQUISITION
        h.attrs["patient_id"] = record
        h.attrs["units"] = ",".join(rec.units or [])
    logger.debug("Wrote %s data=%s fs=%d r_peaks=%d", out.name, sig.shape, int(rec.fs), len(ann.sample))
    return out


def main() -> None:
    parser = make_parser(slug=SLUG, description=__doc__)
    args = parser.parse_args()
    logger = setup_logging(args)

    canonical, raw = resolve_dirs(args, SLUG)
    logger.info("Canonical dir: %s", canonical)
    logger.info("Raw WFDB dir:  %s", raw)

    records = _record_list(raw)
    if args.limit is not None:
        records = records[: args.limit]
        logger.info("Smoke test: limiting to %d records", len(records))

    if not args.skip_download:
        for rec in records:
            already = all((raw / f"{rec}{ext}").exists() for ext in EXTS)
            if already and not args.force:
                continue
            logger.info("Downloading %s …", rec)
            _fetch_record(rec, raw)

    if not args.skip_convert:
        for rec in records:
            out = canonical / f"{rec}.h5"
            if out.exists() and not args.force:
                continue
            _convert_record(raw, rec, canonical, logger=logger)

    n = len(list(canonical.glob("*.h5")))
    logger.info("Canonical h5 count: %d", n)
    if n:
        sample = sorted(canonical.glob("*.h5"))[0]
        with h5py.File(sample, "r") as h:
            data = h["data"]
            r = h["r_peaks"]
            logger.info(
                "  %s data=%s fs=%s leads=%s r_peaks=%d",
                sample.name, tuple(data.shape), h.attrs["fs"],
                h.attrs["lead_names"], r.shape[0],
            )

    maybe_upload_s3(canonical, args.upload_s3, slug=SLUG)


if __name__ == "__main__":
    main()
