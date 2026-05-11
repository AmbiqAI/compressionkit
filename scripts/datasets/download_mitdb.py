"""Ingest the MIT-BIH Arrhythmia Database (mitdb) from PhysioNet.

MITDB: 48 half-hour 2-lead Holter excerpts at 360 Hz with manual beat-level
annotations (>110 k beats) — the classic gold standard for QRS-detection and
arrhythmia-classification benchmarks. Closes the beat-detection-failure
weakness identified on the golden-ECG runs.

This script:
  1. Downloads the raw WFDB files (.dat/.hea/.atr) from PhysioNet.
  2. Converts each record to canonical h5 (``data``, ``r_peaks``,
     ``beat_symbols``).
  3. Optionally re-publishes the bundle to ``s3://<bucket>/mitdb/mitdb.zip``.

Usage::

    python scripts/datasets/download_mitdb.py --limit 3   # smoke
    python scripts/datasets/download_mitdb.py             # full
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

from compressionkit.datasets._download import download_physionet_files  # noqa: E402

SLUG = "mitdb"
VERSION = "1.0.0"
ACQUISITION = "holter"

# Canonical record list for MITDB. Fixed by the publication so we can hard-code
# it; this avoids an extra round-trip to PhysioNet just to enumerate records.
RECORDS: tuple[str, ...] = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
)

EXTS = (".dat", ".hea", ".atr")


def _fetch_record(record: str, raw_dir: Path) -> Path:
    """Download all WFDB files for a record. Returns the .hea path."""
    rels = [f"{record}{ext}" for ext in EXTS]
    paths = download_physionet_files(SLUG, VERSION, rels, raw_dir)
    return next(p for p in paths if p.suffix == ".hea")


def _convert_record(raw_dir: Path, record: str, out_dir: Path, *, logger) -> Path:
    """Read a WFDB record and write its canonical h5 file."""
    rec = wfdb.rdrecord(str(raw_dir / record))
    ann = wfdb.rdann(str(raw_dir / record), "atr")

    # WFDB returns shape (samples, channels); transpose to (leads, samples).
    sig = np.asarray(rec.p_signal, dtype=np.float32).T
    fs = int(rec.fs)
    leads = list(rec.sig_name)

    out = out_dir / f"{record}.h5"
    with h5py.File(out, "w") as h:
        h.create_dataset("data", data=sig)
        # Beat annotations: r_peaks (sample idx) + beat_symbols (1-char codes).
        h.create_dataset("r_peaks", data=np.asarray(ann.sample, dtype=np.int64))
        h.create_dataset(
            "beat_symbols",
            data=np.asarray([s.encode("utf-8") for s in ann.symbol]),
        )
        h.attrs["fs"] = fs
        h.attrs["lead_names"] = ",".join(leads)
        h.attrs["source"] = SLUG
        h.attrs["acquisition"] = ACQUISITION
        h.attrs["patient_id"] = record
        h.attrs["units"] = ",".join(rec.units or [])
    logger.debug("Wrote %s data=%s fs=%d r_peaks=%d", out.name, sig.shape, fs, len(ann.sample))
    return out


def main() -> None:
    parser = make_parser(slug=SLUG, description=__doc__)
    args = parser.parse_args()
    logger = setup_logging(args)

    canonical, raw = resolve_dirs(args, SLUG)
    logger.info("Canonical dir: %s", canonical)
    logger.info("Raw WFDB dir:  %s", raw)

    records = list(RECORDS)
    if args.limit is not None:
        records = records[: args.limit]
        logger.info("Smoke test: limiting to %d records", len(records))

    if not args.skip_download:
        for rec in records:
            already = all((raw / f"{rec}{ext}").exists() for ext in EXTS)
            if already and not args.force:
                logger.debug("Skip download (cached): %s", rec)
                continue
            logger.info("Downloading WFDB record %s …", rec)
            _fetch_record(rec, raw)

    if not args.skip_convert:
        for rec in records:
            out = canonical / f"{rec}.h5"
            if out.exists() and not args.force:
                logger.debug("Skip convert (exists): %s", out.name)
                continue
            _convert_record(raw, rec, canonical, logger=logger)

    # Verification pass.
    n = len(list(canonical.glob("*.h5")))
    logger.info("Canonical h5 count: %d", n)
    if n:
        sample = sorted(canonical.glob("*.h5"))[0]
        with h5py.File(sample, "r") as h:
            data = h["data"]
            r = h["r_peaks"]
            logger.info(
                "  %s data=%s fs=%s leads=%s r_peaks=%d acquisition=%s",
                sample.name, tuple(data.shape), h.attrs["fs"],
                h.attrs["lead_names"], r.shape[0], h.attrs["acquisition"],
            )

    maybe_upload_s3(canonical, args.upload_s3, slug=SLUG)


if __name__ == "__main__":
    main()
