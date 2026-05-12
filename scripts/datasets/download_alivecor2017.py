"""Ingest the PhysioNet/CinC Challenge 2017 single-lead AliveCor dataset.

PhysioNet ``challenge-2017``: 8,528 short single-lead ECG recordings (9-61 s,
typically 30 s) acquired with the AliveCor handheld device at 300 Hz, with
rhythm-class labels (``N``: normal, ``A``: atrial fibrillation, ``O``: other,
``~``: noisy). This is the only sizeable public *smartphone single-lead* set
and is essential for the wearable-acquisition bucket.

Distribution: a single ``training2017.zip`` (94.6 MB) plus ``REFERENCE-v3.csv``
(latest expert labels) — much faster than per-record fetching.

Usage::

    python scripts/datasets/download_alivecor2017.py --limit 10  # smoke
    python scripts/datasets/download_alivecor2017.py             # full
"""

from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import wfdb

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import make_parser, maybe_upload_s3, resolve_dirs, setup_logging

from compressionkit.datasets._download import http_download

SLUG = "alivecor2017"
ACQUISITION = "smartphone-1lead"
BASE = "https://physionet.org/files/challenge-2017/1.0.0"
ZIP_NAME = "training2017.zip"
REFERENCE_CSV = "REFERENCE-v3.csv"
LABEL_NAMES = {"N": "normal", "A": "afib", "O": "other", "~": "noisy"}


def _ensure_raw(raw_dir: Path, *, force: bool, logger) -> tuple[Path, Path]:
    """Download ``training2017.zip`` and ``REFERENCE-v3.csv``; extract zip."""
    zip_path = raw_dir / ZIP_NAME
    ref_path = raw_dir / REFERENCE_CSV

    if force or not zip_path.exists():
        http_download(f"{BASE}/{ZIP_NAME}", zip_path)
    if force or not ref_path.exists():
        http_download(f"{BASE}/{REFERENCE_CSV}", ref_path)

    extracted_dir = raw_dir / "training2017"
    if force or not extracted_dir.exists():
        logger.info("Extracting %s …", ZIP_NAME)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
    return extracted_dir, ref_path


def _read_labels(ref_path: Path) -> dict[str, str]:
    """Return ``{record_id: label_char}``; label_char ∈ N/A/O/~."""
    out: dict[str, str] = {}
    with ref_path.open() as f:
        for row in csv.reader(f):
            if not row or not row[0].startswith("A"):
                continue
            out[row[0]] = row[1].strip()
    return out


def _convert_record(rec_dir: Path, record: str, label: str | None,
                    out_dir: Path, *, logger) -> Path:
    rec = wfdb.rdrecord(str(rec_dir / record))
    sig = np.asarray(rec.p_signal, dtype=np.float32).T  # (1, samples)
    out = out_dir / f"{record}.h5"
    with h5py.File(out, "w") as h:
        h.create_dataset("data", data=sig)
        h.attrs["fs"] = int(rec.fs)
        h.attrs["lead_names"] = ",".join(rec.sig_name) or "I"
        h.attrs["source"] = SLUG
        h.attrs["acquisition"] = ACQUISITION
        h.attrs["patient_id"] = record
        h.attrs["units"] = ",".join(rec.units or [])
        if label is not None:
            h.attrs["rhythm_label"] = label
            h.attrs["rhythm_class"] = LABEL_NAMES.get(label, "unknown")
    logger.debug("Wrote %s data=%s label=%s", out.name, sig.shape, label)
    return out


def main() -> None:
    parser = make_parser(slug=SLUG, description=__doc__)
    args = parser.parse_args()
    logger = setup_logging(args)

    canonical, raw = resolve_dirs(args, SLUG)
    logger.info("Canonical dir: %s", canonical)
    logger.info("Raw dir:       %s", raw)

    if not args.skip_download:
        rec_dir, ref_path = _ensure_raw(raw, force=args.force, logger=logger)
    else:
        rec_dir = raw / "training2017"
        ref_path = raw / REFERENCE_CSV

    labels = _read_labels(ref_path)
    records = sorted(p.stem for p in rec_dir.glob("A*.hea"))
    if args.limit is not None:
        records = records[: args.limit]
        logger.info("Smoke test: limiting to %d records", len(records))
    logger.info("Found %d records, %d labelled", len(records), len(labels))

    if not args.skip_convert:
        for rec in records:
            out = canonical / f"{rec}.h5"
            if out.exists() and not args.force:
                continue
            _convert_record(rec_dir, rec, labels.get(rec), canonical, logger=logger)

    n = len(list(canonical.glob("*.h5")))
    logger.info("Canonical h5 count: %d", n)
    if n:
        sample = sorted(canonical.glob("*.h5"))[0]
        with h5py.File(sample, "r") as h:
            data = h["data"]
            logger.info(
                "  %s data=%s fs=%s lead=%s rhythm=%s acquisition=%s",
                sample.name, tuple(data.shape), h.attrs["fs"],
                h.attrs["lead_names"],
                h.attrs.get("rhythm_class"), h.attrs["acquisition"],
            )

    maybe_upload_s3(canonical, args.upload_s3, slug=SLUG)


if __name__ == "__main__":
    main()
