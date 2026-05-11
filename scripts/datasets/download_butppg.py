"""Ingest the Brno University of Technology Smartphone PPG Database (butppg).

BUT PPG v2.0.0 (PhysioNet slug ``butppg``): 3,888 ten-second PPG recordings from
50 subjects, captured by Xiaomi/Huawei smartphone cameras at 30 Hz, with
synchronized reference 1-lead ECG (Bittium Faros) at 1000 Hz. Most signals
include 100 Hz tri-axial accelerometer data and per-record annotations of HR
(reference), signal quality (binary), BP/SpO2/glycaemia singles, and QRS-peak
positions. Total raw size: ~203 MB.

This is the only public smartphone-camera PPG benchmark with paired ECG and is
the canonical resource for PPG quality-assessment + HR-estimation models.

Usage::

    python scripts/datasets/download_butppg.py --limit 5   # smoke
    python scripts/datasets/download_butppg.py             # full
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import wfdb

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import make_parser, maybe_upload_s3, resolve_dirs, setup_logging  # noqa: E402

from compressionkit.datasets._download import download_physionet_files, http_download, physionet_url  # noqa: E402

SLUG = "butppg"
VERSION = "2.0.0"
ACQUISITION = "smartphone-1lead-ppg"

# Ancillary metadata files at the database root.
ROOT_FILES = ("RECORDS.txt", "quality-hr-ann.csv", "subject-info.csv")

# Expected channel counts so we can disambiguate orientation if wfdb returns a
# malformed shape.
_EXPECTED_CHANS = (1, 1, 3)  # PPG, ECG, ACC


def _to_chan_first(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` reshaped to (channels, samples).

    BUT-PPG ``.hea`` files have an unusual format that confuses wfdb's parser;
    depending on the record, p_signal can come out as either (samples, channels)
    or (channels, samples). We assume samples >> channels (BUT-PPG max 3 channels
    on ACC; PPG/ECG single-channel) and orient the longer axis last.
    """
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.shape[0] > arr.shape[1]:
        return arr.T
    return arr


def _ensure_root_files(raw_dir: Path, *, force: bool, logger) -> None:
    for rel in ROOT_FILES:
        local = raw_dir / rel
        if local.exists() and not force:
            continue
        url = physionet_url(SLUG, VERSION, rel)
        http_download(url, local)


def _read_record_ids(raw_dir: Path) -> list[str]:
    """RECORDS.txt holds entries like ``100001/100001_ECG`` / ``100001/100001_PPG``.

    Return the unique 6-digit session IDs in source order.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    with (raw_dir / "RECORDS.txt").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid = line.split("/", 1)[0]
            if sid not in seen_set:
                seen_set.add(sid)
                seen.append(sid)
    return seen


def _read_quality_hr(raw_dir: Path) -> dict[str, tuple[int, float]]:
    """Map session_id → (quality_flag, reference_hr_bpm)."""
    out: dict[str, tuple[int, float]] = {}
    with (raw_dir / "quality-hr-ann.csv").open() as f:
        reader = csv.reader(f)
        # Optionally skip header if the first cell isn't a 6-digit id.
        for row in reader:
            if not row or not row[0].isdigit():
                continue
            sid = row[0].strip()
            try:
                q = int(row[1])
                hr = float(row[2])
            except (IndexError, ValueError):
                continue
            out[sid] = (q, hr)
    return out


def _fetch_session(sid: str, raw_dir: Path, *, force: bool, logger) -> bool:
    """Download all files for one BUT-PPG session.

    Returns True if PPG and ECG records are available, False if either failed.
    """
    rels: list[str] = []
    # PPG and ECG always exist; ACC and .qrs may be missing for older sessions.
    for kind in ("PPG", "ECG", "ACC"):
        for ext in (".dat", ".hea"):
            rels.append(f"{sid}/{sid}_{kind}{ext}")
    rels.append(f"{sid}/{sid}.qrs")

    for rel in rels:
        local = raw_dir / rel
        if local.exists() and not force:
            continue
        try:
            download_physionet_files(SLUG, VERSION, [rel], raw_dir, show_progress=False)
        except Exception as exc:  # noqa: BLE001 - some files (ACC, qrs) are optional
            if any(rel.endswith(opt) for opt in ("_ACC.dat", "_ACC.hea", ".qrs")):
                logger.debug("optional file missing for %s: %s", sid, rel)
            else:
                logger.warning("failed to download %s: %s", rel, exc)
                return False
    return (raw_dir / sid / f"{sid}_PPG.hea").exists() and (raw_dir / sid / f"{sid}_ECG.hea").exists()


def _convert_session(raw_dir: Path, sid: str, qhr: tuple[int, float] | None,
                     out_dir: Path, *, logger) -> Path | None:
    sess = raw_dir / sid
    ppg_rec = wfdb.rdrecord(str(sess / f"{sid}_PPG"))
    ecg_rec = wfdb.rdrecord(str(sess / f"{sid}_ECG"))
    # BUT-PPG .hea files have non-standard formatting; normalize to (channels, samples).
    ppg = _to_chan_first(np.asarray(ppg_rec.p_signal, dtype=np.float32))
    ecg = _to_chan_first(np.asarray(ecg_rec.p_signal, dtype=np.float32))

    acc_path = sess / f"{sid}_ACC"
    acc = None
    acc_rec = None
    if (sess / f"{sid}_ACC.hea").exists():
        try:
            acc_rec = wfdb.rdrecord(str(acc_path))
            acc = _to_chan_first(np.asarray(acc_rec.p_signal, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            logger.debug("ACC unreadable for %s: %s", sid, exc)

    qrs_idx = None
    if (sess / f"{sid}.qrs").exists():
        try:
            ann = wfdb.rdann(str(sess / sid), "qrs")
            qrs_idx = np.asarray(ann.sample, dtype=np.int64)
        except Exception as exc:  # noqa: BLE001
            logger.debug("QRS unreadable for %s: %s", sid, exc)

    out = out_dir / f"{sid}.h5"
    with h5py.File(out, "w") as h:
        h.create_dataset("data", data=ppg)
        h.create_dataset("ecg", data=ecg)
        if acc is not None:
            h.create_dataset("acc", data=acc)
            h.attrs["fs_acc"] = int(getattr(acc_rec, "fs", 100))
        if qrs_idx is not None:
            h.create_dataset("r_peaks", data=qrs_idx)
        h.attrs["fs"] = int(ppg_rec.fs)
        h.attrs["fs_ecg"] = int(ecg_rec.fs)
        h.attrs["channel_names"] = "PPG"
        h.attrs["lead_names"] = "PPG"
        h.attrs["source"] = SLUG
        h.attrs["acquisition"] = ACQUISITION
        # Subject id is the leading 3 chars (per BUT PPG numbering convention).
        h.attrs["patient_id"] = sid[:3]
        h.attrs["session_id"] = sid
        if qhr is not None:
            h.attrs["quality"] = qhr[0]
            h.attrs["reference_hr_bpm"] = qhr[1]
    logger.debug("Wrote %s ppg=%s ecg=%s acc=%s qrs=%s",
                 out.name, ppg.shape, ecg.shape,
                 acc.shape if acc is not None else None,
                 qrs_idx.shape if qrs_idx is not None else None)
    return out


def main() -> None:
    parser = make_parser(slug=SLUG, description=__doc__)
    args = parser.parse_args()
    logger = setup_logging(args)

    canonical, raw = resolve_dirs(args, SLUG)
    logger.info("Canonical dir: %s", canonical)
    logger.info("Raw WFDB dir:  %s", raw)

    if not args.skip_download:
        _ensure_root_files(raw, force=args.force, logger=logger)
    sessions = _read_record_ids(raw)
    if args.limit is not None:
        sessions = sessions[: args.limit]
        logger.info("Smoke test: limiting to %d sessions", len(sessions))
    logger.info("Discovered %d sessions", len(sessions))

    qhr_map = _read_quality_hr(raw)

    if not args.skip_download:
        for i, sid in enumerate(sessions, 1):
            if i % 100 == 0:
                logger.info("  fetched %d/%d", i, len(sessions))
            _fetch_session(sid, raw, force=args.force, logger=logger)

    if not args.skip_convert:
        for sid in sessions:
            out = canonical / f"{sid}.h5"
            if out.exists() and not args.force:
                continue
            try:
                _convert_session(raw, sid, qhr_map.get(sid), canonical, logger=logger)
            except Exception as exc:  # noqa: BLE001
                logger.warning("convert failed for %s: %s", sid, exc)

    n = len(list(canonical.glob("*.h5")))
    logger.info("Canonical h5 count: %d", n)
    if n:
        sample = sorted(canonical.glob("*.h5"))[0]
        with h5py.File(sample, "r") as h:
            logger.info(
                "  %s ppg=%s fs=%s ecg=%s fs_ecg=%s acquisition=%s",
                sample.name, tuple(h["data"].shape), h.attrs["fs"],
                tuple(h["ecg"].shape) if "ecg" in h else None,
                h.attrs.get("fs_ecg"), h.attrs["acquisition"],
            )

    maybe_upload_s3(canonical, args.upload_s3, slug=SLUG)


if __name__ == "__main__":
    main()
