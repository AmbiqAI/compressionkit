"""Ingest the BIDMC PPG and Respiration Dataset (bidmc) from PhysioNet.

BIDMC: 53 eight-minute simultaneous ECG/PPG/respiration excerpts from adult ICU
patients (Beth Israel Deaconess), sampled at 125 Hz with reference HR/RR/SpO2.
This is the gold-standard clinical-bedside PPG benchmark used in nearly every
PPG paper for HR/RR estimation evaluation.

Records: ``bidmc01`` … ``bidmc53`` (53 records). Each WFDB record has channels
``II`` (ECG), ``V`` or ``AVR`` (ECG), ``PLETH`` (PPG), ``RESP`` (impedance
respiration). Total raw is ~30 MB.

Usage::

    python scripts/datasets/download_bidmc.py --limit 3   # smoke
    python scripts/datasets/download_bidmc.py             # full
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

SLUG = "bidmc"
ACQUISITION = "clinic-bedside-ppg"
RECORDS: tuple[str, ...] = tuple(f"bidmc{i:02d}" for i in range(1, 54))


def _fetch_all(raw_dir: Path, *, force: bool, logger) -> None:
    """One-shot pull of the whole BIDMC PhysioNet database (~30 MB)."""
    sentinel = raw_dir / "bidmc01.hea"
    if sentinel.exists() and not force:
        logger.debug("BIDMC raw already present at %s", raw_dir)
        return
    logger.info("Downloading BIDMC database (%d records) to %s …", len(RECORDS), raw_dir)
    wfdb.io.dl_database("bidmc", dl_dir=str(raw_dir), keep_subdirs=False)


def _convert_record(raw_dir: Path, record: str, out_dir: Path, *, logger) -> Path:
    rec = wfdb.rdrecord(str(raw_dir / record))
    sigs = np.asarray(rec.p_signal, dtype=np.float32).T  # (channels, samples)
    fs = int(rec.fs)
    names = [n.upper().strip(" ,") for n in rec.sig_name]

    # Locate PPG (PLETH) channel and place it as the primary "data" array.
    try:
        ppg_idx = names.index("PLETH")
    except ValueError as e:
        raise RuntimeError(f"{record}: no PLETH channel found in {names}") from e

    ppg = sigs[ppg_idx : ppg_idx + 1, :]  # keep 2D (1, samples)

    # Group remaining channels by type for convenience downstream.
    ecg_idx = [i for i, n in enumerate(names) if n in ("I", "II", "III", "V", "V1", "AVR", "AVL", "AVF")]
    resp_idx = [i for i, n in enumerate(names) if n.startswith("RESP")]

    out = out_dir / f"{record}.h5"
    with h5py.File(out, "w") as h:
        h.create_dataset("data", data=ppg)
        if ecg_idx:
            h.create_dataset("ecg", data=sigs[ecg_idx])
            h.attrs["ecg_lead_names"] = ",".join(names[i] for i in ecg_idx)
        if resp_idx:
            h.create_dataset("resp", data=sigs[resp_idx])
        h.attrs["fs"] = fs
        h.attrs["fs_ecg"] = fs
        h.attrs["fs_resp"] = fs
        h.attrs["channel_names"] = "PLETH"
        h.attrs["lead_names"] = "PLETH"
        h.attrs["source"] = SLUG
        h.attrs["acquisition"] = ACQUISITION
        h.attrs["patient_id"] = record
        h.attrs["units"] = ",".join(rec.units or [])
    logger.debug("Wrote %s ppg=%s ecg=%d resp=%d fs=%d",
                 out.name, ppg.shape, len(ecg_idx), len(resp_idx), fs)
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
        _fetch_all(raw, force=args.force, logger=logger)

    if not args.skip_convert:
        for rec in records:
            out = canonical / f"{rec}.h5"
            if out.exists() and not args.force:
                logger.debug("Skip convert (exists): %s", out.name)
                continue
            _convert_record(raw, rec, canonical, logger=logger)

    n = len(list(canonical.glob("*.h5")))
    logger.info("Canonical h5 count: %d", n)
    if n:
        sample = sorted(canonical.glob("*.h5"))[0]
        with h5py.File(sample, "r") as h:
            logger.info(
                "  %s data=%s fs=%s ecg=%s acquisition=%s",
                sample.name, tuple(h["data"].shape), h.attrs["fs"],
                tuple(h["ecg"].shape) if "ecg" in h else None, h.attrs["acquisition"],
            )

    maybe_upload_s3(canonical, args.upload_s3, slug=SLUG)


if __name__ == "__main__":
    main()
