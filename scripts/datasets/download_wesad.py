"""Ingest the WESAD wearable stress & affect detection dataset (UCI 465).

WESAD: 15 subjects (S2..S17, with S1 / S12 missing per the publication) wearing
an Empatica E4 wrist device + RespiBAN chest strap during baseline,
amusement, stress and meditation conditions. Per subject:

  - Wrist (E4): BVP/PPG @ 64 Hz, ACC @ 32 Hz, EDA @ 4 Hz, TEMP @ 4 Hz.
  - Chest (RespiBAN): ECG / EMG / Resp @ 700 Hz, ACC / EDA / Temp @ 700 Hz.
  - Per-sample stress-condition label (1=baseline, 2=stress, 3=amusement,
    4=meditation, plus 0/5/6/7 for transitions and ignore).

WESAD complements PPG-DaLiA by providing the controlled-stressor PPG bucket.

Usage::

    python scripts/datasets/download_wesad.py --limit 2   # smoke
    python scripts/datasets/download_wesad.py             # full (~1.5 GB)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import make_parser, maybe_upload_s3, resolve_dirs, setup_logging

from compressionkit.datasets._download import extract_archive, http_download

SLUG = "wesad"
ACQUISITION = "wearable-wrist-ppg-stress"
# UCI ships only a 261-byte placeholder pointing at the authors' Sciebo share;
# fetch the real 2.25 GB zip directly. URL confirmed via the project home page
# at https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html .
ZIP_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
ZIP_NAME = "WESAD.zip"
# WESAD ships subject IDs S2..S11, S13..S17 (S1 and S12 are intentionally absent).
SUBJECTS: tuple[str, ...] = tuple(
    f"S{i}" for i in range(2, 18) if i != 12
)


def _ensure_extracted(raw_dir: Path, *, force: bool, logger) -> Path:
    zip_path = raw_dir / ZIP_NAME
    if force or not zip_path.exists():
        http_download(ZIP_URL, zip_path)
    # WESAD UCI zip contains an inner ``WESAD/`` dir with one folder per subject.
    inner_root = next((d for d in raw_dir.rglob("S2") if (d / "S2.pkl").exists()), None)
    if force or inner_root is None:
        extract_archive(zip_path, raw_dir)
        # Sometimes UCI nests it inside another zip.
        for nested in raw_dir.glob("WESAD*.zip"):
            extract_archive(nested, raw_dir)
        inner_root = next((d for d in raw_dir.rglob("S2") if (d / "S2.pkl").exists()), None)
    if inner_root is None:
        raise RuntimeError(f"WESAD layout not found under {raw_dir}")
    return inner_root.parent


def _convert_subject(study_dir: Path, sid: str, out_dir: Path, *, logger) -> Path | None:
    pkl = study_dir / sid / f"{sid}.pkl"
    if not pkl.exists():
        logger.warning("missing pickle: %s", pkl)
        return None
    with pkl.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist = data["signal"]["wrist"]
    chest = data["signal"]["chest"]

    ppg = np.asarray(wrist["BVP"], dtype=np.float32).reshape(1, -1)
    acc_wrist = np.asarray(wrist["ACC"], dtype=np.float32).T
    ecg = np.asarray(chest["ECG"], dtype=np.float32).reshape(1, -1)
    resp = np.asarray(chest["Resp"], dtype=np.float32).reshape(1, -1) if "Resp" in chest else None
    label = np.asarray(data["label"], dtype=np.int16).reshape(-1)  # @ 700 Hz

    out = out_dir / f"{sid}.h5"
    with h5py.File(out, "w") as h:
        h.create_dataset("data", data=ppg)
        h.create_dataset("ecg", data=ecg)
        h.create_dataset("acc", data=acc_wrist)
        if resp is not None:
            h.create_dataset("resp", data=resp)
        h.create_dataset("stress_label", data=label)
        h.attrs["fs"] = 64
        h.attrs["fs_ecg"] = 700
        h.attrs["fs_acc"] = 32
        h.attrs["fs_resp"] = 700
        h.attrs["fs_stress_label"] = 700
        h.attrs["channel_names"] = "BVP"
        h.attrs["lead_names"] = "BVP"
        h.attrs["source"] = SLUG
        h.attrs["acquisition"] = ACQUISITION
        h.attrs["patient_id"] = sid
    logger.debug("Wrote %s ppg=%s ecg=%s acc=%s label=%s",
                 out.name, ppg.shape, ecg.shape, acc_wrist.shape, label.shape)
    return out


def main() -> None:
    parser = make_parser(slug=SLUG, description=__doc__)
    args = parser.parse_args()
    logger = setup_logging(args)

    canonical, raw = resolve_dirs(args, SLUG)
    logger.info("Canonical dir: %s", canonical)
    logger.info("Raw dir:       %s", raw)

    if not args.skip_download:
        study_dir = _ensure_extracted(raw, force=args.force, logger=logger)
    else:
        inner_root = next((d for d in raw.rglob("S2") if (d / "S2.pkl").exists()), None)
        if inner_root is None:
            raise RuntimeError(f"could not find extracted dataset under {raw}")
        study_dir = inner_root.parent
    logger.info("Study dir: %s", study_dir)

    subjects = list(SUBJECTS)
    if args.limit is not None:
        subjects = subjects[: args.limit]
        logger.info("Smoke test: limiting to %d subjects", len(subjects))

    if not args.skip_convert:
        for sid in subjects:
            out = canonical / f"{sid}.h5"
            if out.exists() and not args.force:
                continue
            _convert_subject(study_dir, sid, canonical, logger=logger)

    n = len(list(canonical.glob("*.h5")))
    logger.info("Canonical h5 count: %d", n)
    if n:
        sample = sorted(canonical.glob("*.h5"))[0]
        with h5py.File(sample, "r") as h:
            logger.info(
                "  %s ppg=%s ecg=%s acc=%s label=%s acquisition=%s",
                sample.name, tuple(h["data"].shape), tuple(h["ecg"].shape),
                tuple(h["acc"].shape), tuple(h["stress_label"].shape),
                h.attrs["acquisition"],
            )

    maybe_upload_s3(canonical, args.upload_s3, slug=SLUG)


if __name__ == "__main__":
    main()
