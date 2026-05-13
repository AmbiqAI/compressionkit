"""Ingest the PPG-DaLiA dataset (UCI 495).

PPG-DaLiA: 15 subjects performing 8 daily-life activity scripts (sit, work,
walk, drive, climb stairs, play table soccer, etc.) wearing an Empatica E4
wrist device + RespiBAN chest strap. Per subject:

  - Wrist (E4): BVP/PPG @ 64 Hz, ACC @ 32 Hz, EDA @ 4 Hz, TEMP @ 4 Hz.
  - Chest (RespiBAN): ECG / Resp / EMG @ 700 Hz, ACC / EDA / Temp @ 4 Hz.
  - HR ground truth at 0.5 Hz from chest ECG.
  - Activity labels and per-subject questionnaires.

This is the largest free-living wrist-PPG benchmark with paired chest ECG and
is the standard test set for HR-from-PPG evaluation under motion.

Usage::

    python scripts/datasets/download_ppg_dalia.py --limit 2  # smoke
    python scripts/datasets/download_ppg_dalia.py            # full (~1 GB zip)
"""

from __future__ import annotations

import contextlib
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

SLUG = "ppg_dalia"
ACQUISITION = "wearable-wrist-ppg"
ZIP_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
ZIP_NAME = "ppg_dalia.zip"
SUBJECTS: tuple[str, ...] = tuple(f"S{i}" for i in range(1, 16))


def _ensure_extracted(raw_dir: Path, *, force: bool, logger) -> Path:
    """Download + extract the dataset zip; return the directory holding S1..S15."""
    zip_path = raw_dir / ZIP_NAME
    if force or not zip_path.exists():
        http_download(ZIP_URL, zip_path)
    # First extraction unpacks an outer ``data.zip`` (plus readme.pdf).
    outer = raw_dir / "PPG_FieldStudy"
    if force or not outer.exists():
        extract_archive(zip_path, raw_dir)
        # UCI bundles a nested zip; the inner archive is named ``data.zip``
        # in the 2019 release but historically was ``PPG_FieldStudy*.zip``.
        nested = next(
            (p for p in raw_dir.glob("*.zip") if p.name != ZIP_NAME),
            None,
        )
        if nested is not None:
            extract_archive(nested, raw_dir)
    # Final layout has S1/S1.pkl … S15/S15.pkl under PPG_FieldStudy/.
    candidates = [d for d in raw_dir.rglob("S1") if (d / "S1.pkl").exists()]
    if not candidates:
        raise RuntimeError(f"PPG-DaLiA layout not found under {raw_dir}")
    return candidates[0].parent


def _convert_subject(study_dir: Path, sid: str, out_dir: Path, *, logger) -> Path | None:
    pkl = study_dir / sid / f"{sid}.pkl"
    if not pkl.exists():
        logger.warning("missing pickle: %s", pkl)
        return None
    with pkl.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist = data["signal"]["wrist"]
    chest = data["signal"]["chest"]

    ppg = np.asarray(wrist["BVP"], dtype=np.float32).reshape(1, -1)  # (1, N) @ 64
    acc_wrist = np.asarray(wrist["ACC"], dtype=np.float32).T  # (3, M) @ 32
    ecg = np.asarray(chest["ECG"], dtype=np.float32).reshape(1, -1)  # (1, K) @ 700
    resp = np.asarray(chest.get("Resp"), dtype=np.float32).reshape(1, -1) if "Resp" in chest else None
    hr_label = np.asarray(data["label"], dtype=np.float32).reshape(-1)  # @ 0.5 Hz

    out = out_dir / f"{sid}.h5"
    with h5py.File(out, "w") as h:
        h.create_dataset("data", data=ppg)
        h.create_dataset("ecg", data=ecg)
        h.create_dataset("acc", data=acc_wrist)
        if resp is not None:
            h.create_dataset("resp", data=resp)
        h.create_dataset("hr_label", data=hr_label)
        h.attrs["fs"] = 64
        h.attrs["fs_ecg"] = 700
        h.attrs["fs_acc"] = 32
        h.attrs["fs_resp"] = 700
        h.attrs["fs_hr_label"] = 0.5
        h.attrs["channel_names"] = "BVP"
        h.attrs["lead_names"] = "BVP"
        h.attrs["source"] = SLUG
        h.attrs["acquisition"] = ACQUISITION
        h.attrs["patient_id"] = sid
        if "activity" in data:
            with contextlib.suppress(Exception):
                h.create_dataset("activity", data=np.asarray(data["activity"]).astype(np.int16).reshape(-1))
    logger.debug("Wrote %s ppg=%s ecg=%s acc=%s hr=%s", out.name, ppg.shape, ecg.shape, acc_wrist.shape, hr_label.shape)
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
        candidates = [d for d in raw.rglob("S1") if (d / "S1.pkl").exists()]
        if not candidates:
            raise RuntimeError(f"could not find extracted dataset under {raw}")
        study_dir = candidates[0].parent
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
                "  %s ppg=%s ecg=%s acc=%s hr=%s acquisition=%s",
                sample.name,
                tuple(h["data"].shape),
                tuple(h["ecg"].shape),
                tuple(h["acc"].shape),
                tuple(h["hr_label"].shape),
                h.attrs["acquisition"],
            )

    maybe_upload_s3(canonical, args.upload_s3, slug=SLUG)


if __name__ == "__main__":
    main()
