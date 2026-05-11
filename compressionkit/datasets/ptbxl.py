"""PTB-XL ECG dataset.

The PTB-XL dataset contains 21,799 12-lead ECG recordings (10 s, 500 Hz)
from 18,885 patients.  Pre-processed H5 files are hosted on S3 and can be
downloaded with :meth:`PtbxlDataset.download`.

Reference:
    Wagner, P. et al. (2020). *PTB-XL, a large publicly available
    electrocardiography dataset.* Scientific Data, 7(1), 154.

License:
    Creative Commons Attribution 4.0 International Public License.
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from .defines import DatasetInfo

logger = logging.getLogger(__name__)

# Bucket and key matching heartkit convention.
_S3_BUCKET = "ambiq-ai-datasets"
_S3_KEY = "ptbxl/ptbxl.zip"

# Patient IDs known to be corrupt/missing in the public release.
_BAD_IDS: set[int] = {
    137, 139, 140, 141, 142, 143, 145,
    456, 458, 459, 461, 462,
    2506, 2511,
    3795, 3798, 3800, 3832,
    5817,
    7777, 7779, 7782,
    9821, 9825, 9888,
    11810, 11814, 11817, 11838,
    13791, 13793, 13796, 13797, 13799,
    15742,
    18150,
}

INFO = DatasetInfo(
    name="ptbxl",
    sampling_rate=500,
    num_leads=12,
    description=(
        "21,799 clinical 12-lead ECGs (10 s, 500 Hz) from 18,885 subjects."
    ),
    license="CC BY 4.0",
    requires_agreement=False,
)


class PtbxlDataset:
    """PTB-XL ECG dataset with S3 download and per-patient H5 access.

    Args:
        path: Root directory for the dataset.  H5 files are expected (or
            will be placed) directly under this directory.

    Example::

        ds = PtbxlDataset(path="./datasets/ptbxl")
        ds.download()  # one-time

        # Iterate patient signals
        signal = ds.load_signal(patient_id=1)  # (12, 5000) float32

    """

    def __init__(self, path: str | os.PathLike = "datasets/ptbxl") -> None:
        self.path = Path(path)

    @property
    def info(self) -> DatasetInfo:
        """Dataset metadata."""
        return INFO

    @property
    def name(self) -> str:
        """Short identifier."""
        return INFO.name

    @property
    def sampling_rate(self) -> int:
        """Native sample rate in Hz."""
        return INFO.sampling_rate

    # ------------------------------------------------------------------
    # Patient IDs
    # ------------------------------------------------------------------

    @property
    def patient_ids(self) -> npt.NDArray[np.int32]:
        """All valid patient IDs (1-based), excluding known bad records."""
        all_ids = np.arange(1, 21838, dtype=np.int32)
        bad = np.array(sorted(_BAD_IDS), dtype=np.int32)
        return all_ids[~np.isin(all_ids, bad)]

    def get_train_patient_ids(self) -> npt.NDArray[np.int32]:
        """First 80 % of patient IDs (deterministic split)."""
        ids = self.patient_ids
        n = int(len(ids) * 0.8)
        return ids[:n]

    def get_test_patient_ids(self) -> npt.NDArray[np.int32]:
        """Last 20 % of patient IDs (deterministic split)."""
        ids = self.patient_ids
        n = int(len(ids) * 0.8)
        return ids[n:]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _h5_path(self, patient_id: int) -> Path:
        """Return path to the patient's H5 file."""
        return self.path / f"{patient_id:05d}.h5"

    def load_signal(
        self,
        patient_id: int,
        *,
        leads: list[int] | None = None,
    ) -> np.ndarray:
        """Load ECG signal for a single patient.

        Args:
            patient_id: 1-based patient identifier.
            leads: Subset of lead indices to return.  ``None`` returns all 12.

        Returns:
            ``float32`` array of shape ``(num_leads, samples)`` or
            ``(len(leads), samples)`` if *leads* is specified.
        """
        h5_path = self._h5_path(patient_id)
        with h5py.File(h5_path, "r") as h5:
            data = h5["data"][:].astype(np.float32)
        # Ensure channel-first: (leads, samples)
        if data.ndim == 2 and data.shape[0] > data.shape[1]:
            data = data.T
        if leads is not None:
            data = data[leads]
        return data

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self, *, force: bool = False) -> None:
        """Download pre-processed H5 files from S3.

        This downloads a single zip from the ``ambiq-ai-datasets`` S3 bucket,
        extracts per-patient H5 files, then removes the zip.

        Args:
            force: Re-download even if H5 files already exist.
        """
        import helia_edge as helia  # lazy so the rest works w/o helia

        os.makedirs(self.path, exist_ok=True)
        zip_path = self.path / f"{self.name}.zip"

        # Quick existence check: skip if we already have H5 files
        if not force and any(self.path.glob("*.h5")):
            logger.info(
                "PTB-XL H5 files already present in %s — skipping download. "
                "Pass force=True to re-download.",
                self.path,
            )
            return

        logger.info("Downloading PTB-XL dataset from S3 …")
        did_download = helia.utils.download_s3_file(
            key=_S3_KEY,
            dst=zip_path,
            bucket=_S3_BUCKET,
            checksum="size",
        )
        if did_download:
            logger.info("Extracting %s …", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.path)
            zip_path.unlink(missing_ok=True)
            logger.info("PTB-XL dataset ready at %s", self.path)
        else:
            logger.info("PTB-XL zip unchanged (checksum match) — skipping.")
