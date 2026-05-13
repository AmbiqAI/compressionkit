"""MESA PPG dataset (Multi-Ethnic Study of Atherosclerosis).

MESA polysomnography data contains overnight PPG recordings at 256 Hz stored
as EDF files.  **This is a restricted-access dataset** — users must apply for
access through the `National Sleep Research Resource (NSRR)
<https://sleepdata.org/datasets/mesa>`_ and download the EDF files themselves.

This class provides a consistent loading interface once the data is present
locally.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import physiokit as pk
import pyedflib

from .defines import DatasetInfo

logger = logging.getLogger(__name__)

INFO = DatasetInfo(
    name="mesa",
    sampling_rate=256,
    num_leads=1,
    description=("Overnight PPG recordings (256 Hz) from the MESA polysomnography study.  ~1,900 subjects."),
    license="NSRR Data Use Agreement",
    license_tier="restricted",
    requires_agreement=True,
)

# Default glob pattern to find EDF files within the MESA directory tree.
_DEFAULT_GLOB = "**/*.edf"
_DEFAULT_PPG_LABEL = "Pleth"


class MesaDataset:
    """MESA PPG dataset — user-provided EDF files.

    Args:
        path: Root directory containing MESA data.  EDF files are located
            by recursive glob.
        ppg_label: EDF channel label for the PPG signal.

    Example::

        ds = MesaDataset(path="./datasets/mesa-commercial-use")
        print(len(ds.patient_ids))  # ~1900

        signal = ds.load_signal(patient_id=0, target_rate=64)
    """

    def __init__(
        self,
        path: str | os.PathLike = "datasets/mesa-commercial-use",
        ppg_label: str = _DEFAULT_PPG_LABEL,
    ) -> None:
        self.path = Path(path)
        self.ppg_label = ppg_label
        self._edf_paths: list[Path] | None = None

    def ensure_available(self) -> None:
        """Raise :class:`DatasetNotAvailableError` if no EDF files are present.

        MESA is a restricted dataset; the remediation message points
        at ``MesaDataset.download()`` which requires an NSRR token.
        """
        from compressionkit.datasets.contract import DatasetNotAvailableError

        if self.path.is_dir() and any(self.path.glob(_DEFAULT_GLOB)):
            return
        remediation = (
            "export NSRR_TOKEN=<your-token>\n"
            f'  python -c "from compressionkit.datasets import MesaDataset;'
            f" MesaDataset(path='{self.path}').download()\""
        )
        raise DatasetNotAvailableError("mesa", self.path, remediation)

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
    # EDF discovery
    # ------------------------------------------------------------------

    @property
    def edf_paths(self) -> list[Path]:
        """Sorted list of discovered EDF file paths."""
        if self._edf_paths is None:
            self._edf_paths = sorted(self.path.glob(_DEFAULT_GLOB))
            if not self._edf_paths:
                raise FileNotFoundError(
                    f"No EDF files found under {self.path}. "
                    "MESA is a restricted dataset — you must download the "
                    "EDF files from https://sleepdata.org/datasets/mesa "
                    "and place them under this directory."
                )
        return self._edf_paths

    @property
    def patient_ids(self) -> npt.NDArray[np.int32]:
        """Integer indices ``[0, N)`` for discovered EDF files."""
        return np.arange(len(self.edf_paths), dtype=np.int32)

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

    def load_signal(
        self,
        patient_id: int,
        *,
        target_rate: int | None = None,
        offset_samples: int = 0,
        num_samples: int | None = None,
    ) -> np.ndarray:
        """Load PPG signal for a single patient.

        Args:
            patient_id: 0-based index into :attr:`edf_paths`.
            target_rate: Resample to this rate.  ``None`` keeps native rate.
            offset_samples: Skip this many samples from the start (at native
                rate).
            num_samples: Return exactly this many samples (at *target_rate*
                if resampling, else native).  ``None`` returns all.

        Returns:
            1-D ``float32`` array.
        """
        edf_path = self.edf_paths[patient_id]
        source_rate = self.sampling_rate

        with pyedflib.EdfReader(str(edf_path)) as reader:
            labels = reader.getSignalLabels()
            if self.ppg_label not in labels:
                raise ValueError(f"Channel '{self.ppg_label}' not found in {edf_path}. Available: {labels}")
            idx = labels.index(self.ppg_label)
            source_rate = int(reader.samplefrequency(idx))
            total = reader.getNSamples()[idx]
            start = min(offset_samples, max(total - 1, 0))

            read_len = total - start
            if num_samples is not None:
                out_rate = target_rate or source_rate
                read_len = min(
                    read_len,
                    math.ceil(num_samples * source_rate / out_rate),
                )

            signal = reader.readSignal(idx, start, read_len, digital=False)

        signal = signal.astype(np.float32)

        if target_rate is not None and target_rate != source_rate:
            signal = pk.signal.resample_signal(signal, source_rate, target_rate)

        if num_samples is not None:
            if signal.shape[0] < num_samples:
                raise ValueError(
                    f"Signal from {edf_path} shorter ({signal.shape[0]}) than requested {num_samples} samples."
                )
            signal = signal[:num_samples]

        return signal

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self, *, token: str | None = None, commercial: bool = True, **kwargs) -> int:
        """Download MESA EDF files from NSRR.

        Requires an NSRR account with approved access to the MESA dataset.
        Set the ``NSRR_TOKEN`` environment variable or pass *token* directly.

        Args:
            token: NSRR API token.  Falls back to ``NSRR_TOKEN`` env var.
            commercial: Use the ``mesa-commercial-use`` slug (default).

        Returns:
            Number of files downloaded.

        Raises:
            EnvironmentError: If no token is available.
            RuntimeError: If authentication or download fails.
        """
        from compressionkit.datasets.nsrr import download_nsrr

        db_slug = "mesa-commercial-use" if commercial else "mesa"
        return download_nsrr(
            db_slug,
            self.path,
            subfolder="polysomnography/edfs",
            pattern="*.edf",
            token=token,
        )
