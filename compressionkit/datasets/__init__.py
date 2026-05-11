"""Dataset loading modules for compressionkit.

Provides two layers:

1. **Dataset classes** — download/discover raw data and load individual signals:

   - :class:`PtbxlDataset` — PTB-XL 12-lead ECG (auto-downloads from S3)
   - :class:`MesaDataset` — MESA PPG (restricted; user-provided EDF files)

2. **Pipeline helpers** — build ``tf.data.Dataset`` pipelines from raw data
   for RVQ training (TFRecord caching, streaming, in-memory).
"""

from compressionkit.datasets.defines import DatasetInfo
from compressionkit.datasets.ecg import (
    build_ecg_tfrecord_cache,
    load_ecg_dataset,
    load_ecg_file_splits,
    load_ecg_signal,
    load_ecg_splits,
    make_ecg_inmemory_dataset,
    make_ecg_stream_dataset,
    make_ecg_tfrecord_dataset,
)
from compressionkit.datasets.mesa import MesaDataset
from compressionkit.datasets.ppg import (
    build_ppg_tfrecord_cache,
    load_ppg_dataset,
    load_ppg_file_splits,
    load_ppg_signal,
    load_ppg_splits,
    make_ppg_stream_dataset,
    make_ppg_tfrecord_dataset,
)
from compressionkit.datasets.ptbxl import PtbxlDataset

__all__ = [
    # Dataset classes
    "DatasetInfo",
    "MesaDataset",
    "PtbxlDataset",
    # ECG pipeline helpers
    "build_ecg_tfrecord_cache",
    # PPG pipeline helpers
    "build_ppg_tfrecord_cache",
    "load_ecg_dataset",
    "load_ecg_file_splits",
    "load_ecg_signal",
    "load_ecg_splits",
    "load_ppg_dataset",
    "load_ppg_file_splits",
    "load_ppg_signal",
    "load_ppg_splits",
    "make_ecg_inmemory_dataset",
    "make_ecg_stream_dataset",
    "make_ecg_tfrecord_dataset",
    "make_ppg_stream_dataset",
    "make_ppg_tfrecord_dataset",
]
