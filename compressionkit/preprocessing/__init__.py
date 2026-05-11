"""Preprocessing and augmentation pipelines for compressionkit."""

from compressionkit.preprocessing.ecg import (
    build_augmenter as build_ecg_augmenter,
)
from compressionkit.preprocessing.ecg import (
    build_preprocessor as build_ecg_preprocessor,
)
from compressionkit.preprocessing.ecg import (
    generate_synthetic_ecg_batch,
)
from compressionkit.preprocessing.ppg import (
    build_augmenter,
    build_preprocessor,
    generate_synthetic_ppg_batch,
)
from compressionkit.preprocessing.sanitize import (
    SanitizeConfig,
    SanitizeReport,
    is_clean_window,
    normalize_window,
)

__all__ = [
    "SanitizeConfig",
    "SanitizeReport",
    "build_augmenter",
    "build_ecg_augmenter",
    "build_ecg_preprocessor",
    "build_preprocessor",
    "generate_synthetic_ecg_batch",
    "generate_synthetic_ppg_batch",
    "is_clean_window",
    "normalize_window",
]
