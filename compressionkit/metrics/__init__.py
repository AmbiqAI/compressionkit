"""Training-time Keras metrics for signal compression.

This module provides clean imports for metrics used during model training.
For post-training evaluation utilities, see ``compressionkit.evaluation``.
"""

from compressionkit.evaluation.metrics import PRD, TruePRD
from compressionkit.evaluation.spectral_metrics import (
    psd_band_error,
    spectral_coherence,
    weighted_freq_prd,
)

__all__ = [
    "PRD",
    "TruePRD",
    "psd_band_error",
    "spectral_coherence",
    "weighted_freq_prd",
]
