"""Verify the top-level ``compressionkit`` package exposes its advertised API."""

from __future__ import annotations

import compressionkit

EXPECTED_TOP_LEVEL = [
    "PRD",
    "EmaResidualVectorQuantizer",
    "ResidualVectorQuantizer",
    "FiniteScalarQuantizer",
    "TruePRD",
    "VectorQuantizer",
    "__version__",
    "build_decoder_2d",
    "build_encoder_2d",
    "build_rvq_autoencoder",
    "compute_compression_stats",
    "compute_signal_metrics",
    "evaluate_long_recordings",
    "reconstruct_overlap_add",
]


def test_version_string() -> None:
    assert isinstance(compressionkit.__version__, str)
    assert compressionkit.__version__.count(".") >= 1


def test_top_level_api_exported() -> None:
    """Everything in ``__all__`` must be importable from the top-level package."""
    for name in EXPECTED_TOP_LEVEL:
        assert hasattr(compressionkit, name), f"compressionkit.{name} is missing"
    assert set(EXPECTED_TOP_LEVEL) == set(compressionkit.__all__)
