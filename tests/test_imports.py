"""Smoke tests confirming the public package surface imports cleanly."""

from __future__ import annotations

import importlib

import pytest

PUBLIC_MODULES = [
    "compressionkit",
    "compressionkit.configs.ecg_rvq",
    "compressionkit.configs.ppg_rvq",
    "compressionkit.datasets.ecg",
    "compressionkit.datasets.ppg",
    "compressionkit.dsp",
    "compressionkit.dsp.spiht",
    "compressionkit.dsp.transforms",
    "compressionkit.dsp.wavelet",
    "compressionkit.evaluation.metrics",
    "compressionkit.export.tflite",
    "compressionkit.export.codebook",
    "compressionkit.layers.residual_vector_quantizer",
    "compressionkit.models.rvq_autoencoder",
    "compressionkit.preprocessing.ecg",
    "compressionkit.preprocessing.ppg",
    "compressionkit.recipes",
    "compressionkit.recipes.train_ecg_rvq",
    "compressionkit.recipes.train_ppg_rvq",
    "compressionkit.trainers.common",
    "compressionkit.trainers.ecg_rvq",
    "compressionkit.trainers.ppg_rvq",
]


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_module_imports(module_name: str) -> None:
    """Every advertised public module must import without error."""
    importlib.import_module(module_name)
