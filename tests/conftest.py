"""Pytest configuration for compressionKIT tests.

Keep deterministic defaults and silence verbose TF logging so tests stay
quick and readable.
"""

from __future__ import annotations

import os

# Keep TensorFlow quiet during tests.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _deterministic_seeds() -> None:
    """Seed numpy before every test for deterministic behavior."""
    np.random.seed(0)
