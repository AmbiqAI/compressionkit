"""Pytest configuration for compressionKIT tests.

Keep deterministic defaults and silence verbose TF logging so tests stay
quick and readable.
"""

from __future__ import annotations

import os
import sys

# Ensure the repo root is on sys.path so ``from scripts.…`` imports work
# regardless of the working directory (e.g. in CI).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Keep TensorFlow quiet during tests.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _deterministic_seeds() -> None:
    """Seed numpy before every test for deterministic behavior."""
    np.random.seed(0)
