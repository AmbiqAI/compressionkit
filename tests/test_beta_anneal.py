"""Tests for the BetaAnneal callback."""

from __future__ import annotations

import keras
import pytest
from helia_edge.layers import EmaResidualVectorQuantizer

from compressionkit.callbacks import BetaAnneal


@pytest.fixture
def rvq_layer() -> EmaResidualVectorQuantizer:
    """Build a small EMA RVQ layer for testing."""
    layer = EmaResidualVectorQuantizer(
        num_levels=2,
        num_embeddings=16,
        embedding_dim=8,
        beta=0.25,
        ema_decay=0.99,
    )
    # Build the layer so internal weights exist.
    layer.build((None, 4, 8))
    return layer


def test_beta_anneal_swaps_float_to_variable(rvq_layer):
    """First attach should convert ``beta`` from float to keras.Variable."""
    assert isinstance(rvq_layer.beta, float)
    cb = BetaAnneal(rvq_layer, start=1.0, end=0.1, epochs=10)
    assert isinstance(rvq_layer.beta, keras.Variable)
    assert float(rvq_layer.beta.numpy()) == pytest.approx(0.25)
    # Second instantiation must not double-wrap.
    BetaAnneal(rvq_layer, start=0.5, end=0.05, epochs=5)
    assert isinstance(rvq_layer.beta, keras.Variable)


def test_beta_anneal_cosine_schedule(rvq_layer):
    """Cosine schedule should hit start at epoch 0 and end at final epoch."""
    cb = BetaAnneal(rvq_layer, start=1.0, end=0.1, epochs=4, mode="cosine")
    cb.on_epoch_begin(0)
    assert float(rvq_layer.beta.numpy()) == pytest.approx(1.0, abs=1e-6)
    cb.on_epoch_begin(4)
    assert float(rvq_layer.beta.numpy()) == pytest.approx(0.1, abs=1e-6)
    # Mid-schedule value should be strictly between end and start.
    cb.on_epoch_begin(2)
    mid = float(rvq_layer.beta.numpy())
    assert 0.1 < mid < 1.0
    # Beyond final epoch: clamps to end.
    cb.on_epoch_begin(99)
    assert float(rvq_layer.beta.numpy()) == pytest.approx(0.1, abs=1e-6)


def test_beta_anneal_linear_schedule(rvq_layer):
    """Linear schedule should be exactly halfway at the midpoint."""
    cb = BetaAnneal(rvq_layer, start=1.0, end=0.0, epochs=10, mode="linear")
    cb.on_epoch_begin(5)
    assert float(rvq_layer.beta.numpy()) == pytest.approx(0.5, abs=1e-6)


def test_beta_anneal_invalid_mode(rvq_layer):
    with pytest.raises(ValueError):
        BetaAnneal(rvq_layer, start=1.0, end=0.1, epochs=5, mode="bogus")


def test_beta_anneal_logs_value(rvq_layer):
    cb = BetaAnneal(rvq_layer, start=0.8, end=0.2, epochs=4)
    logs: dict[str, float] = {}
    cb.on_epoch_begin(2, logs=logs)
    cb.on_epoch_end(2, logs=logs)
    assert "rvq_beta" in logs
    assert 0.2 < logs["rvq_beta"] < 0.8
