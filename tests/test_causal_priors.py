"""Tests for compressionkit.generative.causal_priors.

Focused on `SplitHalf` and the save/load round-trip of the WaveNet-style
prior architectures that use it (`build_wavenet_prior`, `build_wavenet_bit_prior`).

Regression coverage for a real bug: these architectures originally used
`keras.layers.Lambda(lambda t: ...)` to split a gated-conv output into its
(filter, gate) halves. A `Lambda` wrapping a Python closure serializes its
bytecode, which reliably FAILED to deserialize via `keras.models.load_model()`
(even with `safe_mode=False`) — breaking any workflow that trains once and
reloads later (anything shipped/deployed). `SplitHalf` is a real Keras layer
with `get_config()`/`get_weights()`-free state, so it has none of that
fragility — these tests assert the round-trip now produces bit-identical
predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from compressionkit.generative.causal_priors import (
    SplitHalf,
    build_wavenet_bit_prior,
    build_wavenet_prior,
)


def test_split_half_basic():
    layer = SplitHalf(split_size=3)
    x = np.arange(12, dtype="float32").reshape(2, 6)
    a, b = layer(x)
    assert np.array_equal(np.asarray(a), x[:, :3])
    assert np.array_equal(np.asarray(b), x[:, 3:])


def test_split_half_get_config_roundtrip():
    layer = SplitHalf(split_size=5, name="my_split")
    config = layer.get_config()
    assert config["split_size"] == 5
    rebuilt = SplitHalf.from_config(config)
    assert rebuilt.split_size == 5


def test_wavenet_prior_save_load_roundtrip(tmp_path: Path):
    """RVQ token prior: predictions must be bit-identical after a save/load cycle."""
    rng = np.random.default_rng(0)
    model = build_wavenet_prior(vocab_size=64, context_length=16, embed_dim=8, num_layers=3)
    tokens = rng.integers(0, 64, size=(3, 16)).astype("int32")

    before = model.predict(tokens, verbose=0)
    path = tmp_path / "wavenet_prior.keras"
    model.save(path)
    reloaded = __import__("keras").models.load_model(path)
    after = reloaded.predict(tokens, verbose=0)

    assert np.array_equal(before, after)


def test_wavenet_bit_prior_save_load_roundtrip(tmp_path: Path):
    """SPIHT bit prior: predictions must be bit-identical after a save/load cycle."""
    rng = np.random.default_rng(1)
    cl = 20
    model = build_wavenet_bit_prior(cl, embed_dim=8, num_layers=3)
    ctx = rng.integers(0, 6, size=(3, cl)).astype("int32")
    bp = rng.integers(0, 16, size=(3, cl)).astype("int32")
    prev = rng.integers(0, 3, size=(3, cl)).astype("int32")

    before = model.predict([ctx, bp, prev], verbose=0)
    path = tmp_path / "wavenet_bit_prior.keras"
    model.save(path)
    reloaded = __import__("keras").models.load_model(path)
    after = reloaded.predict([ctx, bp, prev], verbose=0)

    assert np.array_equal(before, after)


def test_wavenet_prior_trains_after_reload(tmp_path: Path):
    """A reloaded model must still be trainable (the full graph survives intact)."""
    rng = np.random.default_rng(2)
    model = build_wavenet_prior(vocab_size=32, context_length=8, embed_dim=8, num_layers=2)
    tokens = rng.integers(0, 32, size=(16, 8)).astype("int32")
    labels = rng.integers(0, 32, size=(16, 8)).astype("int32")

    path = tmp_path / "m.keras"
    model.save(path)
    reloaded = __import__("keras").models.load_model(path)
    reloaded.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    # Should not raise — confirms the reloaded model is a fully usable, trainable graph.
    reloaded.fit(tokens, labels, epochs=1, verbose=0)
