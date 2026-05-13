"""Tests for the joint rate-distortion training wrapper."""

from __future__ import annotations

import keras
import numpy as np
import pytest
from helia_edge.trainers import VQAutoencoder

from compressionkit.layers import EmaResidualVectorQuantizer
from compressionkit.trainers.rate_distortion import RateDistortionVQAutoencoder

VOCAB = 16
EMBED = 8
T_LAT = 4
B = 3
FRAME = 16


def _tiny_encoder() -> keras.Model:
    """Encoder mapping (B, 1, FRAME, 1) → (B, 1, T_LAT, EMBED)."""
    inp = keras.Input(shape=(1, FRAME, 1))
    x = keras.layers.Conv2D(EMBED, kernel_size=(1, 4), strides=(1, 4), padding="valid", activation="relu")(inp)
    return keras.Model(inp, x, name="tiny_enc")


def _tiny_decoder() -> keras.Model:
    """Decoder mapping (B, 1, T_LAT, EMBED) → (B, 1, FRAME, 1)."""
    inp = keras.Input(shape=(1, T_LAT, EMBED))
    x = keras.layers.Conv2DTranspose(1, kernel_size=(1, 4), strides=(1, 4), padding="valid")(inp)
    return keras.Model(inp, x, name="tiny_dec")


def _tiny_prior() -> keras.Model:
    """Causal CNN prior over (B, T_LAT) tokens → (B, T_LAT, VOCAB)."""
    tokens = keras.Input(shape=(T_LAT,), dtype="int32")
    x = keras.layers.Embedding(VOCAB, 8)(tokens)
    x = keras.layers.Conv1D(8, kernel_size=3, padding="causal", activation="relu")(x)
    logits = keras.layers.Dense(VOCAB)(x)
    return keras.Model(tokens, logits, name="tiny_prior")


def _build_ae() -> VQAutoencoder:
    enc = _tiny_encoder()
    vq = EmaResidualVectorQuantizer(
        num_levels=1,
        num_embeddings=VOCAB,
        embedding_dim=EMBED,
        beta=0.25,
    )
    dec = _tiny_decoder()
    return VQAutoencoder(encoder=enc, vq=vq, decoder=dec)


def test_rd_wrapper_forward_and_loss_shape() -> None:
    ae = _build_ae()
    prior = _tiny_prior()
    rd = RateDistortionVQAutoencoder(
        autoencoder=ae,
        prior=prior,
        rate_weight=0.01,
    )
    # Initialise the inner ae's reconstruction loss via .compile so
    # rd.compute_loss can call ae.compute_loss without crashing.
    ae.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())

    x = np.random.randn(B, 1, FRAME, 1).astype(np.float32)
    y_pred = rd(x, training=True)
    assert y_pred.shape == (B, 1, FRAME, 1)
    assert rd._latent_z.shape == (B, 1, T_LAT, EMBED)
    assert rd._indices_flat.shape[0] == B * T_LAT

    loss = rd.compute_loss(x=x, y=x, y_pred=y_pred)
    assert np.isfinite(float(loss))
    assert float(rd._rate_tracker.result()) > 0  # bits/token always positive


def test_rd_wrapper_rate_term_finite_and_differentiable() -> None:
    """Ensure the rate term contributes a finite, non-zero gradient to the
    encoder when soft assignment is engaged."""
    import tensorflow as tf

    ae = _build_ae()
    prior = _tiny_prior()
    rd = RateDistortionVQAutoencoder(
        autoencoder=ae,
        prior=prior,
        rate_weight=0.05,
    )
    ae.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())

    x = tf.constant(np.random.randn(B, 1, FRAME, 1).astype(np.float32))
    enc_vars = ae.encoder.trainable_variables
    assert enc_vars, "encoder should have trainable variables"

    with tf.GradientTape() as tape:
        y_pred = rd(x, training=True)
        loss = rd.compute_loss(x=x, y=x, y_pred=y_pred)
    grads = tape.gradient(loss, enc_vars)
    assert all(g is not None for g in grads), "encoder should receive gradient"
    assert any(np.isfinite(np.asarray(g)).all() and np.abs(np.asarray(g)).sum() > 0 for g in grads), (
        "encoder gradient should be non-zero"
    )


def test_rd_wrapper_rejects_multilevel_rvq() -> None:
    enc = _tiny_encoder()
    vq = EmaResidualVectorQuantizer(
        num_levels=2,
        num_embeddings=VOCAB,
        embedding_dim=EMBED,
        beta=0.25,
    )
    dec = _tiny_decoder()
    ae = VQAutoencoder(encoder=enc, vq=vq, decoder=dec)
    rd = RateDistortionVQAutoencoder(autoencoder=ae, prior=_tiny_prior())
    x = np.random.randn(B, 1, FRAME, 1).astype(np.float32)
    with pytest.raises(ValueError, match="num_levels==1"):
        rd(x, training=True)
