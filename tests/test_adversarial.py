"""Tests for the multi-scale discriminator and adversarial training wrapper."""

from __future__ import annotations

import keras
import numpy as np
from helia_edge.trainers import VQAutoencoder

from compressionkit.layers import EmaResidualVectorQuantizer, MultiScaleDiscriminator
from compressionkit.trainers.adversarial import AdversarialVQAutoencoder, SetEpochCallback

VOCAB = 16
EMBED = 8
T_LAT = 4
B = 2
FRAME = 64  # must be divisible by sub-disc total stride (4^3=64)


# ---- Tiny model factories ------------------------------------------------


def _tiny_encoder() -> keras.Model:
    inp = keras.Input(shape=(1, FRAME, 1))
    x = keras.layers.Conv2D(
        EMBED,
        kernel_size=(1, FRAME // T_LAT),
        strides=(1, FRAME // T_LAT),
        padding="valid",
        activation="relu",
    )(inp)
    return keras.Model(inp, x, name="tiny_enc")


def _tiny_decoder() -> keras.Model:
    inp = keras.Input(shape=(1, T_LAT, EMBED))
    x = keras.layers.Conv2DTranspose(
        1,
        kernel_size=(1, FRAME // T_LAT),
        strides=(1, FRAME // T_LAT),
        padding="valid",
    )(inp)
    return keras.Model(inp, x, name="tiny_dec")


def _build_ae() -> VQAutoencoder:
    enc = _tiny_encoder()
    vq = EmaResidualVectorQuantizer(
        num_levels=1,
        num_embeddings=VOCAB,
        embedding_dim=EMBED,
        beta=0.25,
    )
    dec = _tiny_decoder()
    ae = VQAutoencoder(encoder=enc, vq=vq, decoder=dec)
    ae.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
    return ae


# ---- Discriminator tests --------------------------------------------------


class TestMultiScaleDiscriminator:
    def test_output_shapes(self):
        disc = MultiScaleDiscriminator(
            FRAME,
            num_scales=2,
            channels=(8, 16, 32, 64),
        )
        x = np.random.randn(B, 1, FRAME, 1).astype(np.float32)
        outputs = disc(x, training=False)

        assert len(outputs) == 2, "Should have 2 scale outputs"
        for scale_out in outputs:
            # Each scale returns [feat_0, feat_1, ..., logit_map]
            assert len(scale_out) == 5  # 4 feature maps + 1 logit
            logits = scale_out[-1]
            assert logits.shape[0] == B
            assert logits.shape[1] == 1
            assert logits.shape[-1] == 1  # single-channel logit

    def test_three_scales(self):
        disc = MultiScaleDiscriminator(
            FRAME,
            num_scales=3,
            channels=(8, 16, 32, 64),
        )
        x = np.random.randn(B, 1, FRAME, 1).astype(np.float32)
        outputs = disc(x, training=False)
        assert len(outputs) == 3

    def test_trainable_params_nonzero(self):
        disc = MultiScaleDiscriminator(
            FRAME,
            num_scales=2,
            channels=(8, 16, 32, 64),
        )
        x = np.random.randn(1, 1, FRAME, 1).astype(np.float32)
        disc(x)  # build
        assert sum(np.prod(v.shape) for v in disc.trainable_variables) > 0

    def test_serialization_roundtrip(self):
        disc = MultiScaleDiscriminator(
            FRAME,
            num_scales=2,
            channels=(8, 16),
            kernel_width=7,
            final_kernel=3,
        )
        config = disc.get_config()
        disc2 = MultiScaleDiscriminator.from_config(config)
        assert disc2._num_scales == 2
        assert disc2._channels == (8, 16)
        assert disc2._kernel_width == 7


# ---- Adversarial wrapper tests --------------------------------------------


class TestAdversarialVQAutoencoder:
    def test_forward_matches_autoencoder(self):
        ae = _build_ae()
        disc = MultiScaleDiscriminator(
            FRAME,
            num_scales=2,
            channels=(4, 8, 16, 32),
        )
        adv = AdversarialVQAutoencoder(
            ae,
            disc,
            adv_weight=1.0,
            feat_weight=10.0,
        )
        x = np.random.randn(B, 1, FRAME, 1).astype(np.float32)
        y_adv = adv(x, training=False)
        assert y_adv.shape == (B, 1, FRAME, 1)

    def test_hinge_losses_finite(self):
        disc = MultiScaleDiscriminator(
            FRAME,
            num_scales=2,
            channels=(4, 8, 16, 32),
        )
        real = np.random.randn(B, 1, FRAME, 1).astype(np.float32)
        fake = np.random.randn(B, 1, FRAME, 1).astype(np.float32)

        disc_real_out = disc(real, training=False)
        disc_fake_out = disc(fake, training=False)

        d_loss = AdversarialVQAutoencoder._hinge_disc_loss(
            disc_real_out,
            disc_fake_out,
        )
        g_loss = AdversarialVQAutoencoder._hinge_gen_loss(disc_fake_out)
        fm_loss = AdversarialVQAutoencoder._feature_matching_loss(
            disc_real_out,
            disc_fake_out,
        )

        assert np.isfinite(float(d_loss))
        assert np.isfinite(float(g_loss))
        assert np.isfinite(float(fm_loss))
        assert float(fm_loss) >= 0

    def test_train_step_runs(self):
        ae = _build_ae()
        disc = MultiScaleDiscriminator(
            FRAME,
            num_scales=2,
            channels=(4, 8, 16, 32),
        )
        adv = AdversarialVQAutoencoder(
            ae,
            disc,
            adv_weight=0.1,
            feat_weight=1.0,
        )
        adv.compile(
            gen_optimizer=keras.optimizers.Adam(1e-4),
            disc_optimizer=keras.optimizers.Adam(1e-4),
        )

        x = np.random.randn(B, 1, FRAME, 1).astype(np.float32)
        logs = adv.train_step((x, x))

        assert "gen_loss" in logs
        assert "disc_loss" in logs
        assert "feat_loss" in logs
        assert "adv_g_loss" in logs
        assert np.isfinite(float(logs["gen_loss"]))
        assert np.isfinite(float(logs["disc_loss"]))

    def test_disc_start_epoch_delays_adversarial(self):
        ae = _build_ae()
        disc = MultiScaleDiscriminator(
            FRAME,
            num_scales=2,
            channels=(4, 8, 16, 32),
        )
        adv = AdversarialVQAutoencoder(
            ae,
            disc,
            adv_weight=1.0,
            feat_weight=1.0,
            disc_start_epoch=5,
        )
        adv.compile(
            gen_optimizer=keras.optimizers.Adam(1e-4),
            disc_optimizer=keras.optimizers.Adam(1e-4),
        )

        x = np.random.randn(B, 1, FRAME, 1).astype(np.float32)

        # Epoch 0 — disc should NOT be active
        adv.set_epoch(0)
        logs = adv.train_step((x, x))
        assert float(logs["adv_g_loss"]) == 0.0
        assert float(logs["feat_loss"]) == 0.0

        # Epoch 5 — disc should be active
        adv.set_epoch(5)
        logs = adv.train_step((x, x))
        # adv_g_loss should be non-zero now
        assert float(logs["disc_loss"]) != 0.0

    def test_set_epoch_callback(self):
        ae = _build_ae()
        disc = MultiScaleDiscriminator(
            FRAME,
            num_scales=2,
            channels=(4, 8, 16, 32),
        )
        adv = AdversarialVQAutoencoder(ae, disc)
        cb = SetEpochCallback()
        cb.set_model(adv)
        cb.on_epoch_begin(7)
        assert adv._current_epoch == 7

    def test_test_step_runs(self):
        ae = _build_ae()
        disc = MultiScaleDiscriminator(
            FRAME,
            num_scales=2,
            channels=(4, 8, 16, 32),
        )
        adv = AdversarialVQAutoencoder(ae, disc)
        adv.compile(
            gen_optimizer=keras.optimizers.Adam(1e-4),
            disc_optimizer=keras.optimizers.Adam(1e-4),
        )
        x = np.random.randn(B, 1, FRAME, 1).astype(np.float32)
        logs = adv.test_step((x, x))
        assert "recon_loss" in logs
        assert np.isfinite(float(logs["recon_loss"]))
