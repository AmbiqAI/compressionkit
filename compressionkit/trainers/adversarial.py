"""Adversarial training wrapper for VQ/RVQ autoencoders.

Wraps an existing ``VQAutoencoder`` and a ``MultiScaleDiscriminator`` into a
single trainable unit with alternating generator/discriminator updates.

The total generator loss is::

    L_G = D + λ_adv · L_adv + λ_feat · L_feat

where ``D`` is the original reconstruction + VQ losses from the wrapped
autoencoder, ``L_adv`` is the generator-side adversarial loss (hinge), and
``L_feat`` is the L1 feature-matching loss across discriminator layers and
scales.

The discriminator is trained with hinge loss::

    L_D = mean(ReLU(1 - D(real))) + mean(ReLU(1 + D(fake)))

Design notes (per AGENTS.md):

* The discriminator is **training-only** — it is discarded at deployment,
  adding zero inference cost.
* Hinge loss chosen over LSGAN/WGAN-GP for simplicity and stability; this
  is what SoundStream and DAC use.
* Feature-matching loss stabilises early training and prevents the generator
  from producing adversarial artifacts.
* ``keras.Model`` subclass with custom ``train_step`` for alternating
  updates — follows the Keras 3 GAN pattern.
"""

from __future__ import annotations

from typing import Any

import keras
import keras.ops as ops


class AdversarialVQAutoencoder(keras.Model):
    """Wrap a VQ autoencoder + discriminator for adversarial training.

    Args:
        autoencoder: Pretrained or fresh ``VQAutoencoder`` (must already
            be compiled with reconstruction loss / extra losses).
        discriminator: A ``MultiScaleDiscriminator`` instance (or any model
            returning ``list[list[Tensor]]`` where each inner list ends
            with a logit map).
        adv_weight: Scalar weight for the adversarial (hinge) generator loss.
        feat_weight: Scalar weight for the L1 feature-matching loss.
        disc_start_epoch: Epoch at which the discriminator begins training.
            Before this epoch the generator trains with reconstruction loss
            only, giving the VQ autoencoder time to stabilise before the
            adversarial signal arrives.  Default 0 (immediate).
    """

    def __init__(
        self,
        autoencoder: keras.Model,
        discriminator: keras.Model,
        *,
        adv_weight: float = 1.0,
        feat_weight: float = 10.0,
        disc_start_epoch: int = 0,
        name: str = "adv_vq_autoencoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.autoencoder = autoencoder
        self.encoder = autoencoder.encoder
        self.vq = autoencoder.vq
        self.decoder = autoencoder.decoder
        self.discriminator = discriminator
        self.adv_weight = float(adv_weight)
        self.feat_weight = float(feat_weight)
        self.disc_start_epoch = int(disc_start_epoch)
        self._current_epoch = 0

        # Float mask (0.0 or 1.0) that gates the adversarial signal to
        # the generator.  Using a keras.Variable avoids the XLA
        # cross-device issue that tf.Variable + tf.cond triggers.
        # The discriminator always trains (pre-warming it during the
        # warmup phase is beneficial), but the generator ignores its
        # signal until the mask flips to 1.0.
        initial_mask = 1.0 if disc_start_epoch == 0 else 0.0
        self._disc_mask = keras.Variable(
            initial_mask,
            trainable=False,
            dtype="float32",
            name="disc_mask",
        )

        # Metric trackers
        self._recon_tracker = keras.metrics.Mean(name="recon_loss")
        self._adv_g_tracker = keras.metrics.Mean(name="adv_g_loss")
        self._feat_tracker = keras.metrics.Mean(name="feat_loss")
        self._gen_total_tracker = keras.metrics.Mean(name="gen_loss")
        self._disc_loss_tracker = keras.metrics.Mean(name="disc_loss")
        self._disc_real_tracker = keras.metrics.Mean(name="disc_real")
        self._disc_fake_tracker = keras.metrics.Mean(name="disc_fake")

    def compile(
        self,
        gen_optimizer: keras.optimizers.Optimizer,
        disc_optimizer: keras.optimizers.Optimizer,
        **kwargs,
    ):
        """Compile with separate optimizers for generator and discriminator.

        The autoencoder must already be compiled (with its reconstruction
        loss etc.) before being passed to this wrapper.
        """
        super().compile(**kwargs)
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

    def call(self, x, training=False):
        """Standard autoencoder forward (no discriminator)."""
        return self.autoencoder(x, training=training)

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hinge_disc_loss(disc_real_outputs, disc_fake_outputs):
        """Discriminator hinge loss across all scales."""
        loss = ops.convert_to_tensor(0.0)
        n = 0
        for real_feats, fake_feats in zip(disc_real_outputs, disc_fake_outputs):
            # Last element is the logit map
            real_logits = real_feats[-1]
            fake_logits = fake_feats[-1]
            loss = loss + ops.mean(ops.relu(1.0 - real_logits))
            loss = loss + ops.mean(ops.relu(1.0 + fake_logits))
            n += 1
        return loss / max(n, 1)

    @staticmethod
    def _hinge_gen_loss(disc_fake_outputs):
        """Generator hinge loss (fool the discriminator)."""
        loss = ops.convert_to_tensor(0.0)
        n = 0
        for fake_feats in disc_fake_outputs:
            fake_logits = fake_feats[-1]
            loss = loss - ops.mean(fake_logits)
            n += 1
        return loss / max(n, 1)

    @staticmethod
    def _feature_matching_loss(disc_real_outputs, disc_fake_outputs):
        """L1 feature-matching loss across all scales and layers."""
        loss = ops.convert_to_tensor(0.0)
        n = 0
        for real_feats, fake_feats in zip(disc_real_outputs, disc_fake_outputs):
            # Exclude the final logit map (use intermediate features only)
            for rf, ff in zip(real_feats[:-1], fake_feats[:-1]):
                loss = loss + ops.mean(ops.abs(rf - ff))
                n += 1
        return loss / max(n, 1)

    # ------------------------------------------------------------------
    # Custom train step with alternating updates
    # ------------------------------------------------------------------

    def train_step(self, data):
        import tensorflow as tf

        x, y = data if isinstance(data, tuple) else (data, data)

        # ========================
        # 1) Discriminator update (always — pre-warms during warmup)
        # ========================
        y_fake_for_disc = ops.stop_gradient(self.autoencoder(x, training=True))

        with tf.GradientTape() as disc_tape:
            disc_real_out = self.discriminator(y, training=True)
            disc_fake_out = self.discriminator(
                y_fake_for_disc,
                training=True,
            )
            disc_loss = self._hinge_disc_loss(disc_real_out, disc_fake_out)

        disc_grads = disc_tape.gradient(
            disc_loss,
            self.discriminator.trainable_variables,
        )
        self.disc_optimizer.apply(
            disc_grads,
            self.discriminator.trainable_variables,
        )

        self._disc_loss_tracker.update_state(disc_loss)
        self._disc_real_tracker.update_state(
            ops.mean(disc_real_out[0][-1]),
        )
        self._disc_fake_tracker.update_state(
            ops.mean(disc_fake_out[0][-1]),
        )

        # ========================
        # 2) Generator update
        # ========================
        with tf.GradientTape() as gen_tape:
            y_fake_g = self.autoencoder(x, training=True)
            recon_loss = self.autoencoder.compute_loss(
                x=x,
                y=y,
                y_pred=y_fake_g,
            )

            disc_fake_g = self.discriminator(y_fake_g, training=False)
            disc_real_g = self.discriminator(y, training=False)
            adv_g_loss = self._hinge_gen_loss(disc_fake_g)
            feat_loss = self._feature_matching_loss(disc_real_g, disc_fake_g)

            # _disc_mask is 0.0 during warmup, 1.0 after disc_start_epoch
            mask = ops.convert_to_tensor(self._disc_mask)
            gen_total = recon_loss + mask * self.adv_weight * adv_g_loss + mask * self.feat_weight * feat_loss

        gen_grads = gen_tape.gradient(
            gen_total,
            self.autoencoder.trainable_variables,
        )
        self.gen_optimizer.apply(
            gen_grads,
            self.autoencoder.trainable_variables,
        )

        # Update trackers (report raw adv/feat, not masked)
        self._recon_tracker.update_state(recon_loss)
        self._adv_g_tracker.update_state(mask * adv_g_loss)
        self._feat_tracker.update_state(mask * feat_loss)
        self._gen_total_tracker.update_state(gen_total)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data if isinstance(data, tuple) else (data, data)
        y_pred = self.autoencoder(x, training=False)
        recon_loss = self.autoencoder.compute_loss(x=x, y=y, y_pred=y_pred)
        self._recon_tracker.update_state(recon_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self._recon_tracker,
            self._adv_g_tracker,
            self._feat_tracker,
            self._gen_total_tracker,
            self._disc_loss_tracker,
            self._disc_real_tracker,
            self._disc_fake_tracker,
        ]

    def set_epoch(self, epoch: int):
        """Called by the ``SetEpochCallback`` to update the current epoch."""
        self._current_epoch = epoch
        mask = 1.0 if epoch >= self.disc_start_epoch else 0.0
        self._disc_mask.assign(mask)


class SetEpochCallback(keras.callbacks.Callback):
    """Callback that updates the adversarial wrapper's epoch counter."""

    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, "set_epoch"):
            self.model.set_epoch(epoch)
