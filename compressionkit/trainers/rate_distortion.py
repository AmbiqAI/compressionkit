"""Joint rate-distortion training wrapper for VQ/RVQ autoencoders.

This module provides a thin wrapper that turns an existing VQ autoencoder
(encoder → quantizer → decoder) plus a token-level prior into a single
trainable model whose loss is

    L = D + λ · R

where ``D`` is the standard reconstruction distortion (MSE + auxiliary
losses already configured on the wrapped autoencoder) and ``R`` is the
expected number of bits per token under the prior.

Design notes (per AGENTS.md):

* ``num_levels == 1`` only.  Multi-level RVQ would require interleaving
  per-level token streams and per-level priors; we keep the prototype
  scoped to the deployed 32× / 64× golden configurations.
* The encoder learns to produce *predictable* tokens via a soft
  assignment over codebook entries.  Distance is taken as the squared
  Euclidean distance between the encoder output and each codebook
  centroid; a temperature-scaled softmax over ``-d`` yields
  ``s_k(z) = p(c=k | z)`` which is differentiable through ``z``.  The
  rate at each position is the cross-entropy of the soft assignment
  against the prior's predicted log-probabilities — this matches the
  EnCodec / SoundStream R-D objective.
* Hard tokens (argmin distance) are still used both for the autoregressive
  prior context and for the decoder input via straight-through estimation,
  so the *deployment behaviour* is unchanged.
* The prior is treated as a black-box ``keras.Model`` mapping
  ``(B, ctx_len) int32 → (B, ctx_len, K) float logits``.  Both the
  dilated-CNN prior used today and the transformer prior in
  ``compressionkit.generative`` satisfy that interface.

Promotion target: this wrapper is general (no ECG/PPG specifics) and is a
candidate to move into HeliaEdge as ``RateDistortionVQAutoencoder``
once the experiment validates the design.
"""

from __future__ import annotations

import math
from typing import Any

import keras
import keras.ops as ops

_LN2 = math.log(2.0)


class RateDistortionVQAutoencoder(keras.Model):
    """Wrap a VQ/RVQ autoencoder + token prior with joint R-D training.

    Args:
        autoencoder: Pretrained or freshly initialised ``VQAutoencoder``
            (or any model exposing ``.encoder``, ``.vq``, ``.decoder``).
            Its existing reconstruction loss / extra losses are reused
            verbatim.
        prior: ``keras.Model`` mapping ``(B, T) int32`` tokens to
            ``(B, T, K) float`` logits.  Receptive field / context
            length is the prior's responsibility.
        rate_weight: Scalar λ — the weight applied to the rate term
            (in bits/token) when summed into the total loss.
        soft_assignment_temperature: Temperature τ used when computing
            soft codebook assignments.  Smaller τ → harder assignment
            (closer to the deployed argmin) but also smaller gradients;
            larger τ → smoother gradients but a coarser approximation
            of the deployed quantizer.  Default 1.0 (matches EnCodec).
        freeze_prior: If True, gradients are not propagated into the
            prior weights — useful for validating that the encoder side
            of the joint loss alone closes the gap.

    Notes:
        - Only ``num_levels == 1`` is supported.  We assert this at
          construction time by inspecting ``autoencoder.vq``.
        - The wrapper assumes a 1-D latent layout ``(B, 1, T_lat, D)``
          (the ECG raw-signal model).  STFT 2-D mode is out of scope
          for this prototype; the same construction generalises trivially
          but bookkeeping for the prior context order would need extra
          care.
    """

    def __init__(
        self,
        autoencoder: keras.Model,
        prior: keras.Model,
        *,
        rate_weight: float = 0.01,
        soft_assignment_temperature: float = 1.0,
        freeze_prior: bool = False,
        name: str = "rd_vq_autoencoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.autoencoder = autoencoder
        self.encoder = autoencoder.encoder
        self.vq = autoencoder.vq
        self.decoder = autoencoder.decoder
        self.prior = prior
        self.rate_weight = float(rate_weight)
        self.soft_temperature = float(soft_assignment_temperature)
        self.freeze_prior = bool(freeze_prior)

        if self.freeze_prior:
            self.prior.trainable = False

        # Codebook + dim metadata are looked up lazily on first call —
        # they're only populated after the vq layer's ``build`` runs.
        # Stored via ``object.__setattr__`` so Keras doesn't try to track
        # them as state (which would fail post-build).
        object.__setattr__(self, "_codebook", None)
        object.__setattr__(self, "_vocab_size", 0)
        object.__setattr__(self, "_embedding_dim", 0)

        # Metric trackers
        self._distortion_tracker = keras.metrics.Mean(name="distortion")
        self._rate_tracker = keras.metrics.Mean(name="rate_bits")
        self._total_loss_tracker = keras.metrics.Mean(name="loss")

        # Eager resolve if vq is already built (common case: compressor
        # was loaded from a checkpoint and called once).
        if getattr(self.vq, "_codebooks", None):
            self._resolve_codebook()

    def _resolve_codebook(self) -> None:
        """Validate num_levels==1 and cache codebook/vocab metadata."""
        codebooks = getattr(self.vq, "_codebooks", None)
        if codebooks is None or len(codebooks) != 1:
            raise ValueError(
                "RateDistortionVQAutoencoder currently supports num_levels==1 only; "
                f"got vq with {0 if codebooks is None else len(codebooks)} codebooks."
            )
        object.__setattr__(self, "_codebook", codebooks[0])
        object.__setattr__(
            self,
            "_vocab_size",
            int(getattr(self.vq, "Ks", [None])[0] or 0),
        )
        object.__setattr__(self, "_embedding_dim", int(getattr(self.vq, "D", 0)))
        if self._vocab_size == 0 or self._embedding_dim == 0:
            raise ValueError("Could not infer vocab_size / embedding_dim from vq layer.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def call(self, x, training=False):
        """Standard autoencoder forward — same output shape as the wrapped AE.

        The rate-term computation is deferred to ``compute_loss`` and
        relies on intermediates stashed during this call.
        """
        z = self.encoder(x, training=training)
        zq, indices_list = self.vq(z, training=training, return_indices=True)
        y = self.decoder(zq, training=training)
        if self._codebook is None:
            # First successful forward → vq layer is built; cache metadata.
            self._resolve_codebook()
        # Stash for compute_loss — these are tensors from the same graph.
        self._latent_z = z
        self._indices_flat = indices_list[0]  # (B*T_lat,) int32
        return y

    # ------------------------------------------------------------------
    # Soft assignment helper
    # ------------------------------------------------------------------

    def _soft_probs_from_z(self, z) -> Any:
        """Compute per-position softmax over codebook distances.

        Args:
            z: Encoder output, shape ``(B, 1, T_lat, D)``.

        Returns:
            Tensor of shape ``(B, T_lat, K)``.  Probabilities sum to 1
            across the last axis and are differentiable in ``z``.
        """
        B = ops.shape(z)[0]
        T = ops.shape(z)[2]
        D = self._embedding_dim
        K = self._vocab_size
        tau = self.soft_temperature

        flat = ops.reshape(z, (-1, D))  # (N, D)
        codebook = self._codebook  # (K, D)
        # Squared L2 distance: ||z||^2 + ||e||^2 - 2 z·e
        z2 = ops.sum(flat * flat, axis=1, keepdims=True)  # (N, 1)
        c2 = ops.sum(codebook * codebook, axis=1)  # (K,)
        c2 = ops.reshape(c2, (1, -1))  # (1, K)
        sim = ops.matmul(flat, ops.transpose(codebook))  # (N, K)
        dist = z2 + c2 - 2.0 * sim  # (N, K)
        logits = -dist / tau  # smaller dist → higher logit
        probs = ops.softmax(logits, axis=-1)  # (N, K)
        return ops.reshape(probs, (B, T, K))

    def _indices_2d(self, indices_flat, B, T) -> Any:
        """Reshape flat (B*T,) indices to (B, T) int32."""
        return ops.cast(ops.reshape(indices_flat, (B, T)), "int32")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        allow_empty=False,
    ):
        # Distortion + auxiliary losses come from the wrapped autoencoder.
        distortion = self.autoencoder.compute_loss(
            x=x,
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            allow_empty=True,
        )

        # ------------------------------------------------------------------
        # Rate term: NLL of *hard* tokens under the prior — this is the
        # rate an arithmetic coder actually pays.  Soft assignment NLL is
        # used only as the differentiable surrogate that carries the
        # gradient back to ``z`` (encoder); the forward value matches the
        # hard rate exactly via straight-through.
        # ------------------------------------------------------------------
        z = self._latent_z
        indices_flat = self._indices_flat
        B = ops.shape(z)[0]
        T = ops.shape(z)[2]

        # Hard tokens for autoregressive context, shape (B, T) int32.
        idx2d = self._indices_2d(indices_flat, B, T)

        # The prior expects to predict c_t given c_{<t}.
        # We feed the same (B, T) tokens and let the prior's causal
        # masking ensure position t only attends to <t.  This matches
        # how both the dilated-causal CNN and the causal transformer
        # are trained today (see scripts/train_rvq_prior.py and
        # scripts/measure_rvq_entropy.py).
        prior_logits = self.prior(idx2d, training=True)  # (B, T, K)
        log_probs = ops.log_softmax(prior_logits, axis=-1)  # (B, T, K)

        # Soft assignment over current emission, (B, T, K).
        soft = self._soft_probs_from_z(z)

        # Differentiable surrogate (carries gradient to encoder):
        soft_nll = -ops.sum(soft * log_probs, axis=-1)  # (B, T) nats
        # Hard NLL — the rate an arithmetic coder pays, matches
        # ``measure_rvq_entropy.py`` evaluation:
        hard_nll = -ops.take_along_axis(log_probs, ops.expand_dims(idx2d, -1), axis=-1)  # (B, T, 1)
        hard_nll = ops.squeeze(hard_nll, axis=-1)  # (B, T)
        # Straight-through: forward = hard_nll, backward via soft_nll.
        nll_per_pos = soft_nll + ops.stop_gradient(hard_nll - soft_nll)
        rate_bits = ops.mean(nll_per_pos) / _LN2  # scalar bits/token

        # Update trackers — distortion and rate separately for clarity.
        self._distortion_tracker.update_state(distortion)
        self._rate_tracker.update_state(rate_bits)
        total = distortion + self.rate_weight * rate_bits
        self._total_loss_tracker.update_state(total)
        return total

    # ------------------------------------------------------------------
    # Metric plumbing
    # ------------------------------------------------------------------

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        results = self.autoencoder.compute_metrics(x, y, y_pred, sample_weight)
        results.setdefault(self._distortion_tracker.name, self._distortion_tracker.result())
        results.setdefault(self._rate_tracker.name, self._rate_tracker.result())
        results.setdefault(self._total_loss_tracker.name, self._total_loss_tracker.result())
        return results

    @property
    def metrics(self):
        return [
            *self.autoencoder.metrics,
            self._distortion_tracker,
            self._rate_tracker,
            self._total_loss_tracker,
        ]


__all__ = ["RateDistortionVQAutoencoder"]
