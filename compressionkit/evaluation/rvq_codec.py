"""RVQ codec adapter for the evaluation harness.

Wraps a trained RVQ autoencoder (Keras model) behind the
:class:`compressionkit.evaluation.codec.Codec` protocol. The actual
deployment bitstream consists of token IDs (one per RVQ level per latent
position); we surface that in :attr:`EncodedFrame.payload` and report
``nbits`` from the per-level codebook sizes::

    nbits_per_token = sum_l ceil(log2(K_l))
    nbits           = nbits_per_token * num_latent_tokens

The adapter additionally populates :attr:`EncodedFrame.side` with
quantities the QoS module (PR 4) consumes:

    side["token_ids"]        — list of (num_tokens,) int arrays, one per level
    side["residual_norms"]   — list of length L (latent norm after each stage)
    side["quant_distances"]  — per-token, per-level L2 distance between the
                                stage residual and the chosen codeword

Single-channel (PPG) and multi-channel (12-lead ECG) trained models are
both supported. STFT-domain (2-D spatial) models are not handled here yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from compressionkit.evaluation.codec import EncodedFrame

__all__ = ["RvqCodec"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bits_per_index(codebook_sizes: list[int]) -> int:
    """Sum of ceil(log2(K_l)) across RVQ levels."""
    return sum(int(math.ceil(math.log2(max(2, k)))) for k in codebook_sizes)


def _load_config(run_dir: Path, modality: str):
    """Load the Pydantic config that was used to train *run_dir*."""
    if modality == "ecg":
        from compressionkit.configs.ecg_rvq import EcgRvqConfig

        cfg = EcgRvqConfig.model_validate_json((run_dir / "config.json").read_text())
    elif modality == "ppg":
        from compressionkit.configs.ppg_rvq import PpgRvqConfig

        cfg = PpgRvqConfig.model_validate_json((run_dir / "config.json").read_text())
    else:
        raise ValueError(f"Unknown modality {modality!r}; expected 'ecg' or 'ppg'")
    return cfg


def _build_model(cfg, modality: str):
    if modality == "ecg":
        from compressionkit.trainers.ecg_rvq import build_model
    else:
        from compressionkit.trainers.ppg_rvq import build_model
    return build_model(cfg)


# ---------------------------------------------------------------------------
# RvqCodec
# ---------------------------------------------------------------------------


@dataclass
class RvqCodec:
    """Codec adapter around a trained RVQ autoencoder.

    Construct via :meth:`from_run_dir` to load weights from a golden run
    directory. The constructor itself accepts a pre-built model for tests.

    Attributes:
        name: Human-readable codec name (defaults to the run directory's name).
        modality: ``"ppg"`` or ``"ecg"``.
        sample_rate: Frame sample rate.
        frame_size: Samples per frame.
        target_cr: Nominal compression ratio (informational).
        model: The trained Keras model with ``.encoder``, ``.vq``, ``.decoder``.
        n_channels: Number of input/output channels (1 for single-lead PPG/ECG,
            12 for 12-lead ECG).
    """

    name: str
    modality: str
    sample_rate: int
    frame_size: int
    target_cr: float
    model: Any  # keras.Model
    n_channels: int = 1
    bits_per_sample: int = 16

    # Cached after first encode
    _codebook_sizes: list[int] = field(default_factory=list, init=False, repr=False)
    _codebooks: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _latent_shape: tuple[int, ...] | None = field(default=None, init=False, repr=False)
    _embedding_dim: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path | str,
        *,
        modality: str,
        target_cr: float | None = None,
        name: str | None = None,
        weights_filename: str = "best_model.weights.h5",
    ) -> RvqCodec:
        """Load a trained RVQ model from a run directory.

        Args:
            run_dir: Directory containing ``config.json`` and the weights file.
            modality: ``"ppg"`` or ``"ecg"``.
            target_cr: Nominal CR (informational). If ``None`` it is derived
                from the data + model config (raw_bits / sum(log2 K_l) * tokens).
            name: Override the codec name (defaults to run-dir basename).
            weights_filename: Override the weights file name.
        """
        import tensorflow as tf  # local import — TF is heavy

        run_dir = Path(run_dir)
        cfg = _load_config(run_dir, modality)
        n_channels = int(cfg.data.num_leads) if modality == "ecg" else 1
        sample_rate = int(cfg.data.effective_sample_rate) if modality == "ecg" else int(cfg.data.sampling_rate)
        frame_size = int(cfg.data.frame_size)

        dummy = np.zeros((1, 1, frame_size, n_channels), dtype=np.float32)
        with tf.device("/CPU:0"):
            model = _build_model(cfg, modality)
            model(dummy, training=False)

        weights_path = run_dir / weights_filename
        if not weights_path.exists():
            raise FileNotFoundError(f"No weights at {weights_path}")
        model.load_weights(str(weights_path))

        codec = cls(
            name=name or run_dir.name,
            modality=modality,
            sample_rate=sample_rate,
            frame_size=frame_size,
            target_cr=float(target_cr) if target_cr is not None else 0.0,
            model=model,
            n_channels=n_channels,
        )
        # Prime caches by running a dummy encode
        codec._prime_caches(dummy)
        if codec.target_cr == 0.0:
            raw_bits = codec.frame_size * codec.bits_per_sample * codec.n_channels
            n_tokens = int(np.prod(codec._latent_shape[1:-1])) if codec._latent_shape else 0
            enc_bits = _bits_per_index(codec._codebook_sizes) * n_tokens
            codec.target_cr = raw_bits / enc_bits if enc_bits > 0 else math.inf
        return codec

    def _prime_caches(self, dummy_input: np.ndarray) -> None:
        """Run encoder once to learn latent shape and snapshot codebooks."""
        import tensorflow as tf

        with tf.device("/CPU:0"):
            z = self.model.encoder(dummy_input, training=False)
        z_np = np.asarray(z)
        self._latent_shape = tuple(z_np.shape)
        self._embedding_dim = int(z_np.shape[-1])
        # Snapshot codebooks (avoid re-fetching every encode/decode)
        vq = self.model.vq
        self._codebook_sizes = [int(k) for k in vq.Ks]
        self._codebooks = [np.asarray(cb) for cb in vq._codebooks]

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def _shape_for_model(self, frame: np.ndarray) -> np.ndarray:
        """Reshape input to ``(1, 1, frame_size, n_channels)``."""
        arr = np.asarray(frame, dtype=np.float32)
        if self.n_channels == 1:
            if arr.ndim == 1:
                pass
            elif arr.ndim == 2 and arr.shape[-1] == 1:
                arr = arr[:, 0]
            else:
                raise ValueError(f"Single-channel codec got input shape {arr.shape}; expected ({self.frame_size},)")
            if arr.shape[0] != self.frame_size:
                raise ValueError(f"Frame length {arr.shape[0]} != {self.frame_size}")
            return arr.reshape(1, 1, self.frame_size, 1)
        # Multi-channel
        if arr.ndim != 2 or arr.shape != (self.frame_size, self.n_channels):
            raise ValueError(
                f"Multi-channel codec got input shape {arr.shape}; expected ({self.frame_size}, {self.n_channels})"
            )
        return arr.reshape(1, 1, self.frame_size, self.n_channels)

    def encode(self, frame: np.ndarray) -> EncodedFrame:
        import tensorflow as tf

        x = self._shape_for_model(frame)
        with tf.device("/CPU:0"):
            z = self.model.encoder(x, training=False)
        z_np = np.asarray(z, dtype=np.float32)

        # Snapshot caches if not yet primed
        if not self._codebook_sizes:
            self._latent_shape = tuple(z_np.shape)
            self._embedding_dim = int(z_np.shape[-1])
            vq = self.model.vq
            self._codebook_sizes = [int(k) for k in vq.Ks]
            self._codebooks = [np.asarray(cb) for cb in vq._codebooks]

        # Run RVQ purely numerically: greedy nearest-neighbor per stage.
        # This avoids depending on the layer's training-time branches.
        flat = z_np.reshape(-1, self._embedding_dim)  # (T, D)
        n_tokens = int(flat.shape[0])
        token_ids: list[np.ndarray] = []
        quant_distances: list[np.ndarray] = []
        residual_norms: list[float] = [float(np.linalg.norm(flat))]

        residual = flat.copy()
        for cb in self._codebooks:
            # cb: (K, D); residual: (T, D)
            dists = (
                np.sum(residual**2, axis=1, keepdims=True)
                - 2.0 * residual @ cb.T
                + np.sum(cb**2, axis=1, keepdims=True).T
            )
            idx = np.argmin(dists, axis=1).astype(np.int32)
            chosen = cb[idx]  # (T, D)
            # Per-token L2 distance between residual and chosen codeword
            per_token_dist = np.linalg.norm(residual - chosen, axis=1).astype(np.float32)
            quant_distances.append(per_token_dist)
            token_ids.append(idx)
            residual = residual - chosen
            residual_norms.append(float(np.linalg.norm(residual)))

        nbits = _bits_per_index(self._codebook_sizes) * n_tokens

        return EncodedFrame(
            payload=token_ids,
            nbits=nbits,
            side={
                "token_ids": token_ids,
                "quant_distances": quant_distances,
                "residual_norms": residual_norms,
                "latent_shape": self._latent_shape,
                "n_tokens": n_tokens,
                "bits_per_token": _bits_per_index(self._codebook_sizes),
                "codebook_sizes": list(self._codebook_sizes),
            },
        )

    def decode(self, encoded: EncodedFrame) -> np.ndarray:
        import tensorflow as tf

        token_ids = encoded.side.get("token_ids") or encoded.payload
        latent_shape = encoded.side.get("latent_shape", self._latent_shape)
        if latent_shape is None:
            raise ValueError("decode requires latent_shape; encode at least one frame first")
        if not self._codebooks:
            raise RuntimeError("Codec caches not primed; call encode() first")

        # Reconstruct quantized latent: sum gathered codebook vectors
        flat = np.zeros((int(np.prod(latent_shape[:-1])), self._embedding_dim), dtype=np.float32)
        for cb, idx in zip(self._codebooks, token_ids):
            flat = flat + cb[idx]
        zq = flat.reshape(latent_shape).astype(np.float32)

        with tf.device("/CPU:0"):
            y = self.model.decoder(zq, training=False)
        y_np = np.asarray(y, dtype=np.float32)
        # y: (1, 1, frame_size, n_channels) → drop batch+spatial dims
        y_np = y_np[0, 0]
        if self.n_channels == 1:
            return y_np[:, 0]
        return y_np
