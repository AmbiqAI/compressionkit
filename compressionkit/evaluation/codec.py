"""Uniform codec interface for evaluation.

Defines a small Protocol so any codec (classical or neural) can be plugged
into the evaluation harness, scorecard runner, adversarial battery, etc.

Adapters wrap existing implementations without modifying them:

- :class:`IdentityCodec` — passthrough; useful for sanity checks.
- :class:`SpihtAcCodec` — bior4.4 DWT + SPIHT with optional arithmetic coding.
- (later) ``RvqCodec`` — wraps a trained Keras RVQ autoencoder.

The :class:`Codec` protocol is intentionally minimal:

    encoded = codec.encode(frame)
    recon   = codec.decode(encoded)
    bits    = encoded.nbits

Codecs may attach arbitrary side information to :attr:`EncodedFrame.side`
(e.g. token IDs, per-stage residual norms, quantization distances). Neural
codecs should populate this so the QoS module can compute confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from compressionkit.dsp.spiht import spiht_decode, spiht_encode
from compressionkit.dsp.wavelet import WaveletCoeffs, dwt_forward, dwt_inverse

__all__ = [
    "Codec",
    "EncodedFrame",
    "IdentityCodec",
    "SpihtAcCodec",
    "compression_ratio",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EncodedFrame:
    """Output of :meth:`Codec.encode`.

    Attributes:
        payload: Opaque encoded representation. Bytes for bitstream codecs;
            arbitrary array (e.g. token IDs) for neural codecs.
        nbits: Actual number of bits used for this frame's payload. This is
            what should be used to compute the *true* compression ratio.
        side: Optional side information attached by the codec. Common keys:
            ``"token_ids"``, ``"quant_distances"``, ``"residual_norms"``,
            ``"metadata"`` (e.g. SPIHT decode metadata).
    """

    payload: Any
    nbits: int
    side: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Codec(Protocol):
    """Uniform codec interface for evaluation."""

    name: str
    """Human-readable codec name (e.g. ``"spiht_ac"``, ``"rvq_64hz_04x"``)."""

    modality: str
    """``"ppg"`` or ``"ecg"``."""

    sample_rate: int
    """Frame sample rate in Hz."""

    frame_size: int
    """Number of samples per frame at the input layer."""

    target_cr: float
    """Nominal target compression ratio (informational; true CR comes from
    :attr:`EncodedFrame.nbits`)."""

    def encode(self, frame: np.ndarray) -> EncodedFrame:
        """Encode a single ``(frame_size,)`` or ``(frame_size, channels)`` frame."""
        ...

    def decode(self, encoded: EncodedFrame) -> np.ndarray:
        """Decode back to a frame with the same shape as the input."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compression_ratio(codec: Codec, encoded: EncodedFrame, bits_per_sample: int | None = None) -> float:
    """True compression ratio for a given encoded frame.

    Args:
        codec: The codec instance (for ``frame_size`` and, if available,
            ``bits_per_sample``).
        encoded: The encoded frame.
        bits_per_sample: Raw bit depth of the source signal. When *None*,
            falls back to ``codec.bits_per_sample`` if the codec exposes
            that attribute, otherwise to 16. Pass an explicit value to
            override.

    Returns:
        Raw bits / encoded bits. Returns ``inf`` when ``nbits == 0``.
    """
    if bits_per_sample is None:
        bits_per_sample = int(getattr(codec, "bits_per_sample", 16))
    raw_bits = codec.frame_size * bits_per_sample
    if encoded.nbits <= 0:
        return math.inf
    return raw_bits / encoded.nbits


# ---------------------------------------------------------------------------
# IdentityCodec — sanity check
# ---------------------------------------------------------------------------


@dataclass
class IdentityCodec:
    """Passthrough codec. Decoded output equals the input exactly."""

    name: str = "identity"
    modality: str = "ppg"
    sample_rate: int = 64
    frame_size: int = 320
    target_cr: float = 1.0

    def encode(self, frame: np.ndarray) -> EncodedFrame:
        arr = np.asarray(frame, dtype=np.float32)
        nbits = int(arr.size) * 32
        return EncodedFrame(payload=arr.copy(), nbits=nbits)

    def decode(self, encoded: EncodedFrame) -> np.ndarray:
        return np.asarray(encoded.payload, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# SpihtAcCodec — bior4.4 DWT + SPIHT (+ AC)
# ---------------------------------------------------------------------------


@dataclass
class SpihtAcCodec:
    """SPIHT-(AC) codec wrapping :mod:`compressionkit.dsp.spiht`.

    The bit budget per frame is derived from ``target_cr`` and ``frame_size``
    at construction time::

        max_bits = floor(frame_size * bits_per_sample / target_cr)

    Single-channel only. For multi-channel inputs, use a per-channel wrapper
    (the comparison code in :mod:`scripts.compare_12lead` shows the pattern).
    """

    name: str = "spiht_ac"
    modality: str = "ppg"
    sample_rate: int = 64
    frame_size: int = 320
    target_cr: float = 8.0
    wavelet: str = "bior4.4"
    levels: int = 6
    use_ac: bool = True
    bits_per_sample: int = 16

    def __post_init__(self) -> None:
        self.max_bits = int(self.frame_size * self.bits_per_sample / self.target_cr)
        if self.max_bits <= 0:
            raise ValueError(
                f"max_bits computed as {self.max_bits} (frame_size={self.frame_size}, "
                f"bits_per_sample={self.bits_per_sample}, target_cr={self.target_cr})"
            )

    def encode(self, frame: np.ndarray) -> EncodedFrame:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"SpihtAcCodec expects 1-D input, got shape {arr.shape}")
        coeffs = dwt_forward(arr, levels=self.levels, wavelet=self.wavelet)
        bitstream, meta = spiht_encode(
            coeffs.approx,
            coeffs.details,
            max_bits=self.max_bits,
            use_ac=self.use_ac,
        )
        # Use the encoder's exact bit count (``meta['n_bits']``) rather than
        # ``len(bitstream) * 8``: the byte buffer is padded up to a byte
        # boundary (and AC adds a flush tail), which would overstate the
        # real bitrate by up to 7-15 bits/frame.
        actual_nbits = int(meta.get("n_bits", len(bitstream) * 8))
        return EncodedFrame(
            payload=bitstream,
            nbits=actual_nbits,
            side={"metadata": meta},
        )

    def decode(self, encoded: EncodedFrame) -> np.ndarray:
        meta = encoded.side.get("metadata")
        if meta is None:
            raise ValueError("SpihtAcCodec.decode requires metadata in encoded.side")
        approx_r, details_r = spiht_decode(encoded.payload, meta)
        recon = dwt_inverse(
            WaveletCoeffs(approx=approx_r, details=details_r),
            wavelet=self.wavelet,
        )
        return recon[: self.frame_size].astype(np.float32)
