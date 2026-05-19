"""Common runtime contracts shared across codec families.

This module defines the minimal interface every codec runtime in
compressionKIT should expose so that downstream users can hydrate a
golden experiment, run round-trip encode/decode on a frame, and pull
sample stimulus data without caring whether the underlying method is
DSP-only (e.g. SPIHT), AI-only (e.g. RVQ), or hybrid.

Each codec family owns its own native API (RVQ uses index arrays,
SPIHT uses bitstreams + metadata). The :class:`Codec` protocol layered
on top is intentionally narrow:

* :meth:`Codec.compress` — frame in, opaque :class:`EncodedFrame` out.
* :meth:`Codec.decompress` — :class:`EncodedFrame` in, frame out.

Concrete codec classes (``RVQCodec``, ``SpihtCodec``, ...) may expose
additional family-specific methods alongside the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Codec",
    "EncodedFrame",
]


@dataclass
class EncodedFrame:
    """Opaque payload produced by :meth:`Codec.compress`.

    Attributes:
        payload: Family-specific encoded representation. Bytes for
            bitstream codecs; arbitrary array (e.g. index tensor) for
            neural codecs.
        nbits: Actual number of bits used for this frame. This is what
            should drive the *true* compression ratio.
        side: Optional side information attached by the codec (e.g.
            SPIHT decode metadata, RVQ index shape).
    """

    payload: Any
    nbits: int
    side: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Codec(Protocol):
    """Uniform runtime interface for compressionKIT codecs.

    Note:
        ``isinstance(obj, Codec)`` only checks for declared attributes
        (``name``, ``modality``, etc.), **not** the presence of
        ``compress``/``decompress``. This is a CPython limitation of
        ``runtime_checkable`` protocols. Use duck-typing or explicit
        ``hasattr`` checks when you need full structural verification.
    """

    name: str
    """Human-readable codec name (e.g. ``"spiht_ac"``, ``"rvq_64hz_04x"``)."""

    modality: str
    """``"ppg"`` or ``"ecg"``."""

    sample_rate: int
    """Frame sample rate in Hz."""

    frame_size: int
    """Number of samples per frame at the input layer."""

    target_cr: float
    """Nominal target compression ratio (informational; the true CR
    comes from :attr:`EncodedFrame.nbits`)."""

    def compress(self, frame: np.ndarray) -> EncodedFrame:
        """Encode a single ``(frame_size,)`` frame to an opaque payload."""
        ...

    def decompress(self, encoded: EncodedFrame) -> np.ndarray:
        """Decode an opaque payload back to a ``(frame_size,)`` frame."""
        ...
