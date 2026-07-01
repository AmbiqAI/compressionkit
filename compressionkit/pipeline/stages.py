"""Stage contracts for the 4-stage compression pipeline.

A codec is composed of four independently swappable stages, each of which may
be a classical-DSP or a learned/AI implementation:

    preprocess  ->  transform  ->  encoder  ->  entropy

* :class:`Preprocessor` — denoise / normalize (e.g. band-pass, z-norm).
* :class:`Transform` — decorrelate (e.g. DWT, STFT, raw passthrough).
* :class:`SubbandEncoder` — quantize / bit-allocate to integer symbols
  (e.g. dead-zone quantizer, RVQ).
* :class:`EntropyCoder` — losslessly pack the symbols, exploiting longer
  trends/patterns (e.g. deflate/LZMA, arithmetic coding, a learned prior).

Each stage carries a small ``ctx``/``meta`` dict forward so the inverse path
can reconstruct. Per-frame stage parameters (norm scale, quantizer steps) are
kept in side information, mirroring the convention of the existing codecs in
:mod:`compressionkit.evaluation.codec` (only the entropy-coded payload counts
toward ``nbits``).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Preprocessor(Protocol):
    """Signal-domain conditioning (denoise / normalize)."""

    name: str

    def forward(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Return ``(processed_signal, ctx)`` for a 1-D frame."""
        ...

    def inverse(self, signal: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        """Undo invertible conditioning (identity for lossy denoising)."""
        ...


@runtime_checkable
class Transform(Protocol):
    """Decorrelating transform between signal and a coefficient representation."""

    name: str

    def forward(self, signal: np.ndarray) -> tuple[Any, dict[str, Any]]:
        """Return ``(representation, ctx)``."""
        ...

    def inverse(self, representation: Any, ctx: dict[str, Any]) -> np.ndarray:
        """Reconstruct the signal from a representation."""
        ...


@runtime_checkable
class SubbandEncoder(Protocol):
    """Quantize / bit-allocate a representation to a flat integer symbol vector."""

    name: str

    def forward(self, representation: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Return ``(symbols, meta)`` where ``symbols`` is a 1-D int array."""
        ...

    def inverse(self, symbols: np.ndarray, meta: dict[str, Any]) -> Any:
        """Reconstruct the (dequantized) representation from symbols."""
        ...


@runtime_checkable
class EntropyCoder(Protocol):
    """Losslessly code an integer symbol vector to a bitstream."""

    name: str

    def encode(self, symbols: np.ndarray) -> tuple[bytes, int]:
        """Return ``(bitstream, nbits)`` for a 1-D int symbol vector."""
        ...

    def decode(self, bitstream: bytes, n_symbols: int) -> np.ndarray:
        """Reconstruct the integer symbol vector."""
        ...


__all__ = [
    "EntropyCoder",
    "Preprocessor",
    "SubbandEncoder",
    "Transform",
]
