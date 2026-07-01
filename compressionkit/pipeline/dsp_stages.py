"""Classical-DSP implementations of the four pipeline stages.

These wrap the reusable building blocks in :mod:`compressionkit.dsp` so that
faithful all-DSP codecs can be assembled by composition. Every stage is
embedded-portable in spirit: fixed transforms, uniform quantization, and
standard lossless coders (deflate/LZMA are portable C).
"""

from __future__ import annotations

import lzma
import math
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from compressionkit.dsp.filters import bandpass
from compressionkit.dsp.wavelet import WaveletCoeffs, dwt_forward, dwt_inverse

# ---------------------------------------------------------------------------
# Preprocessors
# ---------------------------------------------------------------------------


@dataclass
class IdentityPreprocessor:
    """No conditioning."""

    name: str = "identity"

    def forward(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        return np.asarray(frame, dtype=np.float32), {}

    def inverse(self, signal: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        return np.asarray(signal, dtype=np.float32)


@dataclass
class ZNormPreprocessor:
    """Per-frame zero-mean / unit-variance normalization (invertible)."""

    name: str = "znorm"

    def forward(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        arr = np.asarray(frame, dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std()) + 1e-8
        return (arr - mean) / std, {"mean": mean, "std": std}

    def inverse(self, signal: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        return np.asarray(signal, dtype=np.float32) * ctx["std"] + ctx["mean"]


@dataclass
class BandpassPreprocessor:
    """Zero-phase Butterworth band-pass denoise (lossy; inverse is identity)."""

    sample_rate: float
    low_hz: float = 0.5
    high_hz: float = 40.0
    order: int = 3
    name: str = "bandpass"

    def forward(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        filtered = bandpass(frame, self.sample_rate, self.low_hz, self.high_hz, order=self.order)
        return filtered, {}

    def inverse(self, signal: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        return np.asarray(signal, dtype=np.float32)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


@dataclass
class RawTransform:
    """Identity transform; the representation is the signal itself."""

    name: str = "raw"

    def forward(self, signal: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        arr = np.asarray(signal, dtype=np.float32)
        return arr, {"length": int(arr.size)}

    def inverse(self, representation: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        return np.asarray(representation, dtype=np.float32)[: ctx["length"]]


@dataclass
class DwtTransform:
    """Discrete wavelet transform via :mod:`compressionkit.dsp.wavelet`."""

    wavelet: str = "bior4.4"
    levels: int = 6
    name: str = "dwt"

    def forward(self, signal: np.ndarray) -> tuple[WaveletCoeffs, dict[str, Any]]:
        arr = np.asarray(signal, dtype=np.float32)
        coeffs = dwt_forward(arr, levels=self.levels, wavelet=self.wavelet)
        return coeffs, {"length": int(arr.size)}

    def inverse(self, representation: WaveletCoeffs, ctx: dict[str, Any]) -> np.ndarray:
        recon = dwt_inverse(representation, wavelet=self.wavelet)
        return np.asarray(recon, dtype=np.float32)[: ctx["length"]]


# ---------------------------------------------------------------------------
# Sub-band encoder (dead-zone uniform quantizer)
# ---------------------------------------------------------------------------


@dataclass
class DeadzoneQuantizer:
    """Uniform dead-zone quantizer over raw or DWT representations.

    ``scale`` multiplies the per-band RMS to set the step size; the codec
    adjusts it to hit a target compression ratio. The approximation/low band
    uses a finer step (``approx_ratio``) to protect coarse morphology.
    """

    scale: float = 0.25
    approx_ratio: float = 0.25
    name: str = "deadzone"

    def _step(self, band: np.ndarray, ratio: float) -> float:
        rms = float(np.sqrt(np.mean(np.asarray(band, dtype=np.float32) ** 2))) + 1e-12
        return rms * self.scale * ratio

    def forward(self, representation: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if isinstance(representation, WaveletCoeffs):
            approx = np.asarray(representation.approx, dtype=np.float32)
            details = [np.asarray(d, dtype=np.float32) for d in representation.details]
            steps = [self._step(approx, self.approx_ratio)] + [self._step(d, 1.0) for d in details]
            bands = [approx, *details]
            sizes = [int(b.size) for b in bands]
            symbols = np.concatenate([np.round(b / s).astype(np.int32) for b, s in zip(bands, steps, strict=True)])
            meta = {"kind": "dwt", "sizes": sizes, "steps": steps}
            return symbols, meta

        arr = np.asarray(representation, dtype=np.float32)
        step = self._step(arr, 1.0)
        symbols = np.round(arr / step).astype(np.int32)
        return symbols, {"kind": "raw", "sizes": [int(arr.size)], "steps": [step]}

    def inverse(self, symbols: np.ndarray, meta: dict[str, Any]) -> Any:
        symbols = np.asarray(symbols, dtype=np.int32)
        sizes = meta["sizes"]
        steps = meta["steps"]
        offsets = np.cumsum([0, *sizes])
        bands = [symbols[offsets[i] : offsets[i + 1]] * steps[i] for i in range(len(sizes))]
        if meta["kind"] == "dwt":
            return WaveletCoeffs(approx=bands[0].astype(np.float32), details=[b.astype(np.float32) for b in bands[1:]])
        return bands[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Entropy coders
# ---------------------------------------------------------------------------


@dataclass
class RawEntropy:
    """Fixed-width packing; the honest no-entropy-coding baseline."""

    name: str = "raw"

    def encode(self, symbols: np.ndarray) -> tuple[bytes, int]:
        sym = np.asarray(symbols, dtype=np.int32)
        max_abs = int(np.abs(sym).max()) if sym.size else 0
        bit_depth = max(1, math.ceil(math.log2(2 * max_abs + 2)))
        nbits = int(sym.size * bit_depth)
        return sym.tobytes(), nbits

    def decode(self, bitstream: bytes, n_symbols: int) -> np.ndarray:
        return np.frombuffer(bitstream, dtype=np.int32, count=n_symbols).copy()


@dataclass
class DeflateEntropy:
    """zlib/DEFLATE (LZ77 + Huffman) — captures repeated patterns in-frame."""

    level: int = 9
    name: str = "deflate"

    def encode(self, symbols: np.ndarray) -> tuple[bytes, int]:
        payload = zlib.compress(np.asarray(symbols, dtype=np.int32).tobytes(), self.level)
        return payload, int(len(payload) * 8)

    def decode(self, bitstream: bytes, n_symbols: int) -> np.ndarray:
        return np.frombuffer(zlib.decompress(bitstream), dtype=np.int32, count=n_symbols).copy()


@dataclass
class LzmaEntropy:
    """LZMA — stronger context modelling at higher cost."""

    preset: int = 6
    name: str = "lzma"

    def encode(self, symbols: np.ndarray) -> tuple[bytes, int]:
        payload = lzma.compress(np.asarray(symbols, dtype=np.int32).tobytes(), preset=self.preset)
        return payload, int(len(payload) * 8)

    def decode(self, bitstream: bytes, n_symbols: int) -> np.ndarray:
        return np.frombuffer(lzma.decompress(bitstream), dtype=np.int32, count=n_symbols).copy()


__all__ = [
    "BandpassPreprocessor",
    "DeadzoneQuantizer",
    "DeflateEntropy",
    "DwtTransform",
    "IdentityPreprocessor",
    "LzmaEntropy",
    "RawEntropy",
    "RawTransform",
    "ZNormPreprocessor",
]
