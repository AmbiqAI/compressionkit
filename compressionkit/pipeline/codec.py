"""Composable codec assembling the four pipeline stages.

:class:`PipelineCodec` implements the :class:`compressionkit.evaluation.codec.Codec`
protocol by chaining a preprocessor, transform, sub-band encoder, and entropy
coder. Per-frame stage parameters (norm scale, quantizer steps) ride along in
:attr:`EncodedFrame.side`; only the entropy-coded payload counts toward
``nbits`` (matching the convention of the existing classical codecs).

When ``match_cr`` is set the codec runs a short bisection over the quantizer
``scale`` so the achieved per-frame bit budget lands near the requested target
compression ratio, making cross-method comparisons fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from compressionkit.evaluation.codec import EncodedFrame

from .dsp_stages import (
    DeadzoneQuantizer,
    DeflateEntropy,
    DwtTransform,
    ZNormPreprocessor,
)
from .stages import EntropyCoder, Preprocessor, SubbandEncoder, Transform


@dataclass
class PipelineCodec:
    """A codec composed of preprocess -> transform -> encoder -> entropy stages."""

    preprocess: Preprocessor
    transform: Transform
    encoder: SubbandEncoder
    entropy: EntropyCoder
    modality: str
    sample_rate: int
    frame_size: int
    target_cr: float
    bits_per_sample: int = 16
    match_cr: bool = True
    name: str = "pipeline"
    _scale_bounds: tuple[float, float] = field(default=(1e-2, 1e2), repr=False)

    def _encode_at(self, frame: np.ndarray) -> tuple[EncodedFrame, int]:
        processed, pre_ctx = self.preprocess.forward(frame)
        representation, tf_ctx = self.transform.forward(processed)
        symbols, enc_meta = self.encoder.forward(representation)
        bitstream, nbits = self.entropy.encode(symbols)
        side: dict[str, Any] = {
            "pre_ctx": pre_ctx,
            "tf_ctx": tf_ctx,
            "enc_meta": enc_meta,
            "n_symbols": int(np.asarray(symbols).size),
        }
        return EncodedFrame(payload=bitstream, nbits=nbits, side=side), nbits

    def encode(self, frame: np.ndarray) -> EncodedFrame:
        frame = np.asarray(frame, dtype=np.float32)
        if not self.match_cr or not hasattr(self.encoder, "scale"):
            return self._encode_at(frame)[0]

        budget = self.frame_size * self.bits_per_sample / self.target_cr
        lo, hi = self._scale_bounds
        best: EncodedFrame | None = None
        for _ in range(8):
            mid = float(np.sqrt(lo * hi))
            self.encoder.scale = mid  # type: ignore[attr-defined]
            encoded, nbits = self._encode_at(frame)
            best = encoded
            if nbits > budget:
                lo = mid  # coarser quantization (larger scale) -> fewer bits
            else:
                hi = mid
        assert best is not None
        return best

    def decode(self, encoded: EncodedFrame) -> np.ndarray:
        side = encoded.side
        symbols = self.entropy.decode(encoded.payload, side["n_symbols"])
        representation = self.encoder.inverse(symbols, side["enc_meta"])
        signal = self.transform.inverse(representation, side["tf_ctx"])
        frame = self.preprocess.inverse(signal, side["pre_ctx"])
        return np.asarray(frame, dtype=np.float32)[: self.frame_size]


def build_dwt_deadzone_codec(
    *,
    modality: str,
    sample_rate: int,
    frame_size: int,
    target_cr: float,
    preprocess: Preprocessor | None = None,
    wavelet: str = "bior4.4",
    levels: int = 6,
    name: str = "dwt_deadzone_deflate",
) -> PipelineCodec:
    """Build a faithful all-DSP codec: znorm -> DWT -> dead-zone quant -> deflate."""
    return PipelineCodec(
        preprocess=preprocess or ZNormPreprocessor(),
        transform=DwtTransform(wavelet=wavelet, levels=levels),
        encoder=DeadzoneQuantizer(),
        entropy=DeflateEntropy(),
        modality=modality,
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
        name=name,
    )


__all__ = [
    "PipelineCodec",
    "build_dwt_deadzone_codec",
]
