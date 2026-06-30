"""Composable 4-stage compression pipeline (preprocess/transform/encoder/entropy).

This package provides the stage contracts (:mod:`.stages`), classical-DSP stage
implementations (:mod:`.dsp_stages`), and a composing :class:`PipelineCodec`
(:mod:`.codec`) that plugs into the evaluation harness via the standard
``Codec`` protocol.
"""

from __future__ import annotations

from .codec import PipelineCodec, build_dwt_deadzone_codec
from .dsp_stages import (
    BandpassPreprocessor,
    DeadzoneQuantizer,
    DeflateEntropy,
    DwtTransform,
    IdentityPreprocessor,
    LzmaEntropy,
    RawEntropy,
    RawTransform,
    ZNormPreprocessor,
)
from .learned_stages import LearnedDenoisePreprocessor, load_wavelet_gain_preprocessor
from .stages import EntropyCoder, Preprocessor, SubbandEncoder, Transform

__all__ = [
    "BandpassPreprocessor",
    "DeadzoneQuantizer",
    "DeflateEntropy",
    "DwtTransform",
    "EntropyCoder",
    "IdentityPreprocessor",
    "LearnedDenoisePreprocessor",
    "LzmaEntropy",
    "PipelineCodec",
    "Preprocessor",
    "RawEntropy",
    "RawTransform",
    "SubbandEncoder",
    "Transform",
    "ZNormPreprocessor",
    "build_dwt_deadzone_codec",
    "load_wavelet_gain_preprocessor",
]
