"""Shared LiteRT/TFLite ``Interpreter`` resolution for the runtime package.

Every LiteRT-backed runtime component (:class:`~compressionkit.runtime.codec.RVQCodec`,
the quantized denoiser preprocessor) needs the same ``Interpreter`` class and
INT8 quantize/dequantize helpers. Centralized here so the import-fallback
chain (``ai-edge-litert`` -> ``tflite-runtime`` -> ``tensorflow.lite``) and the
quantization math aren't duplicated per module.
"""

from __future__ import annotations

import contextlib

import numpy as np

# Try ai-edge-litert first, then tflite-runtime, then tf.lite
Interpreter = None

with contextlib.suppress(ImportError):
    from ai_edge_litert.interpreter import Interpreter  # type: ignore[assignment]
if Interpreter is None:
    with contextlib.suppress(ImportError):
        from tflite_runtime.interpreter import Interpreter  # type: ignore[assignment]
if Interpreter is None:
    with contextlib.suppress(ImportError):
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore[assignment]

if Interpreter is None:
    raise ImportError("No TFLite runtime found. Install one of: ai-edge-litert, tflite-runtime, or tensorflow.")


def quantize(data: np.ndarray, details: dict) -> np.ndarray:
    """Quantize float32 data to a TFLite tensor's dtype using its quantization params."""
    qparams = details.get("quantization_parameters", {})
    scales = qparams.get("scales", np.array([1.0]))
    zero_points = qparams.get("zero_points", np.array([0]))
    dtype = details["dtype"]
    if dtype in (np.int8, np.int16):
        info = np.iinfo(dtype)
        quantized = np.round(data / scales[0] + zero_points[0])
        return np.clip(quantized, info.min, info.max).astype(dtype)
    return data.astype(dtype)


def dequantize(data: np.ndarray, details: dict) -> np.ndarray:
    """Dequantize a TFLite tensor's output back to float32."""
    qparams = details.get("quantization_parameters", {})
    scales = qparams.get("scales", np.array([1.0]))
    zero_points = qparams.get("zero_points", np.array([0]))
    if details["dtype"] in (np.int8, np.int16):
        return ((data.astype(np.float32) - zero_points[0]) * scales[0]).astype(np.float32)
    return data.astype(np.float32)


__all__ = ["Interpreter", "dequantize", "quantize"]
