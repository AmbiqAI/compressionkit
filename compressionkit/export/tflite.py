"""TFLite export via helia_edge converters."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

import helia_edge as helia
import keras
import numpy as np

logger = logging.getLogger(__name__)


def export_encoder_tflite(
    encoder: keras.Model,
    *,
    rep_dataset: np.ndarray,
    output_dir: Path,
    tflite_name: str = "encoder.tflite",
    header_name: str = "encoder.h",
    c_array_name: str = "encoder",
    quantization: str = "INT8",
    io_type: str = "int8",
) -> tuple[Path, Path]:
    """Export encoder to INT8 TFLite and C header using helia_edge.

    Args:
        encoder: Keras encoder model to export.
        rep_dataset: Representative dataset array for calibration.
        output_dir: Directory to write ``.tflite`` and ``.h`` files.
        tflite_name: Filename for the TFLite model.
        header_name: Filename for the C header.
        c_array_name: Name for the C array in the header.
        quantization: Quantization mode (e.g. ``"INT8"``).
        io_type: I/O type string (e.g. ``"int8"``).

    Returns:
        Tuple of ``(tflite_path, header_path)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = helia.converters.tflite.TfLiteKerasConverter(model=encoder)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        converter.convert(
            test_x=rep_dataset,
            quantization=quantization,
            io_type=io_type,
            mode="KERAS",
            strict=False,
            verbose=0,
        )
    tflite_path = output_dir / tflite_name
    header_path = output_dir / header_name
    converter.export(tflite_path=tflite_path)
    converter.export_header(header_path=header_path, name=c_array_name)
    logger.info("Exported encoder TFLite to %s and header to %s", tflite_path, header_path)
    return tflite_path, header_path


__all__ = ["export_encoder_tflite"]
