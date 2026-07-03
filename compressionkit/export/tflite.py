"""TFLite / LiteRT export via helia_edge converters."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

import helia_edge as helia
import keras
import numpy as np

logger = logging.getLogger(__name__)


def _with_fixed_batch_one(model: keras.Model) -> keras.Model:
    """Return ``model`` with a fixed batch dimension of 1 when its input batch is dynamic."""
    input_shape = model.input_shape
    if not isinstance(input_shape, tuple) or not input_shape or input_shape[0] is not None:
        return model

    fixed_in = keras.Input(
        batch_shape=(1, *input_shape[1:]),
        dtype=model.inputs[0].dtype,
        name=f"{model.name}_batch1_in",
    )
    fixed_out = model(fixed_in)
    fixed_model = keras.Model(fixed_in, fixed_out, name=f"{model.name}_batch1")
    fixed_model.set_weights(model.get_weights())
    return fixed_model


def _convert_and_export(
    model: keras.Model,
    *,
    rep_dataset: np.ndarray,
    output_dir: Path,
    tflite_name: str,
    header_name: str,
    c_array_name: str,
    quantization: str = "INT8",
    io_type: str = "int8",
) -> tuple[Path, Path]:
    """Convert a Keras model to quantized TFLite and export with C header.

    Args:
        model: Keras model to export.
        rep_dataset: Representative dataset array for calibration.
        output_dir: Directory to write files into.
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

    converter = helia.converters.tflite.TfLiteKerasConverter(model=model)
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
    return tflite_path, header_path


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
    paths = _convert_and_export(
        encoder,
        rep_dataset=rep_dataset,
        output_dir=output_dir,
        tflite_name=tflite_name,
        header_name=header_name,
        c_array_name=c_array_name,
        quantization=quantization,
        io_type=io_type,
    )
    logger.info("Exported encoder TFLite to %s and header to %s", paths[0], paths[1])
    return paths


def export_decoder_tflite(
    decoder: keras.Model,
    *,
    rep_latents: np.ndarray,
    output_dir: Path,
    tflite_name: str = "decoder.tflite",
    header_name: str = "decoder.h",
    c_array_name: str = "decoder",
    quantization: str = "INT8",
    io_type: str = "int8",
) -> tuple[Path, Path]:
    """Export decoder to INT8 TFLite and C header.

    Args:
        decoder: Keras decoder model to export.
        rep_latents: Representative latent array for calibration.
        output_dir: Directory to write ``.tflite`` and ``.h`` files.
        tflite_name: Filename for the TFLite model.
        header_name: Filename for the C header.
        c_array_name: Name for the C array in the header.
        quantization: Quantization mode (e.g. ``"INT8"``).
        io_type: I/O type string (e.g. ``"int8"``).

    Returns:
        Tuple of ``(tflite_path, header_path)``.
    """
    export_decoder = _with_fixed_batch_one(decoder)
    export_latents = rep_latents[:1] if export_decoder is not decoder else rep_latents

    paths = _convert_and_export(
        export_decoder,
        rep_dataset=export_latents,
        output_dir=output_dir,
        tflite_name=tflite_name,
        header_name=header_name,
        c_array_name=c_array_name,
        quantization=quantization,
        io_type=io_type,
    )
    logger.info("Exported decoder TFLite to %s and header to %s", paths[0], paths[1])
    return paths


def export_denoiser_tflite(
    model: keras.Model,
    *,
    rep_dataset: np.ndarray,
    output_dir: Path,
    tflite_name: str = "denoiser_gain_model.tflite",
    header_name: str = "denoiser_gain_model.h",
    c_array_name: str = "denoiser_gain_model",
    quantization: str = "INT8",
    io_type: str = "int8",
) -> tuple[Path, Path]:
    """Export a wavelet-gain denoiser to INT8 TFLite and C header.

    See :mod:`compressionkit.models.wavelet_denoiser` for the model itself
    (``detect_denoiser_mode()`` describes the model's input/output calling
    convention, which must be captured separately since it isn't recoverable
    from the converted ``.tflite`` model alone).

    Args:
        model: Trained denoiser Keras model. Expects a ``(1, T, C)`` input
            (``C`` is 1 or 2 depending on whether the model was trained with
            a noise-level feature channel — see ``DenoiserMode.expects_level``).
        rep_dataset: Representative dataset array for INT8 calibration,
            shape ``(N, 1, T, C)`` matching the model's input. Unlike
            :func:`export_decoder_tflite`, the full dataset is used for
            calibration even when the batch dimension gets fixed to 1 for
            export, since ``helia_edge`` already feeds calibration examples
            one at a time regardless.
        output_dir: Directory to write ``.tflite`` and ``.h`` files.
        tflite_name: Filename for the TFLite model.
        header_name: Filename for the C header.
        c_array_name: Name for the C array in the header.
        quantization: Quantization mode (e.g. ``"INT8"``).
        io_type: I/O type string (e.g. ``"int8"``).

    Returns:
        Tuple of ``(tflite_path, header_path)``.
    """
    export_model = _with_fixed_batch_one(model)

    paths = _convert_and_export(
        export_model,
        rep_dataset=rep_dataset,
        output_dir=output_dir,
        tflite_name=tflite_name,
        header_name=header_name,
        c_array_name=c_array_name,
        quantization=quantization,
        io_type=io_type,
    )
    logger.info("Exported denoiser TFLite to %s and header to %s", paths[0], paths[1])
    return paths


__all__ = ["export_decoder_tflite", "export_denoiser_tflite", "export_encoder_tflite"]
