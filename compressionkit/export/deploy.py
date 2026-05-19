"""Unified deployment export for RVQ autoencoder models.

Exports all components needed for deployment:
  1. Encoder — INT8 TFLite + C header  (on-device)
  2. Codebook tables — C header + NumPy archive  (on-device)
  3. Decoder — Keras model + optional TFLite exports  (server / on-device)
  4. Model card JSON with metadata and scorecard summary
  5. Synthetic stimulus data (license-safe examples)

Also writes a deployment manifest (``deploy_manifest.json``) describing
the exported artifacts and their properties.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import keras
import numpy as np

from compressionkit.export.codebook import (
    export_codebooks_header,
    export_codebooks_npz,
    extract_codebooks,
)
from compressionkit.export.tflite import export_decoder_tflite, export_encoder_tflite

logger = logging.getLogger(__name__)


@dataclass
class DeploymentArtifacts:
    """Paths to all exported deployment artifacts."""

    output_dir: Path
    encoder_tflite: Path = field(default_factory=Path)
    encoder_header: Path = field(default_factory=Path)
    decoder_keras: Path = field(default_factory=Path)
    decoder_float32_tflite: Path = field(default_factory=Path)
    decoder_int8_tflite: Path = field(default_factory=Path)
    decoder_int8_header: Path = field(default_factory=Path)
    codebook_npz: Path = field(default_factory=Path)
    codebook_header: Path = field(default_factory=Path)
    sample_data_npz: Path = field(default_factory=Path)
    model_card: Path = field(default_factory=Path)
    manifest: Path = field(default_factory=Path)

    def as_dict(self) -> dict[str, str]:
        """Return artifact paths relative to output_dir."""
        return {
            k: str(v.relative_to(self.output_dir)) if v != Path() else ""
            for k, v in self.__dict__.items()
            if k != "output_dir" and isinstance(v, Path)
        }


def export_for_deployment(
    encoder: keras.Model,
    decoder: keras.Model,
    rvq_weights: list[np.ndarray],
    *,
    rep_dataset: np.ndarray,
    output_dir: str | Path,
    sample_inputs: np.ndarray | None = None,
    sample_targets: np.ndarray | None = None,
    sample_reconstructions: np.ndarray | None = None,
    quantization: str = "INT8",
    io_type: str = "int8",
    codebook_prefix: str = "rvq_codebook",
    model_name: str = "rvq_autoencoder",
    model_version: str = "1.0",
    export_decoder_float32: bool = True,
    export_decoder_int8: bool = False,
    model_card_info: dict | None = None,
) -> DeploymentArtifacts:
    """Export encoder, codebook, decoder, and sample data for deployment.

    This is the primary entry point for producing all artifacts needed
    to run the RVQ autoencoder on an edge device.

    Args:
        encoder: Trained Keras encoder model.
        decoder: Trained Keras decoder model.
        rvq_weights: Weight list from ``rvq.get_weights()``.
        rep_dataset: Representative input dataset for TFLite calibration.
        output_dir: Directory to write all deployment artifacts.
        sample_inputs: Optional sample input frames for validation.
        sample_targets: Optional sample target frames for validation.
        sample_reconstructions: Optional model reconstructions of targets.
        quantization: Quantization mode for TFLite conversion.
        io_type: I/O type string for TFLite conversion.
        codebook_prefix: Prefix for codebook C array names.
        model_name: Name used in the deployment manifest.
        model_version: Semantic version string (e.g. ``"1.0"``).
        export_decoder_float32: Whether to export float32 decoder TFLite.
        export_decoder_int8: Whether to export INT8 decoder TFLite + header.
        model_card_info: Optional dict with modality, sample_rate,
            compression_ratio, license, and scorecard_summary fields.

    Returns:
        ``DeploymentArtifacts`` with paths to all exported files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = DeploymentArtifacts(output_dir=output_dir)

    # 1. Encoder TFLite
    logger.info("Exporting encoder...")
    enc_tflite, enc_header = export_encoder_tflite(
        encoder,
        rep_dataset=rep_dataset,
        output_dir=output_dir,
        quantization=quantization,
        io_type=io_type,
    )
    artifacts.encoder_tflite = enc_tflite
    artifacts.encoder_header = enc_header

    # 2. Decoder (.keras — runs server-side, no quantization needed)
    logger.info("Exporting decoder as .keras...")
    decoder_keras_path = output_dir / "decoder.keras"
    decoder.save(decoder_keras_path)
    artifacts.decoder_keras = decoder_keras_path

    # 2b. Float32 decoder TFLite (server-side, no quantization loss)
    if export_decoder_float32:
        logger.info("Exporting float32 decoder TFLite...")
        rep_latents = encoder.predict(rep_dataset[:32], verbose=0)
        try:
            dec_f32_tflite, _ = export_decoder_tflite(
                decoder,
                rep_latents=rep_latents,
                output_dir=output_dir,
                tflite_name="decoder_float32.tflite",
                header_name="_decoder_float32.h",
                c_array_name="decoder_float32",
                quantization="FP32",
                io_type="float32",
            )
            artifacts.decoder_float32_tflite = dec_f32_tflite
        except Exception:
            logger.warning(
                "Float32 decoder TFLite export failed (non-critical); "
                "decoder.keras is still available for server-side use."
            )

    # 2c. INT8 decoder TFLite (on-device reconstruction)
    if export_decoder_int8:
        logger.info("Exporting INT8 decoder TFLite...")
        if not export_decoder_float32:
            rep_latents = encoder.predict(rep_dataset[:32], verbose=0)
        dec_int8_tflite, dec_int8_header = export_decoder_tflite(
            decoder,
            rep_latents=rep_latents,
            output_dir=output_dir,
            tflite_name="decoder.tflite",
            header_name="decoder.h",
            c_array_name="decoder",
            quantization=quantization,
            io_type=io_type,
        )
        artifacts.decoder_int8_tflite = dec_int8_tflite
        artifacts.decoder_int8_header = dec_int8_header

    # 3. Codebook tables
    logger.info("Exporting codebook tables...")
    artifacts.codebook_npz = export_codebooks_npz(
        rvq_weights,
        output_dir / "codebook.npz",
    )
    artifacts.codebook_header = export_codebooks_header(
        rvq_weights,
        output_dir / "codebook.h",
        prefix=codebook_prefix,
    )

    # 4. Sample data for validation / testing
    sample_data_path = output_dir / "sample_data.npz"
    sample_arrays: dict[str, np.ndarray] = {}
    if sample_inputs is not None:
        sample_arrays["inputs"] = np.asarray(sample_inputs, dtype=np.float32)
    if sample_targets is not None:
        sample_arrays["targets"] = np.asarray(sample_targets, dtype=np.float32)
    if sample_reconstructions is not None:
        sample_arrays["reconstructions"] = np.asarray(sample_reconstructions, dtype=np.float32)
    if sample_arrays:
        np.savez(sample_data_path, **sample_arrays)
        artifacts.sample_data_npz = sample_data_path
        logger.info("Exported sample data (%d arrays) to %s", len(sample_arrays), sample_data_path)

    # 5. Deployment manifest
    codebooks = extract_codebooks(rvq_weights)
    manifest = {
        "model_name": model_name,
        "model_version": model_version,
        "quantization": quantization,
        "io_type": io_type,
        "encoder": {
            "tflite": enc_tflite.name,
            "header": enc_header.name,
            "input_shape": list(encoder.input_shape),
            "output_shape": list(encoder.output_shape),
        },
        "decoder": {
            "keras": decoder_keras_path.name,
            "float32_tflite": artifacts.decoder_float32_tflite.name
            if artifacts.decoder_float32_tflite != Path()
            else None,
            "int8_tflite": artifacts.decoder_int8_tflite.name if artifacts.decoder_int8_tflite != Path() else None,
            "int8_header": artifacts.decoder_int8_header.name if artifacts.decoder_int8_header != Path() else None,
            "input_shape": list(decoder.input_shape),
            "output_shape": list(decoder.output_shape),
        },
        "codebook": {
            "npz": "codebook.npz",
            "header": "codebook.h",
            "num_levels": len(codebooks),
            "num_embeddings": int(codebooks[0].shape[0]) if codebooks else 0,
            "embedding_dim": int(codebooks[0].shape[1]) if codebooks else 0,
        },
        "sample_data": {
            "npz": sample_data_path.name if sample_arrays else None,
            "num_samples": int(sample_arrays["inputs"].shape[0]) if "inputs" in sample_arrays else 0,
            "arrays": list(sample_arrays.keys()),
        },
    }
    manifest_path = output_dir / "deploy_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    artifacts.manifest = manifest_path

    # 6. Model card
    if model_card_info is not None:
        model_card = {
            "model_name": model_name,
            "model_version": model_version,
            "modality": model_card_info.get("modality", "unknown"),
            "sample_rate": model_card_info.get("sample_rate"),
            "compression_ratio": model_card_info.get("compression_ratio"),
            "license": model_card_info.get("license", "other"),
            "scorecard_summary": model_card_info.get("scorecard_summary", {}),
        }
        model_card_path = output_dir / "model_card.json"
        with model_card_path.open("w") as f:
            json.dump(model_card, f, indent=2)
        artifacts.model_card = model_card_path
        logger.info("Exported model card to %s", model_card_path)

    logger.info("Deployment export complete: %s", output_dir)
    return artifacts


__all__ = ["DeploymentArtifacts", "export_for_deployment"]
