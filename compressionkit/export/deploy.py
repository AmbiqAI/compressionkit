"""Unified deployment export for RVQ autoencoder models.

Exports all components needed for deployment:
  1. Encoder — INT8 TFLite + C header and FP32 LiteRT (edge/browser)
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
from compressionkit.export.release import write_checksums, write_json, write_model_card, write_scorecard_artifact
from compressionkit.export.tflite import export_decoder_tflite, export_encoder_tflite
from compressionkit.runtime.base import MANIFEST_VERSION

logger = logging.getLogger(__name__)


@dataclass
class DeploymentArtifacts:
    """Paths to all exported deployment artifacts."""

    output_dir: Path
    encoder_tflite: Path = field(default_factory=Path)
    encoder_header: Path = field(default_factory=Path)
    encoder_float32_tflite: Path = field(default_factory=Path)
    encoder_fp16_tflite: Path = field(default_factory=Path)
    encoder_fp16_header: Path = field(default_factory=Path)
    encoder_int16x8_tflite: Path = field(default_factory=Path)
    encoder_int16x8_header: Path = field(default_factory=Path)
    encoder_keras: Path = field(default_factory=Path)
    decoder_keras: Path = field(default_factory=Path)
    decoder_float32_tflite: Path = field(default_factory=Path)
    decoder_int8_tflite: Path = field(default_factory=Path)
    decoder_int8_header: Path = field(default_factory=Path)
    codebook_npz: Path = field(default_factory=Path)
    codebook_header: Path = field(default_factory=Path)
    sample_data_npz: Path = field(default_factory=Path)
    reference_vectors: Path = field(default_factory=Path)
    quantization_report: Path = field(default_factory=Path)
    model_card: Path = field(default_factory=Path)
    scorecard: Path = field(default_factory=Path)
    codec_spec: Path = field(default_factory=Path)
    manifest: Path = field(default_factory=Path)
    checksums: Path = field(default_factory=Path)
    readme: Path = field(default_factory=Path)

    def as_dict(self) -> dict[str, str]:
        """Return artifact paths relative to output_dir."""
        return {
            k: str(v.relative_to(self.output_dir)) if v != Path() else ""
            for k, v in self.__dict__.items()
            if k != "output_dir" and isinstance(v, Path)
        }


def _render_readme(
    *,
    model_name: str,
    modality: str | None,
    sample_rate: int | float | None,
    compression_ratio: int | float | None,
    quantization: str,
    io_type: str,
) -> str:
    modality_label = str(modality or "unknown").upper()
    sample_rate_label = f"{sample_rate:g} Hz" if isinstance(sample_rate, int | float) else "unknown"
    cr_label = f"{compression_ratio:g}x" if isinstance(compression_ratio, int | float) else "unknown"
    return f"""# compressionKIT RVQ deploy — {model_name}

AI codec deploy package for {modality_label} at {cr_label}.

## Operating point

| Field | Value |
|-------|-------|
| Modality | {modality_label} |
| Sample rate | {sample_rate_label} |
| Compression ratio | {cr_label} |
| Quantization | {quantization} |
| I/O type | {io_type} |

## Files

* `deploy_manifest.json` — top-level package manifest.
* `codec_spec.json` — canonical runtime hydration contract.
* `encoder_float32.tflite` — default FP32 encoder for browser and host LiteRT runtimes.
* `encoder.tflite`, `encoder_fp16.tflite`, `encoder_int16x8.tflite` — INT8, FP16,
  and INT16x8 edge encoder variants (each with a C header).
* `encoder.keras` — float32 Python reference encoder (training/inspection use).
* `codebook.npz`, `codebook.h` — RVQ codebook tables.
* `decoder.keras` and optional decoder TFLite files — reconstruction artifacts.
* `reference_vectors.npz` — known-good encode/decode vectors when sample inputs were exported.
* `model_card.json`, `scorecard.json` — optional release metadata and frozen evaluation summary.

## Python quickstart

```python
from compressionkit.runtime import load_codec

codec = load_codec("./")
indices = codec.encode(frame_batch)
recon = codec.decode(indices)
```
"""


def sync_scorecard_to_deploy(output_dir: str | Path, scorecard_summary: dict) -> Path:
    """Write ``scorecard.json`` into an existing RVQ deploy package.

    Also updates ``deploy_manifest.json``, refreshes the embedded model-card
    scorecard summary when present, and rewrites ``checksums.json`` so the
    deploy package remains self-consistent.
    """
    deploy_dir = Path(output_dir)
    scorecard_path = write_scorecard_artifact(deploy_dir, scorecard_summary)

    manifest_path = deploy_dir / "deploy_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest["scorecard"] = scorecard_path.name
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
            manifest["artifacts"] = artifacts
        artifacts["scorecard"] = scorecard_path.name
        write_json(manifest_path, manifest)

    model_card_path = deploy_dir / "model_card.json"
    if model_card_path.is_file():
        model_card = json.loads(model_card_path.read_text())
        if isinstance(model_card, dict):
            model_card["scorecard_summary"] = scorecard_summary
            write_json(model_card_path, model_card)

    write_checksums(deploy_dir)
    return scorecard_path


def export_for_deployment(
    encoder: keras.Model,
    decoder: keras.Model,
    rvq_weights: list[np.ndarray],
    *,
    rep_dataset: np.ndarray,
    quantization_validation_dataset: np.ndarray | None = None,
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
    export_decoder_int8: bool = True,
    model_card_info: dict | None = None,
    scorecard_summary: dict | None = None,
) -> DeploymentArtifacts:
    """Export encoder, codebook, decoder, and sample data for deployment.

    This is the primary entry point for producing all artifacts needed
    to run the RVQ autoencoder on an edge device.

    Args:
        encoder: Trained Keras encoder model.
        decoder: Trained Keras decoder model.
        rvq_weights: Weight list from ``rvq.get_weights()``.
        rep_dataset: Representative input dataset for TFLite calibration.
        quantization_validation_dataset: Disjoint preprocessed frames for
            post-export INT8-vs-float32 parity validation. Required for INT8.
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
        scorecard_summary: Optional frozen scorecard payload written to
            ``scorecard.json``.

    Returns:
        ``DeploymentArtifacts`` with paths to all exported files.
    """
    output_dir = Path(output_dir)
    if quantization == "INT8" and quantization_validation_dataset is None:
        raise ValueError("INT8 export requires a disjoint quantization_validation_dataset")
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

    # 1b. Float32 encoder LiteRT (browser/host runtime without Keras).
    logger.info("Exporting float32 encoder TFLite...")
    enc_f32_tflite, _ = export_encoder_tflite(
        encoder,
        rep_dataset=rep_dataset,
        output_dir=output_dir,
        tflite_name="encoder_float32.tflite",
        header_name="_encoder_float32.h",
        c_array_name="encoder_float32",
        quantization="FP32",
        io_type="float32",
    )
    artifacts.encoder_float32_tflite = enc_f32_tflite

    # 1c. Additional MCU encoder precisions. ``encoder.tflite`` remains the
    # backwards-compatible INT8 filename; explicit names let an application
    # select its HeliaAOT-supported precision without changing codec payloads.
    logger.info("Exporting FP16 and INT16x8 encoder variants...")
    enc_fp16_tflite, enc_fp16_header = export_encoder_tflite(
        encoder,
        rep_dataset=rep_dataset,
        output_dir=output_dir,
        tflite_name="encoder_fp16.tflite",
        header_name="encoder_fp16.h",
        c_array_name="encoder_fp16",
        quantization="FP16",
        io_type="float32",
    )
    enc_int16x8_tflite, enc_int16x8_header = export_encoder_tflite(
        encoder,
        rep_dataset=rep_dataset,
        output_dir=output_dir,
        tflite_name="encoder_int16x8.tflite",
        header_name="encoder_int16x8.h",
        c_array_name="encoder_int16x8",
        quantization="INT16X8",
        io_type="float32",
    )
    artifacts.encoder_fp16_tflite = enc_fp16_tflite
    artifacts.encoder_fp16_header = enc_fp16_header
    artifacts.encoder_int16x8_tflite = enc_int16x8_tflite
    artifacts.encoder_int16x8_header = enc_int16x8_header

    # 1d. Encoder (.keras -- float32 reference alongside the quantized
    # encoder.tflite used on-device; mirrors the decoder.keras reference below).
    logger.info("Exporting encoder as .keras...")
    encoder_keras_path = output_dir / "encoder.keras"
    encoder.save(encoder_keras_path)
    artifacts.encoder_keras = encoder_keras_path

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

    # 4b. Reference vectors via the exported LiteRT runtime when sample inputs exist.
    if sample_inputs is not None:
        artifacts.reference_vectors = output_dir / "reference_vectors.npz"

    artifacts.codec_spec = output_dir / "codec_spec.json"
    artifacts.manifest = output_dir / "deploy_manifest.json"
    artifacts.checksums = output_dir / "checksums.json"

    if scorecard_summary is None and model_card_info is not None:
        maybe_scorecard = model_card_info.get("scorecard_summary")
        if isinstance(maybe_scorecard, dict) and maybe_scorecard:
            scorecard_summary = maybe_scorecard
    if scorecard_summary is not None:
        artifacts.scorecard = write_scorecard_artifact(output_dir, scorecard_summary)

    # 5. Canonical runtime hydration spec
    codebooks = extract_codebooks(rvq_weights)
    preprocessing_contract = dict(model_card_info.get("preprocessing_contract", {})) if model_card_info else {}
    codec_spec = {
        "family": "rvq",
        "method": "ai",
        "modality": model_card_info.get("modality") if model_card_info else None,
        "sample_rate": model_card_info.get("sample_rate") if model_card_info else None,
        "compression_ratio": model_card_info.get("compression_ratio") if model_card_info else None,
        "experiment_id": model_card_info.get("experiment_id") if model_card_info else None,
        "run_name": model_card_info.get("run_name") if model_card_info else None,
        "package_version": model_version,
        "input_contract": {
            "encoder_input_shape": list(encoder.input_shape),
            "encoder_input_dtype": "float32",
        },
        "preprocessing_contract": preprocessing_contract,
        "encoder": {
            "tflite": enc_f32_tflite.name,
            "header": enc_header.name,
            "default_variant": "float32",
            "default_tflite": enc_f32_tflite.name,
            "float32_tflite": enc_f32_tflite.name,
            "fp16_tflite": enc_fp16_tflite.name,
            "fp16_header": enc_fp16_header.name,
            "int16x8_tflite": enc_int16x8_tflite.name,
            "int16x8_header": enc_int16x8_header.name,
            "int8_tflite": enc_tflite.name,
            "int8_header": enc_header.name,
            "variants": {
                "float32": {"tflite": enc_f32_tflite.name, "input_dtype": "float32"},
                "fp16": {"tflite": enc_fp16_tflite.name, "input_dtype": "float32"},
                "int16x8": {"tflite": enc_int16x8_tflite.name, "input_dtype": "float32"},
                "int8": {"tflite": enc_tflite.name, "input_dtype": io_type},
            },
            "keras": encoder_keras_path.name,
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
    write_json(artifacts.codec_spec, codec_spec)

    # 6. Release metadata and README are written before the manifest so the
    # artifact index sees the complete package.
    if model_card_info is not None:
        model_card_path = write_model_card(
            output_dir,
            model_name=model_name,
            model_version=model_version,
            model_card_info=model_card_info,
        )
        artifacts.model_card = model_card_path
        logger.info("Exported model card to %s", model_card_path)

    artifacts.readme = output_dir / "README.md"
    artifacts.readme.write_text(
        _render_readme(
            model_name=model_name,
            modality=codec_spec.get("modality"),
            sample_rate=codec_spec.get("sample_rate"),
            compression_ratio=codec_spec.get("compression_ratio"),
            quantization=quantization,
            io_type=io_type,
        )
    )

    # 7. Deployment manifest
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "package_version": model_version,
        "family": "rvq",
        "method": "ai",
        "modality": codec_spec.get("modality"),
        "experiment_id": codec_spec.get("experiment_id"),
        "run_name": codec_spec.get("run_name"),
        "compression_ratio": codec_spec.get("compression_ratio"),
        "spec": artifacts.codec_spec.name,
        "scorecard": artifacts.scorecard.name if artifacts.scorecard != Path() else None,
        "checksums": artifacts.checksums.name,
        "model_name": model_name,
        "model_version": model_version,
        "quantization": quantization,
        "io_type": io_type,
        "artifacts": artifacts.as_dict(),
        "encoder": {
            "tflite": enc_f32_tflite.name,
            "header": enc_header.name,
            "default_variant": "float32",
            "default_tflite": enc_f32_tflite.name,
            "float32_tflite": enc_f32_tflite.name,
            "fp16_tflite": enc_fp16_tflite.name,
            "fp16_header": enc_fp16_header.name,
            "int16x8_tflite": enc_int16x8_tflite.name,
            "int16x8_header": enc_int16x8_header.name,
            "int8_tflite": enc_tflite.name,
            "int8_header": enc_header.name,
            "keras": encoder_keras_path.name,
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
    write_json(artifacts.manifest, manifest)

    if sample_inputs is not None:
        from compressionkit.runtime.codec import RVQCodec

        try:
            codec = RVQCodec(output_dir)
            input_frames = np.asarray(sample_inputs, dtype=np.float32)
            encoded_batches: list[np.ndarray] = []
            decoded_batches: list[np.ndarray] = []
            for idx in range(input_frames.shape[0]):
                sample = input_frames[idx : idx + 1]
                encoded = codec.encode(sample).astype(np.int32)
                decoded = codec.decode(encoded).astype(np.float32)
                encoded_batches.append(encoded)
                decoded_batches.append(decoded)
            encoded_indices = np.concatenate(encoded_batches, axis=0)
            decoded_frames = np.concatenate(decoded_batches, axis=0)
            reference_payload = {
                "input_frames": input_frames,
                "indices": encoded_indices,
                "reconstructions": decoded_frames,
            }
            if sample_targets is not None:
                reference_payload["targets"] = np.asarray(sample_targets, dtype=np.float32)
            np.savez_compressed(artifacts.reference_vectors, **reference_payload)
        except Exception:
            logger.warning(
                "Reference-vector export skipped because the deploy package has no usable decoder TFLite.",
                exc_info=True,
            )
            artifacts.reference_vectors = Path()

    # 7b. Compare the deployed INT8 encoder against its float32 companion on
    # a disjoint, real preprocessed validation partition.
    if quantization == "INT8":
        from compressionkit.export.quantization import (
            evaluate_rvq_encoder_quantization,
            write_rvq_encoder_quantization_report,
        )

        report = evaluate_rvq_encoder_quantization(output_dir, quantization_validation_dataset, max_frames=None)
        artifacts.quantization_report = write_rvq_encoder_quantization_report(
            output_dir,
            report,
            calibration_frames=int(np.asarray(rep_dataset).shape[0]),
        )

    # 8. File-integrity manifest (written last so it can include everything else).
    artifacts.checksums = write_checksums(output_dir)

    logger.info("Deployment export complete: %s", output_dir)
    return artifacts


__all__ = ["DeploymentArtifacts", "export_for_deployment", "sync_scorecard_to_deploy"]
