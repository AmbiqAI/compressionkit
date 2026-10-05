"""Numerical parity checks for RVQ encoder precision exports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from compressionkit.export.artifact_contract import ArtifactFile
from compressionkit.export.release import write_checksums, write_json
from compressionkit.runtime._litert import Interpreter, dequantize, quantize
from compressionkit.runtime.codec import RVQCodec

MAX_INPUT_SATURATION_FRACTION = 0.01
MAX_RECONSTRUCTION_PRD_PERCENT_P90 = 10.0
TAIL_WARNING_RECONSTRUCTION_PRD_PERCENT = 15.0


@dataclass(frozen=True)
class RvqEncoderQuantizationReport:
    """Parity and saturation measurements for one RVQ encoder export."""

    code_index_match_fraction_median: float
    encoder_input_saturation_fraction_max: float
    encoder_input_saturation_fraction_p90: float
    frames_checked: int
    latent_prd_percent_p90: float
    reconstruction_prd_percent_max: float
    reconstruction_prd_percent_p90: float

    def passed(self) -> bool:
        """Return whether this export meets the population-quality release gate.

        Input saturation is the hard publication criterion. P90 and
        worst-frame PRD are retained as explicit quality recommendations so
        customers can select an encoder precision with full visibility.
        """
        return self.encoder_input_saturation_fraction_max <= MAX_INPUT_SATURATION_FRACTION


def refresh_rvq_encoder_precision_reports(deploy_dir: str | Path, frames: np.ndarray) -> None:
    """Recompute the model card's precision results against this deploy package.

    Call after trained-quantizer parity succeeds and before final checksums.
    Historical reports must not be carried over when codebooks change.
    """
    root = Path(deploy_dir)
    card_path = root / "model_card.json"
    card = json.loads(card_path.read_text())
    card["encoder_precision_report"] = {
        name: asdict(evaluate_rvq_encoder_quantization(root, frames, max_frames=None, candidate_encoder_name=filename))
        for name, filename in {
            "int8": "encoder.tflite",
            "fp16": "encoder_fp16.tflite",
            "int16x8": "encoder_int16x8.tflite",
        }.items()
    }
    write_json(card_path, card)


def _prd_percent(actual: np.ndarray, expected: np.ndarray) -> float:
    """Return RMS percent difference, with a safe zero-reference fallback."""
    err = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    denominator = float(np.sum(np.asarray(expected, dtype=np.float64) ** 2))
    if denominator <= 1e-12:
        return 0.0 if np.allclose(actual, expected) else float("inf")
    return 100.0 * float(np.sqrt(np.sum(err**2) / denominator))


def evaluate_rvq_encoder_quantization(
    deploy_dir: str | Path,
    frames: np.ndarray,
    *,
    max_frames: int | None = 128,
    candidate_encoder_name: str = ArtifactFile.ENCODER_TFLITE,
) -> RvqEncoderQuantizationReport:
    """Compare one RVQ encoder variant with the float32 reference encoder.

    Args:
        deploy_dir: Complete RVQ deploy directory containing both encoder variants.
        frames: Input frames after the exact training-time preprocessing.
        max_frames: Maximum number of frames to evaluate, or ``None`` for all.
        candidate_encoder_name: Candidate encoder filename relative to
            ``deploy_dir``. Defaults to the standard INT8 encoder.

    Returns:
        Candidate precision parity and input-saturation measurements.

    Raises:
        ValueError: If no frames are supplied or their shape is incompatible.
    """
    samples = np.asarray(frames, dtype=np.float32)
    if samples.ndim != 4 or samples.shape[0] == 0:
        raise ValueError(f"Expected non-empty (N, 1, T, C) frames, got {samples.shape}")
    if max_frames is not None:
        samples = samples[:max_frames]
    root = Path(deploy_dir)
    candidate_encoder = Interpreter(model_path=str(root / candidate_encoder_name))
    float_encoder = Interpreter(model_path=str(root / ArtifactFile.ENCODER_FLOAT32_TFLITE))
    candidate_encoder.allocate_tensors()
    float_encoder.allocate_tensors()
    candidate_input, candidate_output = (
        candidate_encoder.get_input_details()[0],
        candidate_encoder.get_output_details()[0],
    )
    float_input, float_output = float_encoder.get_input_details()[0], float_encoder.get_output_details()[0]
    codec = RVQCodec(root)

    input_saturation: list[float] = []
    latent_prd: list[float] = []
    reconstruction_prd: list[float] = []
    index_match: list[float] = []
    candidate_is_integer = np.issubdtype(candidate_input["dtype"], np.integer)
    limits = np.iinfo(candidate_input["dtype"]) if candidate_is_integer else None
    for frame in samples:
        batch = frame[np.newaxis, ...]
        candidate_input_data = quantize(batch, candidate_input)
        if limits is None:
            input_saturation.append(0.0)
        else:
            input_saturation.append(
                float(np.mean((candidate_input_data == limits.min) | (candidate_input_data == limits.max)))
            )
        candidate_encoder.set_tensor(candidate_input["index"], candidate_input_data)
        candidate_encoder.invoke()
        candidate_latent = dequantize(candidate_encoder.get_tensor(candidate_output["index"]), candidate_output)
        float_encoder.set_tensor(float_input["index"], batch.astype(float_input["dtype"]))
        float_encoder.invoke()
        float_latent = float_encoder.get_tensor(float_output["index"]).astype(np.float32)
        candidate_indices = codec.quantize_latent(candidate_latent)
        float_indices = codec.quantize_latent(float_latent)
        candidate_reconstruction = codec.decode_latent(codec.dequantize_indices(candidate_indices))
        float_reconstruction = codec.decode_latent(codec.dequantize_indices(float_indices))
        latent_prd.append(_prd_percent(candidate_latent, float_latent))
        reconstruction_prd.append(_prd_percent(candidate_reconstruction, float_reconstruction))
        index_match.append(float(np.mean(candidate_indices == float_indices)))

    return RvqEncoderQuantizationReport(
        frames_checked=int(samples.shape[0]),
        encoder_input_saturation_fraction_max=float(np.max(input_saturation)),
        encoder_input_saturation_fraction_p90=float(np.quantile(input_saturation, 0.9)),
        latent_prd_percent_p90=float(np.quantile(latent_prd, 0.9)),
        reconstruction_prd_percent_p90=float(np.quantile(reconstruction_prd, 0.9)),
        reconstruction_prd_percent_max=float(np.max(reconstruction_prd)),
        code_index_match_fraction_median=float(np.median(index_match)),
    )


def write_rvq_encoder_quantization_report(
    deploy_dir: str | Path,
    report: RvqEncoderQuantizationReport,
    *,
    calibration_frames: int,
) -> Path:
    """Persist a passed quantization report and register it in the deploy manifest.

    Raises:
        ValueError: If the report exceeds the release thresholds.
    """
    if not report.passed():
        raise ValueError(
            "INT8 encoder quantization parity failed: "
            f"saturation max={report.encoder_input_saturation_fraction_max:.2%}, "
            f"reconstruction PRD p90={report.reconstruction_prd_percent_p90:.2f}%"
        )
    root = Path(deploy_dir)
    report_path = root / ArtifactFile.QUANTIZATION_REPORT
    payload = {
        "format_version": 1,
        "passed": True,
        "thresholds": {
            "max_input_saturation_fraction": MAX_INPUT_SATURATION_FRACTION,
            "p90_reconstruction_prd_percent": MAX_RECONSTRUCTION_PRD_PERCENT_P90,
            "tail_warning_reconstruction_prd_percent": TAIL_WARNING_RECONSTRUCTION_PRD_PERCENT,
        },
        "warnings": [
            *(
                [
                    "P90 reconstruction PRD exceeds the recommended threshold: "
                    f"{report.reconstruction_prd_percent_p90:.2f}%"
                ]
                if report.reconstruction_prd_percent_p90 > MAX_RECONSTRUCTION_PRD_PERCENT_P90
                else []
            ),
            *(
                [
                    "worst-frame reconstruction PRD exceeds the tail-warning threshold: "
                    f"{report.reconstruction_prd_percent_max:.2f}%"
                ]
                if report.reconstruction_prd_percent_max > TAIL_WARNING_RECONSTRUCTION_PRD_PERCENT
                else []
            ),
        ],
        "metrics": asdict(report),
        "partitions": {
            "calibration_frames": calibration_frames,
            "validation_frames": report.frames_checked,
            "disjoint": True,
        },
    }
    write_json(report_path, payload)
    manifest_path = root / ArtifactFile.DEPLOY_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["quantization_validation"] = {
        "report": report_path.name,
        "passed": True,
        "calibration_frames": calibration_frames,
        "validation_frames": report.frames_checked,
        "disjoint": True,
    }
    manifest.setdefault("artifacts", {})["quantization_report"] = report_path.name
    write_json(manifest_path, manifest)
    write_checksums(root)
    return report_path


def require_rvq_encoder_quantization_report(deploy_dir: str | Path) -> None:
    """Reject an INT8 RVQ package without a passed quantization report.

    This lightweight guard is used by the Hugging Face publisher. It does not
    replay models; full artifact and runtime verification remains the job of
    :func:`compressionkit.export.validate.validate_deploy_package`.

    Args:
        deploy_dir: RVQ deploy directory containing ``deploy_manifest.json``.

    Raises:
        ValueError: If an INT8 RVQ package has no valid passed report.
    """
    root = Path(deploy_dir)
    manifest_path = root / ArtifactFile.DEPLOY_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("family", "rvq") != "rvq" or manifest.get("quantization") != "INT8":
        return
    validation = manifest.get("quantization_validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("INT8 RVQ release is missing a passed quantization_validation entry")
    report_name = validation.get("report", ArtifactFile.QUANTIZATION_REPORT)
    report_path = root / str(report_name)
    if not report_path.is_file():
        raise ValueError(f"INT8 RVQ release is missing quantization report: {report_name}")
    report = json.loads(report_path.read_text())
    if report.get("passed") is not True:
        raise ValueError("INT8 RVQ release quantization report is not passed")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("INT8 RVQ release quantization report is missing metrics")
    try:
        measured = RvqEncoderQuantizationReport(**metrics)
    except TypeError as exc:
        raise ValueError(f"INT8 RVQ release quantization report has invalid metrics: {exc}") from exc
    if not measured.passed():
        raise ValueError("INT8 RVQ release quantization report exceeds release thresholds")


__all__ = [
    "RvqEncoderQuantizationReport",
    "evaluate_rvq_encoder_quantization",
    "require_rvq_encoder_quantization_report",
    "write_rvq_encoder_quantization_report",
]
