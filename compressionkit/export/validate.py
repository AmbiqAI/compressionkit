"""Deploy-package validation helpers.

Checks package integrity (manifest, spec, checksums), runtime hydration,
and optional reference-vector conformance for both RVQ and SPIHT deploy
artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from compressionkit.export.artifact_contract import ArtifactFile, SampleArray
from compressionkit.export.release import sha256_file

__all__ = ["DeployValidationResult", "validate_deploy_package"]


@dataclass(frozen=True)
class DeployValidationResult:
    """Structured result from deploy-package validation."""

    deploy_dir: Path
    family: str
    errors: list[str]
    warnings: list[str]
    checked_files: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _verify_checksums(deploy_dir: Path, checksum_path: Path, errors: list[str]) -> dict[str, Any]:
    checksums = _load_json(checksum_path)
    for rel_path, meta in checksums.items():
        path = deploy_dir / rel_path
        if not path.is_file():
            errors.append(f"checksums.json references missing file: {rel_path}")
            continue
        if not isinstance(meta, dict):
            errors.append(f"checksums.json entry for {rel_path} is not an object")
            continue
        expected_sha = meta.get("sha256")
        if expected_sha and sha256_file(path) != expected_sha:
            errors.append(f"checksum mismatch for {rel_path}")
    return checksums


def _check_file(
    root: Path,
    rel_path: str,
    checked_files: list[str],
    errors: list[str],
    warnings: list[str],
    *,
    required: bool,
    label: str,
) -> None:
    if (root / rel_path).is_file():
        checked_files.append(rel_path)
    elif required:
        errors.append(f"missing {label}: {rel_path}")
    else:
        warnings.append(f"missing {label}: {rel_path}")


def _prd_percent(actual: np.ndarray, expected: np.ndarray) -> float:
    """Percent RMS difference of ``actual`` vs. ``expected`` (0 = identical)."""
    err = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    denom = float(np.sum(np.asarray(expected, dtype=np.float64) ** 2))
    if denom <= 0:
        return 0.0 if np.allclose(err, 0.0) else float("inf")
    return 100.0 * float(np.sqrt(np.sum(err**2) / denom))


def _validate_reference_vectors(
    deploy_dir: Path,
    family: str,
    errors: list[str],
    warnings: list[str],
    *,
    max_vectors: int,
    require_training_reference: bool = False,
) -> None:
    ref_path = deploy_dir / ArtifactFile.REFERENCE_VECTORS
    if not ref_path.exists():
        warnings.append(f"{ArtifactFile.REFERENCE_VECTORS} not present")
        return

    from compressionkit.runtime import load_codec

    codec = load_codec(deploy_dir)
    blob = np.load(ref_path)
    if family in {"spiht", "hybrid"}:
        frames = np.asarray(blob[SampleArray.INPUT_FRAMES], dtype=np.float32)
        payloads = np.asarray(blob["bitstreams"], dtype=np.uint8)
        lengths = np.asarray(blob["bitstream_lengths_bytes"], dtype=np.int32)
        nbits = np.asarray(blob["nbits"], dtype=np.int32)
        recon = np.asarray(blob[SampleArray.RECONSTRUCTIONS], dtype=np.float32)
        for idx in range(min(max_vectors, frames.shape[0])):
            encoded = codec.compress(frames[idx])
            actual = np.frombuffer(bytes(encoded.payload), dtype=np.uint8)
            expected = payloads[idx, : lengths[idx]]
            decoded = codec.decompress(encoded)
            if family == "spiht":
                # Pure DSP: bit-exact and deterministic regardless of hardware.
                if encoded.nbits != int(nbits[idx]):
                    errors.append(f"SPIHT reference nbits mismatch at sample {idx}")
                if actual.shape != expected.shape or not np.array_equal(actual, expected):
                    errors.append(f"SPIHT reference bitstream mismatch at sample {idx}")
                if not np.allclose(decoded, recon[idx], atol=1e-6):
                    errors.append(f"SPIHT reference reconstruction mismatch at sample {idx}")
            else:
                # Hybrid combines a neural denoiser (backend-dependent floating
                # point — not bit-exact across GPU/CPU/library versions) with a
                # SPIHT bitstream stage that is bit-exact on its float input. A
                # ~1e-6 level difference in the denoiser's output can flip a
                # quantization/coding decision at a boundary and change the
                # exact encoded bytes even though the reconstructed signal is
                # practically identical (see issue #46) — so hybrid gets a
                # numeric-tolerance check (bitstream length + reconstruction
                # fidelity) instead of a bit-exact byte comparison.
                length_tolerance = max(4, round(0.05 * int(lengths[idx])))
                if abs(actual.size - int(lengths[idx])) > length_tolerance:
                    errors.append(
                        f"HYBRID reference bitstream length differs by more than "
                        f"{length_tolerance} bytes at sample {idx} "
                        f"(expected ~{int(lengths[idx])}, got {actual.size})"
                    )
                prd = _prd_percent(decoded, recon[idx])
                if prd > 5.0:
                    errors.append(
                        f"HYBRID reference reconstruction differs by {prd:.2f}% PRD at sample {idx} (tolerance 5%)"
                    )
        return

    if family == "rvq":
        source_keys = {"source_latents", "source_indices", "source_quantized_latents", "source_reconstructions"}
        has_source = source_keys.issubset(blob.files)
        if not has_source:
            message = "RVQ reference vectors lack independent trained-quantizer references; re-export required"
            (errors if require_training_reference else warnings).append(message)
        frames = np.asarray(blob[SampleArray.INPUT_FRAMES], dtype=np.float32)
        indices = np.asarray(blob[SampleArray.INDICES], dtype=np.int32)
        recon = np.asarray(blob[SampleArray.RECONSTRUCTIONS], dtype=np.float32)
        if frames.shape[0] == 0:
            errors.append("RVQ reference vectors contain no frames")
            return
        if has_source:
            latent_shape = blob["source_latents"].shape
            if (
                latent_shape[0] != frames.shape[0]
                or blob["source_quantized_latents"].shape != latent_shape
                or blob["source_indices"].shape != (*latent_shape[:-1], codec.num_levels)
                or blob["source_reconstructions"].shape != recon.shape
            ):
                errors.append("RVQ trained-quantizer reference shapes do not match the deployment contract")
                return
        for idx in range(min(max_vectors, frames.shape[0])):
            sample = frames[idx : idx + 1]
            actual_indices = codec.encode(sample)
            if actual_indices.shape != indices[idx : idx + 1].shape or not np.array_equal(
                actual_indices, indices[idx : idx + 1]
            ):
                errors.append(f"RVQ reference index mismatch at sample {idx}")
            actual_recon = codec.decode(actual_indices)
            if not np.allclose(actual_recon, recon[idx : idx + 1], atol=1e-5):
                errors.append(f"RVQ reference reconstruction mismatch at sample {idx}")
            if has_source:
                source_latent = blob["source_latents"][idx : idx + 1]
                deployed_latent = codec.encode_latent(sample)
                if deployed_latent.shape != source_latent.shape or not np.allclose(
                    deployed_latent, source_latent, atol=1e-5, rtol=1e-5
                ):
                    errors.append(f"RVQ trained-encoder latent mismatch at sample {idx} (atol=rtol=1e-5)")
                source_indices = blob["source_indices"][idx : idx + 1]
                actual_source_indices = codec.quantize_latent(source_latent)
                if not np.array_equal(actual_source_indices, source_indices):
                    errors.append(f"RVQ trained-quantizer index mismatch at sample {idx}")
                    continue
                actual_latent = codec.dequantize_indices(source_indices)
                if not np.array_equal(actual_latent, blob["source_quantized_latents"][idx : idx + 1]):
                    errors.append(f"RVQ trained-quantizer latent mismatch at sample {idx}")
                source_recon = codec.decode_latent(actual_latent)
                if not np.allclose(source_recon, blob["source_reconstructions"][idx : idx + 1], atol=1e-5, rtol=1e-5):
                    errors.append(f"RVQ trained-decoder reconstruction mismatch at sample {idx} (atol=rtol=1e-5)")
        return

    warnings.append(f"No reference-vector validator registered for family {family!r}")


def validate_deploy_package(
    deploy_dir: str | Path,
    *,
    check_runtime: bool = True,
    check_reference_vectors: bool = True,
    strict_release: bool = False,
    max_vectors: int = 2,
) -> DeployValidationResult:
    """Validate a deploy package on disk.

    Args:
        deploy_dir: Directory containing deploy artifacts.
        check_runtime: Hydrate the runtime to confirm the package loads.
        check_reference_vectors: Validate stored reference vectors when present.
        strict_release: Treat release-contract extras like model cards and
            frozen scorecards as required.
        max_vectors: Max reference-vector samples to replay.
    """
    root = Path(deploy_dir)
    errors: list[str] = []
    warnings: list[str] = []
    checked_files: list[str] = []

    manifest_path = root / "deploy_manifest.json"
    if not manifest_path.is_file():
        return DeployValidationResult(root, "unknown", [f"missing deploy_manifest.json in {root}"], [], [])
    manifest = _load_json(manifest_path)
    checked_files.append("deploy_manifest.json")

    family = str(manifest.get("family", "rvq"))
    spec_name = str(manifest.get("spec", "codec_spec.json"))
    spec_path = root / spec_name
    if not spec_path.is_file():
        errors.append(f"missing codec spec: {spec_name}")
    else:
        _load_json(spec_path)
        checked_files.append(spec_name)

    checksums_name = str(manifest.get("checksums", "checksums.json"))
    checksums_path = root / checksums_name
    if not checksums_path.is_file():
        errors.append(f"missing checksums file: {checksums_name}")
    else:
        _verify_checksums(root, checksums_path, errors)
        checked_files.append(checksums_name)

    family_spec = None
    try:
        from compressionkit.export.family_registry import get_family_spec

        family_spec = get_family_spec(family)
    except ValueError:
        pass  # Unknown family: fall through to the generic release-extras default below.

    for rel in family_spec.required_artifacts if family_spec is not None else ():
        _check_file(root, rel, checked_files, errors, warnings, required=True, label=f"required {family} artifact")

    default_release_extras = ("model_card.json", "README.md", "scorecard.json", "reference_vectors.npz")
    for rel in family_spec.release_extras if family_spec is not None else default_release_extras:
        _check_file(root, rel, checked_files, errors, warnings, required=strict_release, label="release artifact")

    # Real browser-demo recordings are attached in a separate, source-data
    # release step. They are optional for a clean model export, but once the
    # manifest advertises them both artifacts are part of the package contract.
    demo_recordings = manifest.get("demo_recordings")
    if demo_recordings is not None:
        if not isinstance(demo_recordings, dict):
            errors.append("demo_recordings manifest entry must be an object")
        else:
            _check_file(
                root,
                str(demo_recordings.get("npz", "demo_recordings.npz")),
                checked_files,
                errors,
                warnings,
                required=True,
                label="demo recordings artifact",
            )
            _check_file(
                root,
                str(demo_recordings.get("manifest", "demo_recordings_manifest.json")),
                checked_files,
                errors,
                warnings,
                required=True,
                label="demo recordings artifact",
            )

    quantization_validation = manifest.get("quantization_validation")
    if quantization_validation is not None:
        if not isinstance(quantization_validation, dict):
            errors.append("quantization_validation manifest entry must be an object")
        else:
            report_name = str(quantization_validation.get("report", "quantization_report.json"))
            report_path = root / report_name
            _check_file(
                root,
                report_name,
                checked_files,
                errors,
                warnings,
                required=True,
                label="quantization validation report",
            )
            if report_path.is_file():
                try:
                    report = _load_json(report_path)
                    if report.get("passed") is not True:
                        errors.append("quantization validation report is not passed")
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid quantization validation report: {exc}")
    elif family == "rvq" and manifest.get("quantization") == "INT8" and strict_release:
        errors.append("missing quantization_validation for INT8 RVQ release")

    if family == "rvq" and manifest.get("quantization") == "INT8":
        try:
            from compressionkit.export.quantization import require_rvq_encoder_quantization_report

            require_rvq_encoder_quantization_report(root)
        except ValueError as exc:
            errors.append(str(exc))

    if check_runtime:
        try:
            from compressionkit.runtime import load_codec

            load_codec(root)
        except Exception as exc:
            errors.append(f"runtime hydration failed: {exc}")

    if check_reference_vectors and not errors:
        try:
            _validate_reference_vectors(
                root, family, errors, warnings, max_vectors=max_vectors, require_training_reference=strict_release
            )
        except Exception as exc:
            errors.append(f"reference-vector validation failed: {exc}")

    return DeployValidationResult(
        deploy_dir=root,
        family=family,
        errors=errors,
        warnings=warnings,
        checked_files=sorted(set(checked_files)),
    )
