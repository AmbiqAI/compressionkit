"""Shared helpers for deploy-package release artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON object with stable formatting and return its path."""
    out_path = Path(path)
    with out_path.open("w") as file_obj:
        json.dump(payload, file_obj, indent=2)
    return out_path


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a file."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: str | Path, *, filename: str = "checksums.json") -> Path:
    """Write a checksum manifest for every file under ``output_dir``."""
    root = Path(output_dir)
    checksums: dict[str, dict[str, int | str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == filename:
            continue
        rel_path = path.relative_to(root).as_posix()
        checksums[rel_path] = {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return write_json(root / filename, checksums)


def build_release_metadata(
    *,
    run_name: str,
    modality: str,
    sample_rate: int | float | None = None,
    compression_ratio: int | float | None = None,
    experiment_id: str | None = None,
    dataset_sources: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build common metadata passed from experiments into deploy exporters."""
    metadata: dict[str, Any] = {
        "run_name": run_name,
        "modality": modality,
        "sample_rate": sample_rate,
        "compression_ratio": compression_ratio,
    }
    if experiment_id is not None:
        metadata["experiment_id"] = experiment_id
    if dataset_sources is not None:
        metadata["dataset_sources"] = dataset_sources
    if extra:
        metadata.update(extra)
    return metadata


def build_model_card(
    *,
    model_name: str,
    model_version: str,
    model_card_info: dict[str, Any],
) -> dict[str, Any]:
    """Build the compact deploy-package model card payload."""
    return {
        "model_name": model_name,
        "model_version": model_version,
        "modality": model_card_info.get("modality", "unknown"),
        "sample_rate": model_card_info.get("sample_rate"),
        "compression_ratio": model_card_info.get("compression_ratio"),
        "license": model_card_info.get("license", "other"),
        "dataset_sources": model_card_info.get("dataset_sources"),
        "scorecard_summary": model_card_info.get("scorecard_summary", {}),
    }


def write_model_card(
    output_dir: str | Path,
    *,
    model_name: str,
    model_version: str,
    model_card_info: dict[str, Any],
    filename: str = "model_card.json",
) -> Path:
    """Write a compact deploy-package model card."""
    return write_json(
        Path(output_dir) / filename,
        build_model_card(
            model_name=model_name,
            model_version=model_version,
            model_card_info=model_card_info,
        ),
    )


def write_scorecard_artifact(
    output_dir: str | Path,
    scorecard_summary: dict[str, Any],
    *,
    filename: str = "scorecard.json",
) -> Path:
    """Write a frozen scorecard artifact for a deploy package."""
    return write_json(Path(output_dir) / filename, scorecard_summary)


__all__ = [
    "build_model_card",
    "build_release_metadata",
    "sha256_file",
    "write_checksums",
    "write_json",
    "write_model_card",
    "write_scorecard_artifact",
]
