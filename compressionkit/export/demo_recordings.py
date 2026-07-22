"""Package curated real recordings as stable deployment artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from compressionkit.export.artifact_contract import ArtifactFile, DemoArray
from compressionkit.export.release import write_checksums, write_json


class _DemoClip(Protocol):
    """Minimal common interface implemented by ECG and PPG demo clips."""

    signal: np.ndarray
    quality: object
    source_record: str
    source_sample_rate: int
    start_seconds: float


@dataclass(frozen=True)
class DemoRecordingsExport:
    """Paths written for a portable browser-demo recording bundle."""

    manifest_path: Path
    recordings_path: Path


def export_demo_recordings(
    output_dir: str | Path,
    *,
    clips: list[_DemoClip],
    modality: str,
    sample_rate: int,
    source: dict[str, str],
    seed: int,
) -> DemoRecordingsExport:
    """Write a compact real-recordings NPZ and provenance manifest.

    The recordings are intentionally raw source waveforms after resampling, so
    a web application can apply the same framing and normalization policy as
    its selected codec. Every clip must share output length and sample rate.

    Args:
        output_dir: Destination deploy directory or standalone output directory.
        clips: Quality-gated real demo clips from one modality.
        modality: ``"ecg"`` or ``"ppg"``.
        sample_rate: Output signal rate in Hz.
        source: Dataset attribution and license fields.
        seed: Deterministic source-selection seed for the manifest.

    Returns:
        Paths of the generated NPZ and JSON manifest.
    """
    if modality not in {"ecg", "ppg"}:
        raise ValueError(f"Unsupported demo modality {modality!r}")
    if not clips:
        raise ValueError("At least one demo clip is required")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    signals = np.stack([np.asarray(clip.signal, dtype=np.float32).reshape(-1) for clip in clips])
    if signals.ndim != 2:
        raise ValueError("Demo clips must be one-dimensional and have equal lengths")

    recordings_path = destination / ArtifactFile.DEMO_RECORDINGS
    np.savez_compressed(
        recordings_path,
        **{
            DemoArray.SIGNALS: signals,
            DemoArray.SAMPLE_RATE: np.asarray(sample_rate, dtype=np.int32),
            DemoArray.SOURCE_RECORDS: np.asarray([clip.source_record for clip in clips]),
            DemoArray.SOURCE_SAMPLE_RATES: np.asarray([clip.source_sample_rate for clip in clips], dtype=np.int32),
            DemoArray.START_SECONDS: np.asarray([clip.start_seconds for clip in clips], dtype=np.float32),
        },
    )
    manifest = {
        "format_version": 1,
        "modality": modality,
        "sample_rate": sample_rate,
        "num_recordings": len(clips),
        "samples_per_recording": int(signals.shape[1]),
        "duration_seconds": signals.shape[1] / float(sample_rate),
        "signal_representation": "raw source waveform resampled to sample_rate",
        "framing_note": "Apply the selected codec's framing and normalization policy before inference.",
        "source": source,
        "seed": seed,
        "recordings": [
            {
                "index": index,
                "source_record": clip.source_record,
                "source_sample_rate": clip.source_sample_rate,
                "start_seconds": clip.start_seconds,
                "quality": asdict(clip.quality),
            }
            for index, clip in enumerate(clips)
        ],
    }
    manifest_path = destination / ArtifactFile.DEMO_RECORDINGS_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return DemoRecordingsExport(manifest_path=manifest_path, recordings_path=recordings_path)


def attach_demo_recordings_to_deploy(
    deploy_dir: str | Path,
    bundle: DemoRecordingsExport,
) -> None:
    """Register an existing demo bundle in a deploy manifest and checksums."""
    destination = Path(deploy_dir)
    manifest_path = destination / ArtifactFile.DEPLOY_MANIFEST
    if bundle.recordings_path.parent != destination or bundle.manifest_path.parent != destination:
        raise ValueError("Demo bundle must be written directly inside the deploy directory")
    manifest = json.loads(manifest_path.read_text())
    recordings = json.loads(bundle.manifest_path.read_text())
    manifest["demo_recordings"] = {
        "npz": bundle.recordings_path.name,
        "manifest": bundle.manifest_path.name,
        "num_recordings": recordings["num_recordings"],
        "sample_rate": recordings["sample_rate"],
        "duration_seconds": recordings["duration_seconds"],
    }
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["demo_recordings"] = bundle.recordings_path.name
    artifacts["demo_recordings_manifest"] = bundle.manifest_path.name
    write_json(manifest_path, manifest)
    write_checksums(destination)


__all__ = ["DemoRecordingsExport", "attach_demo_recordings_to_deploy", "export_demo_recordings"]
