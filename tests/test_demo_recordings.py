"""Tests for deployable real-recordings demo bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from compressionkit.export.artifact_contract import DemoArray
from compressionkit.export.demo_recordings import attach_demo_recordings_to_deploy, export_demo_recordings


@dataclass(frozen=True)
class _Quality:
    score: float


@dataclass(frozen=True)
class _Clip:
    signal: np.ndarray
    quality: _Quality
    source_record: str
    source_sample_rate: int
    start_seconds: float


def test_export_and_attach_demo_recordings(tmp_path) -> None:
    """The public bundle has stable arrays, attribution, manifest registration, and checksums."""
    deploy_dir = tmp_path / "deploy"
    deploy_dir.mkdir()
    (deploy_dir / "deploy_manifest.json").write_text(json.dumps({"artifacts": {}}))
    clips = [
        _Clip(np.arange(20, dtype=np.float32), _Quality(score=0.9), "subject-a", 125, 3.5),
        _Clip(np.arange(20, dtype=np.float32) + 1, _Quality(score=0.95), "subject-b", 125, 6.5),
    ]

    bundle = export_demo_recordings(
        deploy_dir,
        clips=clips,
        modality="ppg",
        sample_rate=64,
        source={"dataset": "BIDMC", "license": "ODC-By-1.0"},
        seed=42,
    )
    attach_demo_recordings_to_deploy(deploy_dir, bundle)

    with np.load(bundle.recordings_path) as recordings:
        assert recordings[DemoArray.SIGNALS].shape == (2, 20)
        assert int(recordings[DemoArray.SAMPLE_RATE]) == 64
        assert recordings[DemoArray.SOURCE_RECORDS].tolist() == ["subject-a", "subject-b"]
    manifest = json.loads(bundle.manifest_path.read_text())
    assert manifest["source"]["license"] == "ODC-By-1.0"
    assert manifest["recordings"][0]["quality"]["score"] == 0.9
    deploy_manifest = json.loads((deploy_dir / "deploy_manifest.json").read_text())
    assert deploy_manifest["artifacts"]["demo_recordings"] == "demo_recordings.npz"
    assert (deploy_dir / "checksums.json").is_file()
