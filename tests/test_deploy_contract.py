"""Tests for the v1 deploy-package artifact contract."""

from __future__ import annotations

import json

from compressionkit.export.deploy import sync_scorecard_to_deploy
from compressionkit.export.release import write_checksums, write_json
from compressionkit.export.validate import validate_deploy_package


def _write_minimal_manifest(root, *, family: str, scorecard: str | None = "scorecard.json") -> None:
    write_json(
        root / "deploy_manifest.json",
        {
            "manifest_version": 1,
            "package_version": "1.0",
            "family": family,
            "method": "ai" if family == "rvq" else "dsp",
            "modality": "ppg",
            "compression_ratio": 4,
            "spec": "codec_spec.json",
            "scorecard": scorecard,
            "checksums": "checksums.json",
        },
    )
    write_json(root / "codec_spec.json", {"family": family, "modality": "ppg"})


def test_strict_rvq_contract_requires_release_artifacts(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="rvq", scorecard=None)
    (tmp_path / "encoder.tflite").write_bytes(b"encoder")
    (tmp_path / "codebook.npz").write_bytes(b"codebook")
    (tmp_path / "codebook.h").write_text("/* codebook */")
    write_checksums(tmp_path)

    result = validate_deploy_package(
        tmp_path,
        check_runtime=False,
        check_reference_vectors=False,
        strict_release=True,
    )

    assert not result.ok
    assert "missing release artifact: model_card.json" in result.errors
    assert "missing release artifact: scorecard.json" in result.errors
    assert "missing release artifact: reference_vectors.npz" in result.errors
    assert "missing release artifact: sample_data.npz" in result.errors


def test_strict_rvq_contract_accepts_complete_file_set(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="rvq")
    for rel in [
        "encoder.tflite",
        "codebook.npz",
        "codebook.h",
        "model_card.json",
        "scorecard.json",
        "reference_vectors.npz",
        "sample_data.npz",
        "README.md",
    ]:
        (tmp_path / rel).write_bytes(b"artifact")
    write_checksums(tmp_path)

    result = validate_deploy_package(
        tmp_path,
        check_runtime=False,
        check_reference_vectors=False,
        strict_release=True,
    )

    assert result.ok
    assert result.warnings == []
    assert "scorecard.json" in result.checked_files
    assert "sample_data.npz" in result.checked_files


def test_strict_spiht_contract_accepts_complete_file_set(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="spiht")
    for rel in [
        "sample_stimulus.npz",
        "reference_vectors.npz",
        "spiht_app_config.h",
        "model_card.json",
        "scorecard.json",
        "README.md",
    ]:
        (tmp_path / rel).write_bytes(b"artifact")
    write_checksums(tmp_path)

    result = validate_deploy_package(
        tmp_path,
        check_runtime=False,
        check_reference_vectors=False,
        strict_release=True,
    )

    assert result.ok
    assert "sample_stimulus.npz" in result.checked_files


def test_strict_hybrid_contract_accepts_complete_file_set(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="hybrid")
    for rel in [
        "sample_stimulus.npz",
        "reference_vectors.npz",
        "spiht_app_config.h",
        "denoiser_gain_model.keras",
        "denoiser_train_config.json",
        "hybrid_manifest.json",
        "model_card.json",
        "scorecard.json",
        "README.md",
    ]:
        (tmp_path / rel).write_bytes(b"artifact")
    write_checksums(tmp_path)

    result = validate_deploy_package(
        tmp_path,
        check_runtime=False,
        check_reference_vectors=False,
        strict_release=True,
    )

    assert result.ok
    assert "hybrid_manifest.json" in result.checked_files
    assert "denoiser_gain_model.keras" in result.checked_files


def test_checksum_mismatch_is_error(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="rvq")
    for rel in ["encoder.tflite", "codebook.npz", "codebook.h"]:
        (tmp_path / rel).write_bytes(b"artifact")
    write_checksums(tmp_path)
    (tmp_path / "encoder.tflite").write_bytes(b"changed")

    result = validate_deploy_package(
        tmp_path,
        check_runtime=False,
        check_reference_vectors=False,
    )

    assert "checksum mismatch for encoder.tflite" in result.errors


def test_manifest_scorecard_points_to_written_scorecard(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="rvq")
    manifest = json.loads((tmp_path / "deploy_manifest.json").read_text())
    assert manifest["scorecard"] == "scorecard.json"


def test_sync_scorecard_to_deploy_updates_manifest_model_card_and_checksums(tmp_path) -> None:
    _write_minimal_manifest(tmp_path, family="rvq", scorecard=None)
    write_json(
        tmp_path / "model_card.json",
        {
            "model_name": "ppg_rvq_64hz_04x_golden",
            "model_version": "1.0",
            "scorecard_summary": {},
        },
    )
    write_checksums(tmp_path)

    scorecard = {"time_domain": {"prd_percent": {"mean": 2.5}}}
    scorecard_path = sync_scorecard_to_deploy(tmp_path, scorecard)

    assert scorecard_path.name == "scorecard.json"
    assert json.loads(scorecard_path.read_text()) == scorecard

    manifest = json.loads((tmp_path / "deploy_manifest.json").read_text())
    assert manifest["scorecard"] == "scorecard.json"
    assert manifest["artifacts"]["scorecard"] == "scorecard.json"

    model_card = json.loads((tmp_path / "model_card.json").read_text())
    assert model_card["scorecard_summary"] == scorecard

    checksums = json.loads((tmp_path / "checksums.json").read_text())
    assert "scorecard.json" in checksums


def _write_hybrid_reference_vectors(root, *, frame, payload, recon) -> None:
    import numpy as np

    np.savez_compressed(
        root / "reference_vectors.npz",
        input_frames=frame[None, :],
        bitstreams=payload[None, :],
        bitstream_lengths_bytes=np.array([payload.size], dtype=np.int32),
        nbits=np.array([payload.size * 8], dtype=np.int32),
        reconstructions=recon[None, :],
    )


class _FakeHybridCodec:
    """Minimal Codec stand-in for exercising the hybrid reference-vector check."""

    def __init__(self, *, payload_size: int, decoded) -> None:
        self._payload_size = payload_size
        self._decoded = decoded

    def compress(self, frame):
        from compressionkit.runtime.base import EncodedFrame

        return EncodedFrame(payload=bytes(range(self._payload_size)), nbits=self._payload_size * 8)

    def decompress(self, encoded):
        return self._decoded


def test_hybrid_reference_vectors_tolerate_tiny_numeric_drift(tmp_path, monkeypatch) -> None:
    """GPU vs CPU denoiser jitter should NOT fail hybrid strict validation (issue #46)."""
    import numpy as np

    from compressionkit.export import validate as validate_mod

    frame = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    payload = np.arange(16, dtype=np.uint8)
    recon = frame.copy()
    # A tiny (~1e-5) numeric perturbation, representative of GPU/CPU float drift.
    decoded = recon + 1e-5

    _write_hybrid_reference_vectors(tmp_path, frame=frame, payload=payload, recon=recon)
    monkeypatch.setattr(
        "compressionkit.runtime.load_codec",
        lambda deploy_dir: _FakeHybridCodec(payload_size=16, decoded=decoded),
    )

    errors: list[str] = []
    warnings: list[str] = []
    validate_mod._validate_reference_vectors(tmp_path, "hybrid", errors, warnings, max_vectors=1)

    assert errors == []


def test_hybrid_reference_vectors_still_catch_real_regressions(tmp_path, monkeypatch) -> None:
    """A large systematic shift (e.g. the DC-shift denoiser bug) must still fail."""
    import numpy as np

    from compressionkit.export import validate as validate_mod

    frame = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    payload = np.arange(16, dtype=np.uint8)
    recon = frame.copy()
    # A large systematic offset, like the approx-band DC-shift bug (~48% PRD).
    decoded = recon - 0.5

    _write_hybrid_reference_vectors(tmp_path, frame=frame, payload=payload, recon=recon)
    monkeypatch.setattr(
        "compressionkit.runtime.load_codec",
        lambda deploy_dir: _FakeHybridCodec(payload_size=16, decoded=decoded),
    )

    errors: list[str] = []
    warnings: list[str] = []
    validate_mod._validate_reference_vectors(tmp_path, "hybrid", errors, warnings, max_vectors=1)

    assert any("PRD" in e for e in errors)


def test_hybrid_reference_vectors_flag_grossly_different_bitstream_length(tmp_path, monkeypatch) -> None:
    import numpy as np

    from compressionkit.export import validate as validate_mod

    frame = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    payload = np.arange(16, dtype=np.uint8)
    recon = frame.copy()

    _write_hybrid_reference_vectors(tmp_path, frame=frame, payload=payload, recon=recon)
    monkeypatch.setattr(
        "compressionkit.runtime.load_codec",
        lambda deploy_dir: _FakeHybridCodec(payload_size=64, decoded=recon),  # far larger than expected 16 bytes
    )

    errors: list[str] = []
    warnings: list[str] = []
    validate_mod._validate_reference_vectors(tmp_path, "hybrid", errors, warnings, max_vectors=1)

    assert any("bitstream length" in e for e in errors)


def test_spiht_reference_vectors_remain_bit_exact(tmp_path, monkeypatch) -> None:
    """Pure DSP SPIHT is deterministic — it should NOT get the hybrid tolerance."""
    import numpy as np

    from compressionkit.export import validate as validate_mod

    frame = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    payload = np.arange(16, dtype=np.uint8)
    recon = frame.copy()
    # Even a tiny drift must fail for SPIHT, since it has no floating-point model stage.
    decoded = recon + 1e-5

    _write_hybrid_reference_vectors(tmp_path, frame=frame, payload=payload, recon=recon)
    monkeypatch.setattr(
        "compressionkit.runtime.load_codec",
        lambda deploy_dir: _FakeHybridCodec(payload_size=16, decoded=decoded),
    )

    errors: list[str] = []
    warnings: list[str] = []
    validate_mod._validate_reference_vectors(tmp_path, "spiht", errors, warnings, max_vectors=1)

    assert any("reconstruction mismatch" in e for e in errors)
