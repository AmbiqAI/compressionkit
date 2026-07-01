"""Tests for the dataset acquisition contract (issue #26)."""

from __future__ import annotations

from pathlib import Path

import pytest

from compressionkit.datasets import (
    DATASET_REGISTRY,
    DatasetNotAvailableError,
    MesaDataset,
    PtbxlDataset,
    ensure_dataset_available,
    resolve_dataset,
)
from compressionkit.experiments.registry import GOLDEN_REGISTRY


def test_license_tier_is_set_for_shipped_datasets() -> None:
    from compressionkit.datasets.mesa import INFO as mesa_info
    from compressionkit.datasets.ptbxl import INFO as ptbxl_info

    assert ptbxl_info.license_tier == "open"
    assert mesa_info.license_tier == "restricted"


def test_registry_keys_match_known_dataset_ids() -> None:
    assert set(DATASET_REGISTRY) == {"mesa", "ptb-xl", "ppg-unified-strict-sanitize-v1"}


def test_every_golden_experiment_resolves_through_registry(tmp_path: Path) -> None:
    for exp in GOLDEN_REGISTRY:
        ds = resolve_dataset(exp.dataset_id, root=tmp_path)
        assert ds.path.parent == tmp_path


def test_resolve_dataset_rejects_unknown_id() -> None:
    with pytest.raises(KeyError):
        resolve_dataset("nonexistent")


def test_ensure_available_ptbxl_missing_raises(tmp_path: Path) -> None:
    ds = PtbxlDataset(path=tmp_path / "ptbxl-empty")
    with pytest.raises(DatasetNotAvailableError) as exc:
        ds.ensure_available()
    assert "PtbxlDataset" in exc.value.remediation


def test_ensure_available_mesa_missing_raises(tmp_path: Path) -> None:
    ds = MesaDataset(path=tmp_path / "mesa-empty")
    with pytest.raises(DatasetNotAvailableError) as exc:
        ds.ensure_available()
    assert "NSRR_TOKEN" in exc.value.remediation


def test_ensure_available_ptbxl_present(tmp_path: Path) -> None:
    root = tmp_path / "ptbxl"
    root.mkdir()
    (root / "fold_1.h5").write_bytes(b"")
    PtbxlDataset(path=root).ensure_available()


def test_ensure_available_mesa_present(tmp_path: Path) -> None:
    root = tmp_path / "mesa"
    (root / "polysomnography" / "edfs").mkdir(parents=True)
    (root / "polysomnography" / "edfs" / "x.edf").write_bytes(b"")
    MesaDataset(path=root).ensure_available()


def test_ensure_dataset_available_helper(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotAvailableError):
        ensure_dataset_available("ptb-xl", root=tmp_path)


def test_ensure_available_ppg_unified_strict_sanitize_present(tmp_path: Path) -> None:
    root = tmp_path / "ppg_cache_strict_sanitize"
    for slug in ("bidmc", "butppg", "ppg_dalia", "wesad"):
        source_dir = root / slug
        source_dir.mkdir(parents=True)
        (source_dir / "train.tfrecord").write_bytes(b"")
        (source_dir / "val.tfrecord").write_bytes(b"")
        (source_dir / "metadata.json").write_text("{}")

    ensure_dataset_available("ppg-unified-strict-sanitize-v1", root=tmp_path)
