"""Tests for two-stage golden family registration and prior config (#27)."""

from __future__ import annotations

from pathlib import Path

import pytest

from compressionkit.configs.rvq_prior import RvqPriorConfig
from compressionkit.experiments.registry import (
    GOLDEN_REGISTRY,
    get_golden,
    list_two_stage_children,
)
from compressionkit.recipes import get_recipe

_TWO_STAGE_EXPERIMENTS = [exp for exp in GOLDEN_REGISTRY if exp.structure == "two_stage"]


def test_two_stage_entries_registered() -> None:
    ids = {exp.experiment_id for exp in _TWO_STAGE_EXPERIMENTS}
    assert ids == {"ppg-rvq-4x-prior", "ppg-rvq-8x-prior", "ecg-rvq-4x-prior", "ecg-rvq-8x-prior"}


@pytest.mark.parametrize("exp", _TWO_STAGE_EXPERIMENTS, ids=lambda e: e.experiment_id)
def test_two_stage_entry_invariants(exp) -> None:
    # Parent exists and is a codec.
    parent = get_golden(exp.parent)
    assert parent.structure == "codec"
    assert parent.modality == exp.modality
    assert parent.compression_ratio == exp.compression_ratio
    # Bundled HF repo: same as parent codec.
    assert exp.hf_repo_id == parent.hf_repo_id
    # Bundled run dir: same as parent codec.
    assert exp.run_name == parent.run_name
    # Recipe is registered.
    assert get_recipe(exp.recipe).name == exp.recipe
    # Config path is on disk.
    assert exp.config_path.is_file()


@pytest.mark.parametrize("exp", _TWO_STAGE_EXPERIMENTS, ids=lambda e: e.experiment_id)
def test_prior_yaml_round_trip(exp) -> None:
    cfg = RvqPriorConfig.from_yaml(exp.config_path)
    assert cfg.parent_experiment == exp.parent
    assert cfg.arch.embed_dim > 0
    assert cfg.training.epochs > 0


def test_list_two_stage_children() -> None:
    children = list_two_stage_children("ecg-rvq-4x")
    assert {c.experiment_id for c in children} == {"ecg-rvq-4x-prior"}
    assert list_two_stage_children("ecg-rvq-2x") == []


def test_prior_config_rejects_extra() -> None:
    with pytest.raises(ValueError):
        RvqPriorConfig.model_validate({"parent_experiment": "ecg-rvq-4x", "unknown_field": 1})


def test_prior_config_resolve_parent_run_dir_default(tmp_path: Path) -> None:
    cfg = RvqPriorConfig(parent_experiment="ecg-rvq-4x")
    assert cfg.parent_run_dir is None
    cfg2 = cfg.model_copy(update={"parent_run_dir": tmp_path / "ecg_rvq_256hz_04x_golden"})
    assert cfg2.parent_run_dir.name == "ecg_rvq_256hz_04x_golden"
