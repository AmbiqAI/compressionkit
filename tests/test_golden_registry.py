"""Tests for the v1 golden experiment registry."""

from __future__ import annotations

import re

import pytest

from compressionkit.experiments import (
    GOLDEN_REGISTRY,
    GoldenExperiment,
    get_golden,
    list_goldens,
)
from compressionkit.recipes import get_recipe

_HF_REPO_RE = re.compile(r"^Ambiq/compressionkit-(ppg|ecg)-(\d+)x$")
_RUN_NAME_RE = re.compile(r"^(ppg|ecg)_rvq_\d+hz_\d{2}x_golden$")


def test_registry_is_non_empty() -> None:
    assert len(GOLDEN_REGISTRY) > 0


def test_experiment_ids_are_unique() -> None:
    ids = [exp.experiment_id for exp in GOLDEN_REGISTRY]
    assert len(ids) == len(set(ids))


def test_v1_codec_set_present() -> None:
    by_id = {exp.experiment_id for exp in GOLDEN_REGISTRY}
    for cr in (2, 4, 8, 16, 32):
        assert f"ppg-rvq-{cr}x" in by_id
    for cr in (2, 4, 8, 16, 32, 64):
        assert f"ecg-rvq-{cr}x" in by_id


@pytest.mark.parametrize("exp", GOLDEN_REGISTRY, ids=lambda e: e.experiment_id)
def test_entry_invariants(exp: GoldenExperiment) -> None:
    # Config exists relative to repo root (tests run from repo root in CI).
    assert exp.config_path.is_file(), f"missing config for {exp.experiment_id}: {exp.config_path}"

    # Run name and HF repo follow AGENTS.md conventions for codec entries.
    if exp.family == "codec":
        assert _RUN_NAME_RE.match(exp.run_name)
        assert _HF_REPO_RE.match(exp.hf_repo_id)
        modality_in_repo, cr_in_repo = _HF_REPO_RE.match(exp.hf_repo_id).groups()
        assert modality_in_repo == exp.modality
        assert int(cr_in_repo) == exp.compression_ratio

    # Recipe is registered (import side effect of compressionkit.recipes).
    spec = get_recipe(exp.recipe)
    assert spec.name == exp.recipe


def test_list_goldens_filter() -> None:
    ppg = list_goldens("ppg")
    ecg = list_goldens("ecg")
    assert {e.modality for e in ppg} == {"ppg"}
    assert {e.modality for e in ecg} == {"ecg"}
    assert len(ppg) + len(ecg) == len(GOLDEN_REGISTRY)


def test_get_golden_unknown() -> None:
    with pytest.raises(KeyError):
        get_golden("does-not-exist")


def test_two_stage_requires_parent() -> None:
    with pytest.raises(ValueError):
        GoldenExperiment(
            experiment_id="ppg-rvq-4x-prior",
            modality="ppg",
            family="two_stage",
            recipe="train-ppg-rvq",
            config_path=GOLDEN_REGISTRY[0].config_path,
            run_name=GOLDEN_REGISTRY[0].run_name,
            sample_rate=64,
            compression_ratio=4,
            hf_repo_id="Ambiq/compressionkit-ppg-4x",
            dataset_id="mesa",
        )


def test_codec_rejects_parent() -> None:
    with pytest.raises(ValueError):
        GoldenExperiment(
            experiment_id="ppg-rvq-4x-bogus",
            modality="ppg",
            family="codec",
            parent="ppg-rvq-4x",
            recipe="train-ppg-rvq",
            config_path=GOLDEN_REGISTRY[0].config_path,
            run_name="ppg_rvq_64hz_04x_golden",
            sample_rate=64,
            compression_ratio=4,
            hf_repo_id="Ambiq/compressionkit-ppg-4x",
            dataset_id="mesa",
        )
