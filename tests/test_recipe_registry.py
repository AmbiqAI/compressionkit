"""Tests for the recipe registry and ``@recipe`` decorator."""

from __future__ import annotations

import pytest

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.configs.ppg_rvq import PpgRvqConfig
from compressionkit.recipes import (
    RecipeSpec,
    dispatch,
    get_recipe,
    list_recipes,
    recipe,
)
from compressionkit.recipes import train_ecg_rvq as ecg_mod
from compressionkit.recipes import train_ppg_rvq as ppg_mod


def test_golden_recipes_registered() -> None:
    names = {spec.name for spec in list_recipes()}
    assert {"train-ppg-rvq", "train-ecg-rvq"}.issubset(names)


def test_get_recipe_returns_expected_spec() -> None:
    spec = get_recipe("train-ppg-rvq")
    assert isinstance(spec, RecipeSpec)
    assert spec.config_cls is PpgRvqConfig
    assert spec.train_fn is ppg_mod.train

    spec = get_recipe("train-ecg-rvq")
    assert spec.config_cls is EcgRvqConfig
    assert spec.train_fn is ecg_mod.train


def test_decorator_attaches_main_to_module() -> None:
    assert callable(getattr(ppg_mod, "main", None))
    assert callable(getattr(ecg_mod, "main", None))


def test_duplicate_registration_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @recipe("train-ppg-rvq", config_cls=PpgRvqConfig)
        def _dup(cfg: PpgRvqConfig) -> None:  # pragma: no cover
            pass


def test_dispatch_list_prints_recipes(capsys: pytest.CaptureFixture[str]) -> None:
    rc = dispatch(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "train-ppg-rvq" in out
    assert "train-ecg-rvq" in out
