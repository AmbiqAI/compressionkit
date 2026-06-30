"""Recipe registry and ``@recipe`` decorator.

The decorator tags a recipe's ``train`` function with a CLI name and a
Pydantic config class. Doing so:

1. Stores the recipe in a process-wide registry, which drives the
   ``compressionkit`` multiplexing CLI and lets tests enumerate recipes.
2. Attaches an auto-generated ``main()`` to the decorated function's
   module, so ``pyproject.toml`` can point a console script directly at
   ``compressionkit.recipes.my_recipe:main`` without hand-written argparse.

Example::

    from compressionkit.configs.ppg_rvq import PpgRvqConfig
    from compressionkit.recipes import recipe

    @recipe("train-ppg-rvq", config_cls=PpgRvqConfig)
    def train(cfg: PpgRvqConfig) -> dict:
        ...
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecipeSpec:
    """Registered recipe metadata."""

    name: str
    train_fn: Callable[[Any], Any]
    config_cls: type
    description: str
    module: str


_REGISTRY: dict[str, RecipeSpec] = {}


def recipe[ConfigT](
    name: str,
    *,
    config_cls: type[ConfigT],
) -> Callable[[Callable[[ConfigT], Any]], Callable[[ConfigT], Any]]:
    """Register a training function as a named recipe.

    Args:
        name: CLI-style name (e.g. ``"train-ppg-rvq"``). Must be unique.
        config_cls: Pydantic config class; must expose a ``from_yaml`` classmethod.

    Returns:
        A decorator that registers the function and attaches ``main`` to the
        decorated function's module.
    """

    def decorator(train_fn: Callable[[ConfigT], Any]) -> Callable[[ConfigT], Any]:
        if name in _REGISTRY and _REGISTRY[name].train_fn is not train_fn:
            raise ValueError(f"Recipe {name!r} already registered by {_REGISTRY[name].module}")

        description = (train_fn.__doc__ or "").strip().splitlines()[0] if train_fn.__doc__ else ""
        spec = RecipeSpec(
            name=name,
            train_fn=train_fn,
            config_cls=config_cls,
            description=description,
            module=train_fn.__module__,
        )
        _REGISTRY[name] = spec

        def main(argv: list[str] | None = None) -> int:
            return _run_single(spec, argv)

        main.__doc__ = f"Console entry point for recipe {name!r}."

        # Expose ``main`` on the recipe's module so pyproject.toml can wire it
        # up as a console script without any boilerplate in the recipe file.
        module = sys.modules.get(train_fn.__module__)
        if module is not None:
            module.main = main  # type: ignore[attr-defined]
        train_fn._recipe_main = main  # type: ignore[attr-defined]
        return train_fn

    return decorator


def get_recipe(name: str) -> RecipeSpec:
    """Look up a registered recipe by name."""
    try:
        return _REGISTRY[name]
    except KeyError as err:
        raise KeyError(f"Unknown recipe {name!r}. Known: {sorted(_REGISTRY)}") from err


def list_recipes() -> list[RecipeSpec]:
    """Return every registered recipe, sorted by name."""
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def _build_config_parser(spec: RecipeSpec, parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML configuration file.")


def _run_single(spec: RecipeSpec, argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(prog=spec.name, description=spec.description or None)
    _build_config_parser(spec, parser)
    args = parser.parse_args(argv)
    cfg = spec.config_cls.from_yaml(str(args.config))  # type: ignore[attr-defined]
    spec.train_fn(cfg)
    return 0


def dispatch(argv: list[str] | None = None) -> int:
    """Multiplexing CLI: ``compressionkit <recipe> --config …``.

    Imports :mod:`compressionkit.recipes` to trigger registration of all
    package-shipped recipes, then dispatches to the named recipe.
    """
    import compressionkit.recipes  # noqa: F401  — import side effect: registers recipes

    parser = argparse.ArgumentParser(
        prog="compressionkit",
        description="Run a registered compressionkit recipe.",
    )
    sub = parser.add_subparsers(dest="recipe", metavar="RECIPE", required=True)
    sub.add_parser("list", help="List all registered recipes.")
    # `golden` forwards every remaining argument to the experiments CLI.
    golden = sub.add_parser("golden", help="Run v1 golden experiments (see 'compressionkit golden --help').")
    golden.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    # `playbook` forwards every remaining argument to the playbook CLI.
    playbook = sub.add_parser(
        "playbook", help="Browse and run the compression playbook (see 'compressionkit playbook --help')."
    )
    playbook.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    for spec in list_recipes():
        sp = sub.add_parser(spec.name, help=spec.description or None, description=spec.description or None)
        _build_config_parser(spec, sp)

    args = parser.parse_args(argv)
    if args.recipe == "list":
        for spec in list_recipes():
            print(f"{spec.name:30s}  {spec.description}")
        return 0
    if args.recipe == "golden":
        from compressionkit.experiments.cli import main as golden_main

        return golden_main(args.args)
    if args.recipe == "playbook":
        from compressionkit.playbook.cli import main as playbook_main

        return playbook_main(args.args)
    return _run_single(get_recipe(args.recipe), ["--config", str(args.config)])


__all__ = ["RecipeSpec", "dispatch", "get_recipe", "list_recipes", "recipe"]
