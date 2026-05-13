"""End-to-end training recipes for compressionkit.

A *recipe* is a top-to-bottom readable Python module that composes the
reusable building blocks in :mod:`compressionkit.trainers` into a
complete training run. Recipes register themselves with the
:func:`recipe` decorator, which gives each one a CLI-ready ``main``
entry point and lists it in the process-wide registry used by the
``compressionkit`` multiplexing command.

Two locations exist:

* **Golden recipes** live here in :mod:`compressionkit.recipes` and ship
  with the package. They are the reference training flows for each signal
  type (PPG, ECG).
* **Active experiments** live in the top-level ``recipes/`` folder of
  the repository. Those are meant to be copied from a golden recipe and
  freely modified — they are not part of the importable package.

Importing this package auto-imports the shipped recipes so that the
registry is populated (needed for ``compressionkit list`` and the
multiplexing dispatcher).
"""

from __future__ import annotations

# Import shipped recipes so registration side-effects fire. Kept at the
# bottom to avoid any circular imports from the registry module.
from compressionkit.recipes import (
    train_ecg_rvq,
    train_ppg_h5_rvq,
    train_ppg_rvq,
    train_rvq_prior,
)
from compressionkit.recipes._registry import (
    RecipeSpec,
    dispatch,
    get_recipe,
    list_recipes,
    recipe,
)

__all__ = [
    "RecipeSpec",
    "dispatch",
    "get_recipe",
    "list_recipes",
    "recipe",
]
