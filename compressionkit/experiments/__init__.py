"""Golden experiment registry and lifecycle runner.

A *golden experiment* is the canonical recipe for a single compressionkit
release artifact: a config, the trained run name, a HuggingFace repo id,
the dataset it requires, and (once first published) its frozen evaluation
metrics.

The registry is the single source of truth used by the lifecycle runner
to drive ingest → train → evaluate → export → scorecard → model card →
optional HuggingFace publish from one command:

    compressionkit golden list
    compressionkit golden run ppg-rvq-4x
    compressionkit golden run-all --modality ppg

See issue #25 for scope and issue #29 for the v1 hardening tracker.
"""

from __future__ import annotations

from compressionkit.experiments.registry import (
    GOLDEN_REGISTRY,
    GoldenExperiment,
    GoldenFamily,
    GoldenMethod,
    GoldenModality,
    get_golden,
    list_goldens,
)
from compressionkit.experiments.runner import run_golden

__all__ = [
    "GOLDEN_REGISTRY",
    "GoldenExperiment",
    "GoldenFamily",
    "GoldenMethod",
    "GoldenModality",
    "get_golden",
    "list_goldens",
    "run_golden",
]
