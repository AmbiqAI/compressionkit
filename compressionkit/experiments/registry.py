"""Golden experiment registry — Pydantic model and v1 entries.

The :data:`GOLDEN_REGISTRY` is the canonical list of v1 release-grade
runs. Each :class:`GoldenExperiment` declares everything a user (or the
lifecycle runner in :mod:`compressionkit.experiments.runner`) needs to
reproduce it from a clean checkout:

* ``experiment_id`` — slug used on the CLI (``compressionkit golden run <id>``).
* ``modality`` — ``"ppg"`` or ``"ecg"`` (extensible to future signals).
* ``family`` — ``"codec"`` for single-stage RVQ, ``"two_stage"`` for codec + prior.
* ``parent`` — for ``two_stage`` entries, the ``experiment_id`` of the codec they pair with.
* ``recipe`` — registered training recipe name (see :mod:`compressionkit.recipes`).
* ``config_path`` — YAML config path, relative to the repository root.
* ``run_name`` — ``{modality}_rvq_{sample_rate}hz_{cr:02d}x_golden`` (matches AGENTS.md naming).
* ``hf_repo_id`` — ``AmbiqAI/compressionkit-{modality}-{cr}x``.
* ``dataset_id`` — short, stable identifier consumed by the dataset
  acquisition contract (#26).
* ``expected_metrics`` — optional, frozen scorecard summary populated
  once a golden run is published.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

GoldenModality = Literal["ppg", "ecg"]
GoldenFamily = Literal["codec", "two_stage"]


class GoldenExperiment(BaseModel):
    """Declarative description of a v1 golden release experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(
        ...,
        description="Slug used on the CLI, e.g. 'ppg-rvq-4x'.",
        pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
    )
    modality: GoldenModality
    family: GoldenFamily = "codec"
    parent: str | None = Field(
        default=None,
        description="For two-stage entries, the experiment_id of the paired codec.",
    )
    recipe: str = Field(..., description="Registered training recipe name.")
    config_path: Path = Field(..., description="YAML config path (repo-relative).")
    run_name: str = Field(..., description="Run directory name under results/.")
    sample_rate: int = Field(..., gt=0, description="Frame sample rate in Hz.")
    compression_ratio: int = Field(..., gt=1, description="Target compression ratio (×).")
    hf_repo_id: str = Field(..., description="AmbiqAI/compressionkit-{modality}-{cr}x")
    dataset_id: str = Field(..., description="Stable dataset identifier (see #26).")
    expected_metrics: dict[str, float] | None = Field(
        default=None,
        description="Frozen scorecard summary; populated after the first release run.",
    )

    @model_validator(mode="after")
    def _check_naming(self) -> GoldenExperiment:
        expected_run = f"{self.modality}_rvq_{self.sample_rate}hz_{self.compression_ratio:02d}x_golden"
        if self.family == "codec" and self.run_name != expected_run:
            raise ValueError(
                f"run_name {self.run_name!r} does not match the AGENTS.md convention "
                f"{expected_run!r} for experiment {self.experiment_id!r}"
            )
        expected_repo = f"AmbiqAI/compressionkit-{self.modality}-{self.compression_ratio}x"
        if self.family == "codec" and self.hf_repo_id != expected_repo:
            raise ValueError(
                f"hf_repo_id {self.hf_repo_id!r} does not match {expected_repo!r} "
                f"for experiment {self.experiment_id!r}"
            )
        if self.family == "two_stage" and self.parent is None:
            raise ValueError(
                f"two_stage experiment {self.experiment_id!r} must declare a parent codec id"
            )
        if self.family == "codec" and self.parent is not None:
            raise ValueError(
                f"codec experiment {self.experiment_id!r} must not declare a parent"
            )
        return self


def _ppg_codec(cr: int) -> GoldenExperiment:
    return GoldenExperiment(
        experiment_id=f"ppg-rvq-{cr}x",
        modality="ppg",
        family="codec",
        recipe="train-ppg-rvq",
        config_path=Path(f"configs/ppg_rvq_64hz_{cr:02d}x_golden.yaml"),
        run_name=f"ppg_rvq_64hz_{cr:02d}x_golden",
        sample_rate=64,
        compression_ratio=cr,
        hf_repo_id=f"AmbiqAI/compressionkit-ppg-{cr}x",
        dataset_id="mesa",
    )


def _ecg_codec(cr: int) -> GoldenExperiment:
    return GoldenExperiment(
        experiment_id=f"ecg-rvq-{cr}x",
        modality="ecg",
        family="codec",
        recipe="train-ecg-rvq",
        config_path=Path(f"configs/ecg_rvq_256hz_{cr:02d}x_golden.yaml"),
        run_name=f"ecg_rvq_256hz_{cr:02d}x_golden",
        sample_rate=256,
        compression_ratio=cr,
        hf_repo_id=f"AmbiqAI/compressionkit-ecg-{cr}x",
        dataset_id="ptb-xl",
    )


# v1 golden codec experiments. Two-stage paired entries land via #27.
GOLDEN_REGISTRY: list[GoldenExperiment] = [
    *(_ppg_codec(cr) for cr in (2, 4, 8, 16, 32)),
    *(_ecg_codec(cr) for cr in (2, 4, 8, 16, 32, 64)),
]


_BY_ID: dict[str, GoldenExperiment] = {exp.experiment_id: exp for exp in GOLDEN_REGISTRY}
if len(_BY_ID) != len(GOLDEN_REGISTRY):
    raise RuntimeError("duplicate experiment_id in GOLDEN_REGISTRY")


def list_goldens(modality: GoldenModality | None = None) -> list[GoldenExperiment]:
    """Return registered golden experiments, optionally filtered by modality."""
    if modality is None:
        return list(GOLDEN_REGISTRY)
    return [exp for exp in GOLDEN_REGISTRY if exp.modality == modality]


def get_golden(experiment_id: str) -> GoldenExperiment:
    """Look up a golden experiment by id."""
    try:
        return _BY_ID[experiment_id]
    except KeyError as err:
        raise KeyError(
            f"Unknown golden experiment {experiment_id!r}. Known: {sorted(_BY_ID)}"
        ) from err


__all__ = [
    "GOLDEN_REGISTRY",
    "GoldenExperiment",
    "GoldenFamily",
    "GoldenModality",
    "get_golden",
    "list_goldens",
]
