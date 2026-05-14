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
* ``hf_repo_id`` — ``Ambiq/compressionkit-{modality}-{cr}x``.
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
    hf_repo_id: str = Field(..., description="Ambiq/compressionkit-{modality}-{cr}x")
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
        expected_repo = f"Ambiq/compressionkit-{self.modality}-{self.compression_ratio}x"
        if self.family == "codec" and self.hf_repo_id != expected_repo:
            raise ValueError(
                f"hf_repo_id {self.hf_repo_id!r} does not match {expected_repo!r} for experiment {self.experiment_id!r}"
            )
        if self.family == "two_stage" and self.parent is None:
            raise ValueError(f"two_stage experiment {self.experiment_id!r} must declare a parent codec id")
        if self.family == "codec" and self.parent is not None:
            raise ValueError(f"codec experiment {self.experiment_id!r} must not declare a parent")
        return self


# v1 golden expected metrics — frozen from the first release-grade runs.
_PPG_EXPECTED_METRICS: dict[int, dict[str, float]] = {
    2: {"prd_median": 2.2707, "prd_p90": 10.7564, "spectral_err_median": 0.026309, "cr_codec": 2.0},
    4: {"prd_median": 2.7741, "prd_p90": 14.4301, "spectral_err_median": 0.035686, "cr_codec": 4.0},
    8: {"prd_median": 4.1954, "prd_p90": 22.446, "spectral_err_median": 0.033576, "cr_codec": 8.0},
    16: {"prd_median": 5.4119, "prd_p90": 25.0669, "spectral_err_median": 0.048616, "cr_codec": 16.0},
    32: {"prd_median": 6.9379, "prd_p90": 38.9309, "spectral_err_median": 0.047229, "cr_codec": 32.0},
}
_ECG_EXPECTED_METRICS: dict[int, dict[str, float]] = {
    2: {"prd_median": 2.3178, "prd_p90": 3.4712, "spectral_err_median": 0.035264, "cr_codec": 2.0},
    4: {"prd_median": 3.2977, "prd_p90": 5.271, "spectral_err_median": 0.046192, "cr_codec": 4.0},
    8: {"prd_median": 5.7411, "prd_p90": 10.0261, "spectral_err_median": 0.072537, "cr_codec": 8.0},
    16: {"prd_median": 9.1709, "prd_p90": 17.6363, "spectral_err_median": 0.124397, "cr_codec": 16.0},
    32: {"prd_median": 13.3351, "prd_p90": 22.949, "spectral_err_median": 0.19842, "cr_codec": 32.0},
    64: {"prd_median": 19.9447, "prd_p90": 31.8647, "spectral_err_median": 0.267612, "cr_codec": 64.0},
}


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
        hf_repo_id=f"Ambiq/compressionkit-ppg-{cr}x",
        dataset_id="mesa",
        expected_metrics=_PPG_EXPECTED_METRICS.get(cr),
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
        hf_repo_id=f"Ambiq/compressionkit-ecg-{cr}x",
        dataset_id="ptb-xl",
        expected_metrics=_ECG_EXPECTED_METRICS.get(cr),
    )


def _ppg_two_stage(cr: int) -> GoldenExperiment:
    parent_id = f"ppg-rvq-{cr}x"
    return GoldenExperiment(
        experiment_id=f"ppg-rvq-{cr}x-prior",
        modality="ppg",
        family="two_stage",
        parent=parent_id,
        recipe="train-rvq-prior",
        config_path=Path(f"configs/ppg_rvq_64hz_{cr:02d}x_golden_prior.yaml"),
        run_name=f"ppg_rvq_64hz_{cr:02d}x_golden",
        sample_rate=64,
        compression_ratio=cr,
        hf_repo_id=f"Ambiq/compressionkit-ppg-{cr}x",
        dataset_id="mesa",
    )


def _ecg_two_stage(cr: int) -> GoldenExperiment:
    parent_id = f"ecg-rvq-{cr}x"
    return GoldenExperiment(
        experiment_id=f"ecg-rvq-{cr}x-prior",
        modality="ecg",
        family="two_stage",
        parent=parent_id,
        recipe="train-rvq-prior",
        config_path=Path(f"configs/ecg_rvq_256hz_{cr:02d}x_golden_prior.yaml"),
        run_name=f"ecg_rvq_256hz_{cr:02d}x_golden",
        sample_rate=256,
        compression_ratio=cr,
        hf_repo_id=f"Ambiq/compressionkit-ecg-{cr}x",
        dataset_id="ptb-xl",
    )


# v1 golden codec experiments. Two-stage paired entries land via #27.
GOLDEN_REGISTRY: list[GoldenExperiment] = [
    *(_ppg_codec(cr) for cr in (2, 4, 8, 16, 32)),
    *(_ecg_codec(cr) for cr in (2, 4, 8, 16, 32, 64)),
    # Two-stage (codec + entropy prior) paired entries for selected operating points.
    *(_ppg_two_stage(cr) for cr in (4, 8)),
    *(_ecg_two_stage(cr) for cr in (4, 8)),
]


_BY_ID: dict[str, GoldenExperiment] = {exp.experiment_id: exp for exp in GOLDEN_REGISTRY}
if len(_BY_ID) != len(GOLDEN_REGISTRY):
    raise RuntimeError("duplicate experiment_id in GOLDEN_REGISTRY")

# Every two_stage entry must point at a registered codec parent.
for _exp in GOLDEN_REGISTRY:
    if _exp.family == "two_stage" and _exp.parent not in _BY_ID:
        raise RuntimeError(f"two_stage {_exp.experiment_id!r} parent {_exp.parent!r} not in registry")


def list_two_stage_children(parent_id: str) -> list[GoldenExperiment]:
    """Return every two_stage experiment paired to the given codec parent."""
    return [exp for exp in GOLDEN_REGISTRY if exp.family == "two_stage" and exp.parent == parent_id]


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
        raise KeyError(f"Unknown golden experiment {experiment_id!r}. Known: {sorted(_BY_ID)}") from err


__all__ = [
    "GOLDEN_REGISTRY",
    "GoldenExperiment",
    "GoldenFamily",
    "GoldenModality",
    "get_golden",
    "list_goldens",
    "list_two_stage_children",
]
