"""Golden experiment registry — Pydantic model and v1 entries.

The :data:`GOLDEN_REGISTRY` is the canonical list of v1 published bundles
and local comparison lanes. Each :class:`GoldenExperiment` declares what a
user (or the lifecycle runner in :mod:`compressionkit.experiments.runner`)
needs to reproduce it from a clean checkout:

* ``experiment_id`` — slug used on the CLI (``compressionkit golden run <id>``).
* ``modality`` — ``"ppg"`` or ``"ecg"`` (extensible to future signals).
* ``structure`` — ``"codec"`` for single-stage, ``"two_stage"`` for codec + prior.
  Deliberately distinct from the deploy manifest's ``family`` field (``"rvq"``/
  ``"spiht"``/``"hybrid"``, see :data:`compressionkit.runtime.base.CodecFamily`)
  — this describes the *release shape* (one artifact vs. a paired codec+prior),
  not the codec method.
* ``method`` — ``"rvq"`` (neural), ``"spiht"`` (DSP), or ``"hybrid"`` (DSP+AI).
  Drives the ``run_name`` infix and the HF repo naming convention. Maps 1:1
  onto the deploy manifest's ``family`` field once exported.
* ``parent`` — for ``two_stage`` entries, the ``experiment_id`` of the codec they pair with.
* ``recipe`` — registered training recipe name (see :mod:`compressionkit.recipes`).
  Optional for DSP-only entries that have nothing to train.
* ``config_path`` — YAML config path, relative to the repository root.
* ``run_name`` — ``{modality}_{method}_{sample_rate}hz_{cr:02d}x_golden``
  (matches AGENTS.md naming; ``method="rvq"`` preserves the historical infix).
* ``hf_repo_id`` — reserved publication target. The v1 public bundles are RVQ;
    SPIHT, hybrid, and two-stage entries are local comparison/package lanes until
    they are promoted to a public HuggingFace surface.
* ``dataset_id`` — short, stable identifier consumed by the dataset
  acquisition contract (#26).
* ``expected_metrics`` — optional, frozen scorecard summary populated
    once a golden run is published.

The registry is intentionally the owning abstraction for the v1 golden matrix.
If a modality / method / CR operating point is part of the published surface or
the standardized comparison evidence, it should exist here even if its deploy
artifacts still need to be regenerated locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

GoldenModality = Literal["ppg", "ecg"]
GoldenStructure = Literal["codec", "two_stage"]
GoldenMethod = Literal["rvq", "spiht", "hybrid"]

_PPG_GOLDEN_CRS: tuple[int, ...] = (2, 4, 8, 16, 32)
_ECG_GOLDEN_CRS: tuple[int, ...] = (2, 4, 8, 16, 32, 64)


class HybridSpec(BaseModel):
    """Flexible description of a hybrid (DSP + learned) codec operating point.

    The hybrid lane is intentionally a *family* of approaches, not a single
    pipeline. ``strategy`` is an open dispatch key resolved by the hybrid golden
    runner, so new approaches (unrolled shrinkage, noise predictors,
    denoiser+RVQ, DSP-first, ...) can be added by registering a builder under a
    new key without changing this model. Likewise ``backend`` allows the same
    denoiser to feed either the SPIHT or RVQ entropy/codec stage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str = Field(
        default="wavelet_gain_spiht",
        description="Dispatch key resolved by the hybrid golden runner.",
    )
    denoiser_path: Path = Field(
        ...,
        description="Trained denoiser artifact (e.g. gain_model.keras), repo-relative.",
    )
    backend: Literal["spiht", "rvq"] = Field(
        default="spiht",
        description="Codec/entropy stage applied after denoising.",
    )
    wavelet: str = Field(default="bior4.4", description="DWT wavelet for the SPIHT backend.")
    levels: int = Field(default=6, gt=0, description="DWT decomposition levels.")
    prefilter_low_hz: float | None = Field(
        default=None,
        description="Optional Butterworth high-pass corner (Hz) applied before denoising.",
    )
    prefilter_high_hz: float | None = Field(
        default=None,
        description="Optional Butterworth low-pass corner (Hz) applied before denoising.",
    )
    prefilter_order: int = Field(default=3, gt=0, description="Butterworth order if pre-filter enabled.")


class GoldenExperiment(BaseModel):
    """Declarative description of a v1 golden release experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(
        ...,
        description="Slug used on the CLI, e.g. 'ppg-rvq-4x'.",
        pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
    )
    modality: GoldenModality
    structure: GoldenStructure = Field(
        default="codec",
        description="Release shape: 'codec' (single-stage) or 'two_stage' (codec + prior). "
        "Distinct from the deploy manifest's 'family' (rvq/spiht/hybrid) — see 'method' below.",
    )
    method: GoldenMethod = Field(
        default="rvq",
        description="Codec method: 'rvq' (neural), 'spiht' (DSP), or 'hybrid' (DSP+AI). "
        "Maps 1:1 onto the deploy manifest's 'family' field.",
    )
    parent: str | None = Field(
        default=None,
        description="For two-stage entries, the experiment_id of the paired codec.",
    )
    recipe: str | None = Field(
        default=None,
        description="Registered training recipe name. Optional for DSP-only methods.",
    )
    config_path: Path | None = Field(
        default=None,
        description="YAML config path (repo-relative). Optional for DSP-only methods.",
    )
    run_name: str = Field(..., description="Run directory name under results/.")
    sample_rate: int = Field(..., gt=0, description="Frame sample rate in Hz.")
    compression_ratio: int = Field(..., gt=1, description="Target compression ratio (×).")
    hf_repo_id: str = Field(
        ...,
        description="HF repo: 'Ambiq/compressionkit-{modality}-{cr}x' for RVQ, "
        "'Ambiq/compressionkit-{modality}-{method}-{cr}x' otherwise.",
    )
    dataset_id: str = Field(..., description="Stable dataset identifier (see #26).")
    expected_metrics: dict[str, float] | None = Field(
        default=None,
        description="Frozen scorecard summary; populated after the first release run.",
    )
    hybrid: HybridSpec | None = Field(
        default=None,
        description="Hybrid-lane spec; required when method=='hybrid', forbidden otherwise.",
    )

    @model_validator(mode="after")
    def _check_naming(self) -> GoldenExperiment:
        expected_run = f"{self.modality}_{self.method}_{self.sample_rate}hz_{self.compression_ratio:02d}x_golden"
        if self.structure == "codec" and self.run_name != expected_run:
            raise ValueError(
                f"run_name {self.run_name!r} does not match the AGENTS.md convention "
                f"{expected_run!r} for experiment {self.experiment_id!r}"
            )
        if self.method == "rvq":
            # Back-compat: RVQ goldens omit the method infix in the HF repo id.
            expected_repo = f"Ambiq/compressionkit-{self.modality}-{self.compression_ratio}x"
        else:
            expected_repo = f"Ambiq/compressionkit-{self.modality}-{self.method}-{self.compression_ratio}x"
        if self.structure in ("codec", "two_stage") and self.hf_repo_id != expected_repo:
            raise ValueError(
                f"hf_repo_id {self.hf_repo_id!r} does not match {expected_repo!r} for experiment {self.experiment_id!r}"
            )
        if self.structure == "two_stage" and self.parent is None:
            raise ValueError(f"two_stage experiment {self.experiment_id!r} must declare a parent codec id")
        if self.structure == "codec" and self.parent is not None:
            raise ValueError(f"codec experiment {self.experiment_id!r} must not declare a parent")
        if self.method == "hybrid" and self.hybrid is None:
            raise ValueError(f"hybrid experiment {self.experiment_id!r} must declare a 'hybrid' spec")
        if self.method != "hybrid" and self.hybrid is not None:
            raise ValueError(f"non-hybrid experiment {self.experiment_id!r} must not declare a 'hybrid' spec")
        return self


# v1 golden expected metrics — frozen from the first release-grade runs.
_PPG_EXPECTED_METRICS: dict[int, dict[str, float]] = {
    2: {"prd_median": 0.6875, "prd_p90": 2.8864, "hr_mae_bpm": 0.1094, "cr_codec": 2.0},
    4: {"prd_median": 1.1458, "prd_p90": 3.1181, "hr_mae_bpm": 0.2192, "cr_codec": 4.0},
    8: {"prd_median": 2.7146, "prd_p90": 4.3139, "hr_mae_bpm": 0.5090, "cr_codec": 8.0},
    16: {"prd_median": 5.2531, "prd_p90": 7.7378, "hr_mae_bpm": 0.8808, "cr_codec": 16.0},
    32: {"prd_median": 10.7410, "prd_p90": 16.0615, "hr_mae_bpm": 1.4738, "cr_codec": 32.0},
}
_ECG_EXPECTED_METRICS: dict[int, dict[str, float]] = {
    2: {"prd_median": 2.0039, "prd_p90": 2.9044, "spectral_err_median": 0.006124, "cr_codec": 2.0},
    4: {"prd_median": 3.0530, "prd_p90": 4.2482, "spectral_err_median": 0.011147, "cr_codec": 4.0},
    8: {"prd_median": 4.8852, "prd_p90": 8.1841, "spectral_err_median": 0.020608, "cr_codec": 8.0},
    16: {"prd_median": 7.8739, "prd_p90": 14.9355, "spectral_err_median": 0.030978, "cr_codec": 16.0},
    32: {"prd_median": 12.4878, "prd_p90": 21.8332, "spectral_err_median": 0.066962, "cr_codec": 32.0},
    64: {"prd_median": 18.4424, "prd_p90": 30.6340, "spectral_err_median": 0.110633, "cr_codec": 64.0},
}
_PPG_SPIHT_EXPECTED_METRICS: dict[int, dict[str, float]] = {
    2: {"prd_median": 0.0088, "prd_p90": 0.0330, "hr_mae_bpm": 0.0, "cr_codec": 1.9999},
    4: {"prd_median": 0.1741, "prd_p90": 0.4466, "hr_mae_bpm": 0.0190, "cr_codec": 3.9996},
    8: {"prd_median": 1.5710, "prd_p90": 2.3442, "hr_mae_bpm": 0.2076, "cr_codec": 7.9974},
    16: {"prd_median": 7.1828, "prd_p90": 9.8993, "hr_mae_bpm": 1.2214, "cr_codec": 15.9909},
    32: {"prd_median": 19.2314, "prd_p90": 25.7223, "hr_mae_bpm": 2.1094, "cr_codec": 31.9426},
}
_ECG_SPIHT_EXPECTED_METRICS: dict[int, dict[str, float]] = {
    2: {"prd_median": 0.0537, "prd_p90": 0.1295, "spectral_err_median": 0.000197, "cr_codec": 1.9999},
    4: {"prd_median": 0.8082, "prd_p90": 1.9396, "spectral_err_median": 0.003115, "cr_codec": 3.9997},
    8: {"prd_median": 3.0432, "prd_p90": 7.4458, "spectral_err_median": 0.015764, "cr_codec": 7.9989},
    16: {"prd_median": 7.5208, "prd_p90": 14.7345, "spectral_err_median": 0.066214, "cr_codec": 15.9941},
    32: {"prd_median": 17.1000, "prd_p90": 25.2481, "spectral_err_median": 0.167569, "cr_codec": 31.9703},
    64: {"prd_median": 35.5814, "prd_p90": 47.4770, "spectral_err_median": 0.314775, "cr_codec": 63.8494},
}


def _ppg_codec(cr: int) -> GoldenExperiment:
    return GoldenExperiment(
        experiment_id=f"ppg-rvq-{cr}x",
        modality="ppg",
        structure="codec",
        recipe="train-ppg-rvq",
        config_path=Path(f"configs/ppg_rvq_64hz_{cr:02d}x_golden.yaml"),
        run_name=f"ppg_rvq_64hz_{cr:02d}x_golden",
        sample_rate=64,
        compression_ratio=cr,
        hf_repo_id=f"Ambiq/compressionkit-ppg-{cr}x",
        dataset_id="ppg-unified-strict-sanitize-v1",
        expected_metrics=_PPG_EXPECTED_METRICS.get(cr),
    )


def _ecg_codec(cr: int) -> GoldenExperiment:
    return GoldenExperiment(
        experiment_id=f"ecg-rvq-{cr}x",
        modality="ecg",
        structure="codec",
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
        structure="two_stage",
        parent=parent_id,
        recipe="train-rvq-prior",
        config_path=Path(f"configs/ppg_rvq_64hz_{cr:02d}x_golden_prior.yaml"),
        run_name=f"ppg_rvq_64hz_{cr:02d}x_golden",
        sample_rate=64,
        compression_ratio=cr,
        hf_repo_id=f"Ambiq/compressionkit-ppg-{cr}x",
        dataset_id="ppg-unified-strict-sanitize-v1",
    )


def _ecg_two_stage(cr: int) -> GoldenExperiment:
    parent_id = f"ecg-rvq-{cr}x"
    return GoldenExperiment(
        experiment_id=f"ecg-rvq-{cr}x-prior",
        modality="ecg",
        structure="two_stage",
        parent=parent_id,
        recipe="train-rvq-prior",
        config_path=Path(f"configs/ecg_rvq_256hz_{cr:02d}x_golden_prior.yaml"),
        run_name=f"ecg_rvq_256hz_{cr:02d}x_golden",
        sample_rate=256,
        compression_ratio=cr,
        hf_repo_id=f"Ambiq/compressionkit-ecg-{cr}x",
        dataset_id="ptb-xl",
    )


def _ppg_spiht(cr: int) -> GoldenExperiment:
    """DSP-only SPIHT golden for PPG.

    Uses ``coif5`` with 6 DWT levels per AGENTS.md. Since SPIHT has no
    trained weights, ``recipe`` and ``config_path`` are intentionally
    left unset \u2014 the operating point is fully described by the fields
    on this :class:`GoldenExperiment`, the export packager generates the
    rest.
    """
    return GoldenExperiment(
        experiment_id=f"ppg-spiht-{cr}x",
        modality="ppg",
        structure="codec",
        method="spiht",
        run_name=f"ppg_spiht_64hz_{cr:02d}x_golden",
        sample_rate=64,
        compression_ratio=cr,
        hf_repo_id=f"Ambiq/compressionkit-ppg-spiht-{cr}x",
        dataset_id="ppg-unified-strict-sanitize-v1",
        expected_metrics=_PPG_SPIHT_EXPECTED_METRICS.get(cr),
    )


def _ecg_spiht(cr: int) -> GoldenExperiment:
    """DSP-only SPIHT golden for ECG (``bior4.4``, L=6)."""
    return GoldenExperiment(
        experiment_id=f"ecg-spiht-{cr}x",
        modality="ecg",
        structure="codec",
        method="spiht",
        run_name=f"ecg_spiht_256hz_{cr:02d}x_golden",
        sample_rate=256,
        compression_ratio=cr,
        hf_repo_id=f"Ambiq/compressionkit-ecg-spiht-{cr}x",
        dataset_id="ptb-xl",
        expected_metrics=_ECG_SPIHT_EXPECTED_METRICS.get(cr),
    )


# Locked v1 hybrid denoisers (see internal/v1_three_lane_plan.md). The hybrid
# lane wires these into the SPIHT backend via the wavelet-gain strategy; the
# HybridSpec keeps the door open for other strategies/backends per CR.
_ECG_HYBRID_DENOISER = Path("results/wavelet_denoiser_ecg_v2_full_16k_120ep/gain_model.keras")
_PPG_HYBRID_DENOISER = Path("results/wavelet_denoiser_ppg_v2_amp_noharm/gain_model.keras")


def _ppg_hybrid(cr: int) -> GoldenExperiment:
    """Hybrid (learned wavelet-gain denoiser + SPIHT) golden for PPG."""
    return GoldenExperiment(
        experiment_id=f"ppg-hybrid-{cr}x",
        modality="ppg",
        structure="codec",
        method="hybrid",
        run_name=f"ppg_hybrid_64hz_{cr:02d}x_golden",
        sample_rate=64,
        compression_ratio=cr,
        hf_repo_id=f"Ambiq/compressionkit-ppg-hybrid-{cr}x",
        dataset_id="ppg-unified-strict-sanitize-v1",
        hybrid=HybridSpec(
            strategy="wavelet_gain_spiht",
            denoiser_path=_PPG_HYBRID_DENOISER,
            backend="spiht",
            # Must match the denoiser's training transform (bior4.4, L=6) so the
            # packed-coefficient layout the gain network expects is preserved.
            wavelet="bior4.4",
            levels=6,
        ),
    )


def _ecg_hybrid(cr: int) -> GoldenExperiment:
    """Hybrid (learned wavelet-gain denoiser + SPIHT) golden for ECG."""
    return GoldenExperiment(
        experiment_id=f"ecg-hybrid-{cr}x",
        modality="ecg",
        structure="codec",
        method="hybrid",
        run_name=f"ecg_hybrid_256hz_{cr:02d}x_golden",
        sample_rate=256,
        compression_ratio=cr,
        hf_repo_id=f"Ambiq/compressionkit-ecg-hybrid-{cr}x",
        dataset_id="ptb-xl",
        hybrid=HybridSpec(
            strategy="wavelet_gain_spiht",
            denoiser_path=_ECG_HYBRID_DENOISER,
            backend="spiht",
            wavelet="bior4.4",
            levels=6,
        ),
    )


# v1 golden codec experiments and local comparison lanes.
GOLDEN_REGISTRY: list[GoldenExperiment] = [
    *(_ppg_codec(cr) for cr in _PPG_GOLDEN_CRS),
    *(_ecg_codec(cr) for cr in _ECG_GOLDEN_CRS),
    # DSP-only SPIHT goldens (no trained weights). Register the same
    # operating-point matrix as AI so scorecards and release packaging stay
    # family-symmetric at the golden surface.
    *(_ppg_spiht(cr) for cr in _PPG_GOLDEN_CRS),
    *(_ecg_spiht(cr) for cr in _ECG_GOLDEN_CRS),
    # Hybrid (learned denoiser + SPIHT) goldens — third v1 lane. Same operating
    # points; HybridSpec keeps the per-CR approach swappable.
    *(_ppg_hybrid(cr) for cr in _PPG_GOLDEN_CRS),
    *(_ecg_hybrid(cr) for cr in _ECG_GOLDEN_CRS),
    # Two-stage (codec + entropy prior) paired entries for selected operating points.
    *(_ppg_two_stage(cr) for cr in (4, 8)),
    *(_ecg_two_stage(cr) for cr in (4, 8)),
]


_BY_ID: dict[str, GoldenExperiment] = {exp.experiment_id: exp for exp in GOLDEN_REGISTRY}
if len(_BY_ID) != len(GOLDEN_REGISTRY):
    raise RuntimeError("duplicate experiment_id in GOLDEN_REGISTRY")

# Every two_stage entry must point at a registered codec parent.
for _exp in GOLDEN_REGISTRY:
    if _exp.structure == "two_stage" and _exp.parent not in _BY_ID:
        raise RuntimeError(f"two_stage {_exp.experiment_id!r} parent {_exp.parent!r} not in registry")


def list_two_stage_children(parent_id: str) -> list[GoldenExperiment]:
    """Return every two_stage experiment paired to the given codec parent."""
    return [exp for exp in GOLDEN_REGISTRY if exp.structure == "two_stage" and exp.parent == parent_id]


def list_goldens(
    modality: GoldenModality | None = None,
    method: GoldenMethod | None = None,
) -> list[GoldenExperiment]:
    """Return registered golden experiments, optionally filtered by modality and method."""
    rows = list(GOLDEN_REGISTRY)
    if modality is not None:
        rows = [exp for exp in rows if exp.modality == modality]
    if method is not None:
        rows = [exp for exp in rows if exp.method == method]
    return rows


def get_golden(experiment_id: str) -> GoldenExperiment:
    """Look up a golden experiment by id."""
    try:
        return _BY_ID[experiment_id]
    except KeyError as err:
        raise KeyError(f"Unknown golden experiment {experiment_id!r}. Known: {sorted(_BY_ID)}") from err


__all__ = [
    "GOLDEN_REGISTRY",
    "GoldenExperiment",
    "GoldenMethod",
    "GoldenModality",
    "GoldenStructure",
    "HybridSpec",
    "get_golden",
    "list_goldens",
    "list_two_stage_children",
]
