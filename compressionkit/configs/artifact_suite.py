"""Config schema for the role-routing PPG artifact suite.

Lives in ``configs`` (not ``preprocessing``) so it can be referenced by
:class:`compressionkit.configs.ppg_rvq.DataConfig` without importing the heavy
``preprocessing`` package (which would create a circular import). The runtime
augmenter that consumes this config lives in
``compressionkit.preprocessing.artifact_suite``.

See ``experiments/ppg_augmentation_layering_design.md`` for the full design.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ArtifactRole = Literal["recover", "remove", "abstain"]
ArtifactParam = Literal["std", "snr_db", "amplitude", "fraction", "max_warp_fraction", "scale_spread"]


class ArtifactSpec(BaseModel):
    """One artifact in the suite with its role and severity range.

    Attributes:
        name: Artifact identifier (e.g. ``"gaussian"``, ``"baseline_wander"``).
        role: Routing role (``recover`` / ``remove`` / ``abstain``).
        prob: Per-sample probability the artifact fires.
        severity_min: Lower bound of the sampled severity.
        severity_max: Upper bound of the sampled severity.
        param: Parameter the severity controls.
        higher_is_worse: Whether a larger ``param`` value means more corruption.
            ``snr_db`` is the notable exception (lower SNR = worse), so it uses
            ``False``.
    """

    name: str
    role: ArtifactRole
    param: ArtifactParam
    prob: float = Field(default=0.0, ge=0.0, le=1.0)
    severity_min: float
    severity_max: float
    higher_is_worse: bool = True

    @model_validator(mode="after")
    def _check_range(self) -> ArtifactSpec:
        if self.severity_max < self.severity_min:
            self.severity_min, self.severity_max = self.severity_max, self.severity_min
        return self

    @property
    def easy_anchor(self) -> float:
        """Mildest severity value (curriculum ramps from here)."""
        return self.severity_min if self.higher_is_worse else self.severity_max


class NoiseBudgetConfig(BaseModel):
    """Guard against over-corruption that destroys the signal."""

    max_simultaneous: int = Field(default=2, ge=0)
    min_post_corruption_snr_db: float = 3.0
    enforce: Literal["scale_down", "skip", "off"] = "scale_down"


class CurriculumConfig(BaseModel):
    """Ramp a global severity scale over training."""

    enabled: bool = False
    start_scale: float = Field(default=0.2, ge=0.0, le=1.0)
    end_scale: float = Field(default=1.0, ge=0.0, le=1.0)
    ramp_epochs: int = Field(default=50, ge=1)
    schedule: Literal["linear", "cosine"] = "linear"

    def scale_at(self, epoch: int) -> float:
        """Severity scale in ``[start_scale, end_scale]`` at the given epoch."""
        if not self.enabled:
            return 1.0
        frac = min(max(epoch / float(self.ramp_epochs), 0.0), 1.0)
        if self.schedule == "cosine":
            frac = 0.5 * (1.0 - math.cos(math.pi * frac))
        return float(self.start_scale + frac * (self.end_scale - self.start_scale))


def default_artifact_specs() -> list[ArtifactSpec]:
    """Recommended default routing aligned with the Round-3 locked severities."""
    return [
        ArtifactSpec(
            name="gaussian",
            role="remove",
            param="std",
            prob=0.3,
            severity_min=0.10,
            severity_max=0.25,
            higher_is_worse=True,
        ),
        ArtifactSpec(
            name="motion",
            role="remove",
            param="snr_db",
            prob=0.3,
            severity_min=6.0,
            severity_max=15.0,
            higher_is_worse=False,
        ),
        ArtifactSpec(
            name="empirical_noise",
            role="remove",
            param="snr_db",
            prob=0.3,
            severity_min=6.0,
            severity_max=15.0,
            higher_is_worse=False,
        ),
        ArtifactSpec(
            name="baseline_wander",
            role="recover",
            param="amplitude",
            prob=0.3,
            severity_min=0.20,
            severity_max=0.60,
            higher_is_worse=True,
        ),
        # Morphology tracks — stubbed off for the initial pipeline.
        ArtifactSpec(
            name="time_warp",
            role="recover",
            param="max_warp_fraction",
            prob=0.0,
            severity_min=0.05,
            severity_max=0.15,
            higher_is_worse=True,
        ),
        ArtifactSpec(
            name="beat_scale",
            role="recover",
            param="scale_spread",
            prob=0.0,
            severity_min=0.10,
            severity_max=0.30,
            higher_is_worse=True,
        ),
        # Abstention tracks.
        ArtifactSpec(
            name="cutout",
            role="abstain",
            param="fraction",
            prob=0.1,
            severity_min=0.10,
            severity_max=0.40,
            higher_is_worse=True,
        ),
        ArtifactSpec(
            name="null_frame",
            role="abstain",
            param="fraction",
            prob=0.02,
            severity_min=1.0,
            severity_max=1.0,
            higher_is_worse=True,
        ),
    ]


class ArtifactSuiteConfig(BaseModel):
    """Top-level config for the role-routing artifact suite."""

    enabled: bool = False
    faithful_all: bool = Field(
        default=False,
        description="Override every artifact to role='recover' (faithful compression).",
    )
    sample_rate: int = 64
    normalize_after: bool = Field(
        default=True,
        description=(
            "Apply-before-norm policy: corrupt the RAW window, then layer-normalize "
            "each branch by its own per-window stats. Abstain masks are applied after "
            "normalization so dropout regions stay exactly zero."
        ),
    )
    epsilon: float = Field(default=1e-3, description="LayerNorm epsilon for internal normalization.")
    artifacts: list[ArtifactSpec] = Field(default_factory=default_artifact_specs)
    noise_budget: NoiseBudgetConfig = Field(default_factory=NoiseBudgetConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)

    def effective_specs(self) -> list[ArtifactSpec]:
        """Specs with ``faithful_all`` applied.

        When ``faithful_all`` is set, every non-abstain artifact is forced to
        ``recover`` so the model must reconstruct all corruption.
        """
        if not self.faithful_all:
            return self.artifacts
        out: list[ArtifactSpec] = []
        for spec in self.artifacts:
            if spec.role == "abstain":
                out.append(spec)
            else:
                out.append(spec.model_copy(update={"role": "recover"}))
        return out


__all__ = [
    "ArtifactParam",
    "ArtifactRole",
    "ArtifactSpec",
    "ArtifactSuiteConfig",
    "CurriculumConfig",
    "NoiseBudgetConfig",
    "default_artifact_specs",
]
