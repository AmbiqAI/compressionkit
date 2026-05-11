"""Pydantic configuration for the canonical-h5 PPG RVQ training pipeline.

This config is the lean cousin of :class:`compressionkit.configs.ppg_rvq.PpgRvqConfig`.
It is dedicated to the new h5-backed loader (multi-dataset, patient-disjoint
splits, window-level sanitization) and skips the EDF/TFRecord machinery that
the legacy MESA-based config carries.

Reuses ``ModelConfig``, ``TrainingConfig``, ``EvaluationConfig``, and
``OutputConfig`` from :mod:`compressionkit.configs.ppg_rvq` so improvements to
losses, schedulers, and callbacks land in both pipelines.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from compressionkit.configs.ppg_rvq import (
    EvaluationConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)


class SourceConfig(BaseModel):
    """One canonical h5 dataset to mix into training.

    Attributes:
        slug: Subdirectory name under ``root`` (e.g. ``bidmc``, ``ppg_dalia``).
        glob: File glob within ``<root>/<slug>/``.
        butppg_quality_only: Drop sessions with ``quality=0`` (butppg only).
    """

    slug: str
    glob: str = "*.h5"
    butppg_quality_only: bool = True


class SanitizeThresholds(BaseModel):
    """Pass-through to :class:`compressionkit.preprocessing.SanitizeConfig`."""

    enabled: bool = True
    min_std: float = 1e-4
    max_saturation_frac: float = 0.10
    max_abs_z: float = 8.0
    max_outlier_frac: float = 0.02


class SplitFractions(BaseModel):
    """Patient-disjoint split fractions; remainder is test."""

    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 1337


class H5DataConfig(BaseModel):
    """Data-loader configuration for the canonical-h5 PPG pipeline."""

    root: str = "/home/vscode/datasets"
    sources: list[SourceConfig] = Field(
        default_factory=lambda: [
            SourceConfig(slug="bidmc"),
            SourceConfig(slug="ppg_dalia"),
            SourceConfig(slug="wesad"),
        ]
    )
    target_fs: int = 64
    window_seconds: float = 4.0
    hop_seconds: float | None = None  # None → non-overlapping
    batch_size: int = 64
    shuffle_buffer: int = 4096
    steps_per_epoch: int = 250
    epochs: int = 50
    sanitize: SanitizeThresholds = Field(default_factory=SanitizeThresholds)
    split: SplitFractions = Field(default_factory=SplitFractions)
    normalize: bool = True

    @property
    def window_samples(self) -> int:
        return int(round(self.window_seconds * self.target_fs))


class PpgH5RvqConfig(BaseModel):
    """Top-level configuration for the canonical-h5 PPG RVQ pipeline."""

    run_name: str = "ppg_h5_rvq_run"
    data: H5DataConfig = Field(default_factory=H5DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> PpgH5RvqConfig:
        """Load and validate config from a YAML file."""
        import yaml

        with Path(path).open("r") as f:
            user_cfg = yaml.safe_load(f) or {}
        return cls.model_validate(user_cfg)


__all__ = [
    "H5DataConfig",
    "PpgH5RvqConfig",
    "SanitizeThresholds",
    "SourceConfig",
    "SplitFractions",
]
