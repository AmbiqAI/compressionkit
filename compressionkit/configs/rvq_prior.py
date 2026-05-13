"""Config schema for training an RVQ entropy prior (#27).

The prior is the second stage of the two-stage codec family: a small
causal token transformer trained on RVQ indices extracted from a
frozen parent codec.

A prior config is *always* paired to a parent codec, identified by its
golden ``experiment_id`` (e.g. ``"ecg-rvq-4x"``). The trainer resolves
the parent's run directory and reuses the parent's data settings for
token extraction.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PriorArchConfig(BaseModel):
    """Causal-transformer prior architecture."""

    model_config = ConfigDict(extra="forbid")

    embed_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 128


class PriorTrainingConfig(BaseModel):
    """Token extraction + optimization knobs."""

    model_config = ConfigDict(extra="forbid")

    num_train_files: int = Field(default=200, gt=0)
    context_frames: int = Field(default=4, gt=0)
    stride_tokens: int = Field(default=8, gt=0)
    epochs: int = Field(default=10, gt=0)
    batch_size: int = Field(default=128, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    weight_decay: float = 1e-4


class RvqPriorConfig(BaseModel):
    """Top-level prior training config."""

    model_config = ConfigDict(extra="forbid")

    parent_experiment: str = Field(
        ...,
        description="experiment_id of the parent codec (must be a registered GOLDEN entry).",
    )
    parent_run_dir: Path | None = Field(
        default=None,
        description="Override of the parent's run directory. Normally resolved by the runner.",
    )
    arch: PriorArchConfig = Field(default_factory=PriorArchConfig)
    training: PriorTrainingConfig = Field(default_factory=PriorTrainingConfig)
    export_tflite: bool = Field(
        default=True,
        description="Emit prior_int8.tflite alongside the parent's deploy/ directory.",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RvqPriorConfig:
        """Load and validate from a YAML file."""
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))


__all__ = [
    "PriorArchConfig",
    "PriorTrainingConfig",
    "RvqPriorConfig",
]
