"""Golden recipe: train an RVQ entropy prior on a frozen parent codec (#27)."""

from __future__ import annotations

from typing import Any

from compressionkit.configs.rvq_prior import RvqPriorConfig
from compressionkit.recipes._registry import recipe
from compressionkit.trainers.rvq_prior import train_prior_from_config


@recipe("train-rvq-prior", config_cls=RvqPriorConfig)
def train(cfg: RvqPriorConfig) -> dict[str, Any]:
    """Train a causal-transformer entropy prior against a registered parent codec."""
    return train_prior_from_config(cfg)


if __name__ == "__main__":
    import sys

    sys.exit(main())  # noqa: F821  # main is injected by @recipe


__all__ = ["train"]
