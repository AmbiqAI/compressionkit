"""Example experimental recipe — PPG RVQ with a custom extra loss.

Starting point for users who want to modify the golden PPG recipe. This
file is **not** part of the installed package — it's intended as a
template you copy and tweak. Run it with:

    uv run python recipes/example_ppg_rvq_with_custom_loss.py --config configs/ppg_rvq_08x_ds8_l2.yaml

The demo override swaps in an L1 reconstruction loss on top of the
default MSE + enabled extras. Replace or extend to prototype ideas
without touching the golden recipe or the trainer helpers.
"""

from __future__ import annotations

import sys
from typing import Any

import keras

from compressionkit.configs.ppg_rvq import PpgRvqConfig
from compressionkit.recipes import recipe
from compressionkit.recipes.train_ppg_rvq import train as golden_train
from compressionkit.trainers import ppg_rvq as ppg_trainer


def _l1_recon_loss(weight: float = 0.1) -> callable:
    """Simple L1 reconstruction loss — example of a custom extra loss."""

    def loss(y_true, y_pred):
        return weight * keras.ops.mean(keras.ops.abs(y_true - y_pred))

    loss.__name__ = "l1_recon"
    return loss


_original_build_extra_losses = ppg_trainer.build_extra_losses


def _build_extra_losses_with_l1(cfg: PpgRvqConfig) -> list[callable]:
    extras = _original_build_extra_losses(cfg)
    extras.append(_l1_recon_loss(weight=0.1))
    return extras


@recipe("train-ppg-rvq-l1", config_cls=PpgRvqConfig)
def train(cfg: PpgRvqConfig) -> dict[str, Any]:
    """Run the golden PPG recipe with an extra L1 reconstruction loss."""
    ppg_trainer.build_extra_losses = _build_extra_losses_with_l1
    try:
        return golden_train(cfg)
    finally:
        ppg_trainer.build_extra_losses = _original_build_extra_losses


if __name__ == "__main__":
    sys.exit(main())  # noqa: F821  # main is injected by @recipe
