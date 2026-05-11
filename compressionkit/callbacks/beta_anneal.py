"""Commitment-loss (beta) annealing callback for RVQ training.

Converts the RVQ layer's plain-float ``beta`` attribute into a tracked
``keras.Variable`` on first attach so the value can be updated per epoch
*without* forcing the model to retrace its train function. The callback
then schedules ``beta`` from ``start`` to ``end`` over ``epochs`` epochs,
holding at ``end`` for any remaining epochs.

Typical recipe: start with high beta (e.g., 1.0) to lock the encoder
output near the codebook early, then anneal toward a low beta (e.g., 0.1)
so the encoder is freer to refine its representation. This avoids the
common "codebook collapse then recovery" instability while still
allowing fine-tuning of the encoder/codebook coupling.
"""

from __future__ import annotations

import math

import keras


def _coerce_beta_variable(rvq_layer: keras.layers.Layer) -> keras.Variable:
    """Replace ``rvq_layer.beta`` with a ``keras.Variable`` if it is still a Python float.

    The RVQ layer (helia_edge or local) stores ``self.beta`` as a Python
    float; multiplying by it inside ``call()`` bakes the value into the
    traced graph. Converting it to a ``keras.Variable`` lets us
    ``.assign()`` new values at epoch boundaries with no retrace.

    Keras layers block adding new state after ``build()``; we therefore
    use ``object.__setattr__`` to bypass the tracked-attribute check.
    This is safe because the variable only appears as a scalar multiplier
    inside ``add_loss`` (not as a trainable weight) and does not need to
    participate in layer serialisation.
    """
    current = getattr(rvq_layer, "beta", None)
    if isinstance(current, keras.Variable):
        return current
    initial = float(current) if current is not None else 0.25
    var = keras.Variable(
        initial,
        trainable=False,
        dtype="float32",
        name=f"{rvq_layer.name}_beta",
    )
    object.__setattr__(rvq_layer, "beta", var)
    return var


class BetaAnneal(keras.callbacks.Callback):
    """Anneal RVQ commitment-loss weight ``beta`` over training epochs.

    Args:
        rvq_layer: The RVQ layer whose ``beta`` will be annealed.
        start: Initial beta value at epoch 0.
        end: Final beta value at and after ``epochs``.
        epochs: Number of epochs over which to anneal.
        mode: ``"cosine"`` (smooth) or ``"linear"`` interpolation.
    """

    def __init__(
        self,
        rvq_layer: keras.layers.Layer,
        *,
        start: float = 1.0,
        end: float = 0.1,
        epochs: int = 50,
        mode: str = "cosine",
    ) -> None:
        super().__init__()
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if mode not in {"cosine", "linear"}:
            raise ValueError(f"unsupported mode: {mode}")
        self.rvq_layer = rvq_layer
        self.start = float(start)
        self.end = float(end)
        self.epochs = int(epochs)
        self.mode = mode
        self._beta_var = _coerce_beta_variable(rvq_layer)

    def _value_at(self, epoch: int) -> float:
        t = min(max(epoch, 0), self.epochs) / self.epochs
        if self.mode == "cosine":
            return self.end + 0.5 * (self.start - self.end) * (1.0 + math.cos(math.pi * t))
        return self.start + (self.end - self.start) * t

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        new_beta = self._value_at(epoch)
        self._beta_var.assign(new_beta)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is not None:
            logs["rvq_beta"] = float(self._beta_var.numpy())
