"""Snake activation layer for periodic signal reconstruction.

Snake(x) = x + (1/α) * sin²(α * x)

The learnable per-channel frequency parameter α provides an inductive bias
for periodic/quasi-periodic signals (ECG, PPG, audio).  This is the same
activation used in BigVGAN, DAC, and EnCodec decoders.

Reference:
    Ziyin, Liu, Tilman Hartwig, and Masahito Ueda. "Neural networks fail to
    learn periodic functions and how to fix it." NeurIPS 2020.
"""

from __future__ import annotations

import keras


@keras.saving.register_keras_serializable(package="compressionkit")
class Snake(keras.layers.Layer):
    """Snake activation with learnable per-channel frequency.

    Args:
        alpha_init: Initial value for the frequency parameter α.
            Higher values → higher-frequency sinusoidal component.
        alpha_logscale: If True, learn log(α) for numerical stability
            and constrain α > 0.
    """

    def __init__(
        self,
        alpha_init: float = 1.0,
        alpha_logscale: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha_init = alpha_init
        self.alpha_logscale = alpha_logscale

    def build(self, input_shape):
        channels = input_shape[-1]
        if self.alpha_logscale:
            import math

            init_val = math.log(self.alpha_init)
            self.log_alpha = self.add_weight(
                name="log_alpha",
                shape=(channels,),
                initializer=keras.initializers.Constant(init_val),
                trainable=True,
            )
        else:
            self.alpha = self.add_weight(
                name="alpha",
                shape=(channels,),
                initializer=keras.initializers.Constant(self.alpha_init),
                trainable=True,
            )
        self.built = True

    def call(self, x):
        if self.alpha_logscale:
            alpha = keras.ops.exp(self.log_alpha)
        else:
            alpha = self.alpha
        # sin²(α*x) = (1 - cos(2αx)) / 2, but direct is fine for clarity
        return x + (1.0 / (alpha + 1e-9)) * keras.ops.power(keras.ops.sin(alpha * x), 2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha_init": self.alpha_init,
                "alpha_logscale": self.alpha_logscale,
            }
        )
        return config
