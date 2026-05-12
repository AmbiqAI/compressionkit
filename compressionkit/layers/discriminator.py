"""Multi-scale 1D waveform discriminator for adversarial codec training.

Follows the SoundStream / EnCodec / DAC discriminator design adapted for
physiological signals (ECG at 256 Hz, PPG at 64 Hz).  Each sub-discriminator
is a strided-conv stack that classifies overlapping patches as real or fake.
Multiple sub-discriminators operate at different temporal resolutions
(achieved via average-pooling the input by successive factors of 2).

Input convention: ``(B, 1, T, 1)`` — the same 4D layout used everywhere in
this repo (pseudo-2D with singleton height and single channel).

Design choices per AGENTS.md:

* Pure Conv2D(1,k) ops → portable to LiteRT INT8.
* No spectral norm (adds runtime cost on edge, marginal benefit for 1-ch
  signals at these sample rates).  Weight norm is optional.
* Returns per-layer feature maps alongside the final logit map so the
  trainer can compute feature-matching loss.
"""

from __future__ import annotations

import keras
from keras import layers


def _disc_block(
    x,
    filters: int,
    kernel_width: int,
    stride: int,
    *,
    use_weight_norm: bool = False,
    name_prefix: str = "",
):
    """Single conv → (optional weight-norm) → LeakyReLU block."""
    conv = layers.Conv2D(
        filters,
        kernel_size=(1, kernel_width),
        strides=(1, stride),
        padding="same",
        name=f"{name_prefix}_conv" if name_prefix else None,
    )
    if use_weight_norm:
        # Keras 3 doesn't have built-in WeightNormalization; we skip it
        # to keep the module portable and simple.  The layer is already
        # constrained by the adversarial objective.
        pass
    x = conv(x)
    x = layers.LeakyReLU(0.1, name=f"{name_prefix}_lrelu" if name_prefix else None)(x)
    return x


def build_sub_discriminator(
    frame_size: int,
    *,
    channels: tuple[int, ...] = (16, 32, 64, 128),
    kernel_width: int = 15,
    final_kernel: int = 5,
    use_weight_norm: bool = False,
    name: str = "sub_disc",
) -> keras.Model:
    """Build one sub-discriminator (patch-level real/fake classifier).

    Architecture:
        Conv(channels[0], k, s=1) → LReLU
        Conv(channels[1], k, s=4) → LReLU
        Conv(channels[2], k, s=4) → LReLU
        Conv(channels[3], k, s=4) → LReLU
        Conv(1, final_k, s=1)  # per-patch logit

    Args:
        frame_size: Temporal length of the input signal (e.g. 512 for ECG,
            320 for PPG).
        channels: Number of filters at each strided-conv stage.
        kernel_width: Kernel width for the strided conv stages.
        final_kernel: Kernel width for the final 1×1-ish projection.
        use_weight_norm: Whether to apply weight normalisation (currently
            a no-op; placeholder for future work).
        name: Model name.

    Returns:
        A ``keras.Model`` mapping ``(B, 1, T, 1)`` to a list of tensors:
        ``[feat_0, feat_1, ..., feat_N, logit_map]``.  The logit map has
        shape ``(B, 1, T', 1)`` where ``T'`` depends on the total stride.
        Feature maps are returned for feature-matching loss.
    """
    inp = keras.Input(shape=(1, frame_size, 1), name=f"{name}_input")
    features = []
    x = inp

    # First conv with stride 1 (no downsampling)
    x = _disc_block(
        x, channels[0], kernel_width, stride=1,
        use_weight_norm=use_weight_norm, name_prefix=f"{name}_s0",
    )
    features.append(x)

    # Strided conv blocks (downsample by 4 each)
    for i, ch in enumerate(channels[1:], start=1):
        x = _disc_block(
            x, ch, kernel_width, stride=4,
            use_weight_norm=use_weight_norm, name_prefix=f"{name}_s{i}",
        )
        features.append(x)

    # Final projection to per-patch logits
    logits = layers.Conv2D(
        1,
        kernel_size=(1, final_kernel),
        strides=(1, 1),
        padding="same",
        name=f"{name}_logits",
    )(x)

    return keras.Model(inp, [*features, logits], name=name)


@keras.saving.register_keras_serializable(package="compressionkit")
class MultiScaleDiscriminator(keras.Model):
    """Multi-scale waveform discriminator.

    Applies ``num_scales`` sub-discriminators at progressively down-sampled
    versions of the input (factor-2 average pooling between scales).  This
    forces different sub-discriminators to focus on different frequency bands.

    Args:
        frame_size: Temporal length of the raw signal.
        num_scales: Number of resolution scales (default 3 — matches
            SoundStream for moderate-length signals).
        channels: Channel widths for each sub-discriminator.
        kernel_width: Kernel width for strided conv stages.
        final_kernel: Kernel width for the logit projection.
    """

    def __init__(
        self,
        frame_size: int,
        *,
        num_scales: int = 3,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        kernel_width: int = 15,
        final_kernel: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._frame_size = frame_size
        self._num_scales = num_scales
        self._channels = channels
        self._kernel_width = kernel_width
        self._final_kernel = final_kernel

        self.sub_discriminators: list[keras.Model] = []
        self.downsamplers: list[layers.AveragePooling2D] = []

        fs = frame_size
        for s in range(num_scales):
            self.sub_discriminators.append(
                build_sub_discriminator(
                    fs,
                    channels=channels,
                    kernel_width=kernel_width,
                    final_kernel=final_kernel,
                    name=f"disc_scale_{s}",
                )
            )
            if s < num_scales - 1:
                self.downsamplers.append(
                    layers.AveragePooling2D(
                        pool_size=(1, 2), padding="same",
                        name=f"pool_scale_{s}",
                    )
                )
            fs = (fs + 1) // 2  # ceil division for odd lengths

    def call(self, x, training=False):
        """Run all sub-discriminators and return per-scale outputs.

        Args:
            x: Waveform tensor ``(B, 1, T, 1)``.

        Returns:
            List of ``num_scales`` lists.  Each inner list is
            ``[feat_0, ..., feat_N, logit_map]`` from one sub-discriminator.
        """
        results = []
        for s, disc in enumerate(self.sub_discriminators):
            results.append(disc(x, training=training))
            if s < len(self.downsamplers):
                x = self.downsamplers[s](x)
        return results

    def get_config(self):
        config = super().get_config()
        config.update({
            "frame_size": self._frame_size,
            "num_scales": self._num_scales,
            "channels": self._channels,
            "kernel_width": self._kernel_width,
            "final_kernel": self._final_kernel,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["channels"] = tuple(config["channels"])
        return cls(**config)
