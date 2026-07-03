"""Wavelet-domain gain denoiser for the hybrid denoise + SPIHT codec.

Recent speech-enhancement designs (GTCRN, FastEnhancer, DeepFilterNet v2)
converge on a small causal network that predicts a *bounded mask* over a
sparsifying transform. This module is the wavelet-domain, edge-deployable
analogue: a tiny causal depthwise-separable conv stack that predicts a
per-coefficient **gain** ``g in [0, 1]`` over packed DWT coefficients.

To stay competitive on *clean* input (the "can't do poor on good SNR"
guardrail) the network is **noise-level conditioned**: alongside the packed
coefficients it receives a per-window noise indicator (the log finest-band
energy). This lets the gain back off toward 1 when the input is clean and
shrink hard only when noise is present — a learned, adaptive analogue of the
BayesShrink ``sigma_n``-driven threshold.

Design choices tied to the deployment constraints (see ``AGENTS.md``):

* **Gain-only output.** The denoised coefficient is ``g * c``; the network can
  only *attenuate* coefficients, never synthesize them. This is a structural
  anti-imprinting guarantee — the model cannot invent a heartbeat from noise,
  which is verified by the RR-autocorr probe in the empirical regime sweep.
* **Static shapes / fixed layout.** ``frame_size`` and DWT ``levels`` are fixed,
  so the packed coefficient length is constant and there is no dynamic
  allocation.
* **LiteRT-friendly ops only.** Causal (left-padded) Conv2D / DepthwiseConv2D,
  ReLU, sigmoid, multiply, concat — all quantizable to INT8 / INT16x8.
* **Near-identity initialization.** The gain head starts close to ``1`` so an
  untrained model degrades gracefully to plain SPIHT.

The network operates on the codebase's ``(B, 1, T, C)`` temporal convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import keras
import numpy as np

from compressionkit.dsp.wavelet_features import (
    band_ratio_feature_np,
    finest_band_indices,
    level_feature_np,
)

__all__ = [
    "BandRatioFeature",
    "DenoiserMode",
    "FinestLevelFeature",
    "LevelGate",
    "as_coeff_denoiser",
    "band_ratio_feature_np",
    "build_wavelet_denoiser_v2",
    "build_wavelet_gain_denoiser",
    "build_wavelet_noise_predictor",
    "build_wavelet_unrolled_shrinkage",
    "detect_denoiser_mode",
    "finest_band_indices",
    "level_feature_np",
]


@keras.saving.register_keras_serializable(package="compressionkit")
class FinestLevelFeature(keras.layers.Layer):
    """Append a per-window noise-level channel to packed coefficients.

    Input ``(B, 1, T, 1)`` packed coefficients → output ``(B, 1, T, 2)`` where
    channel 0 is the coefficients and channel 1 is ``log1p(std(finest band))``
    broadcast across time. The finest detail band's energy is a robust proxy
    for the input noise level.
    """

    def __init__(self, start: int, length: int, **kwargs):
        super().__init__(**kwargs)
        self.start = int(start)
        self.length = int(length)

    def call(self, x):
        finest = x[:, :, self.start : self.start + self.length, :]
        level = keras.ops.std(finest, axis=2, keepdims=True)  # (B, 1, 1, 1)
        level = keras.ops.log1p(level)
        level = keras.ops.broadcast_to(level, keras.ops.shape(x))
        return keras.ops.concatenate([x, level], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start, "length": self.length})
        return config


@keras.saving.register_keras_serializable(package="compressionkit")
class BandRatioFeature(keras.layers.Layer):
    """Append a per-window log band-ratio feature to packed coefficients.

    Uses the ratio of the finest detail-band RMS to the coarser-band RMS as a
    more morphology-robust noise proxy than raw finest-band level alone. Sharp
    QRS morphology raises both fine and coarse energy, while broadband noise
    disproportionately raises the finest band, so the ratio is better suited to
    clean-vs-noisy gating on real ECG.
    """

    def __init__(self, start: int, length: int, **kwargs):
        super().__init__(**kwargs)
        self.start = int(start)
        self.length = int(length)

    def call(self, x):
        coeff = x[:, :, :, :1]
        finest = coeff[:, :, self.start : self.start + self.length, :]
        coarse = keras.ops.concatenate(
            [coeff[:, :, : self.start, :], coeff[:, :, self.start + self.length :, :]],
            axis=2,
        )
        fine_rms = keras.ops.sqrt(keras.ops.mean(keras.ops.square(finest), axis=2, keepdims=True) + 1e-6)
        coarse_rms = keras.ops.sqrt(keras.ops.mean(keras.ops.square(coarse), axis=2, keepdims=True) + 1e-6)
        ratio = keras.ops.log1p(fine_rms / (coarse_rms + 1e-6))
        ratio = keras.ops.broadcast_to(ratio, keras.ops.shape(coeff))
        return keras.ops.concatenate([coeff, ratio], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start, "length": self.length})
        return config


@keras.saving.register_keras_serializable(package="compressionkit")
class SelectChannel(keras.layers.Layer):
    """Select a single channel from ``(B, 1, T, C)`` -> ``(B, 1, T, 1)``."""

    def __init__(self, index: int, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, x):
        return x[:, :, :, self.index : self.index + 1]

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


@keras.saving.register_keras_serializable(package="compressionkit")
class NoiseGatedGain(keras.layers.Layer):
    """Combine a learned suppression with a noise-level gate into a gain.

    Given suppression ``s in [0, 1]`` and per-window level ``L``, the gain is::

        gate = sigmoid(a * (L - b))      # learnable a > 0, b
        g    = 1 - s * gate

    As the estimated noise level ``L`` drops below ``b`` the gate closes
    (``gate -> 0``) and the gain returns to ``1`` (exact identity), so the
    network cannot over-denoise clean input regardless of loss balance.
    """

    def __init__(self, a_init: float = 1.0, b_init: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.a_init = float(a_init)
        self.b_init = float(b_init)

    def build(self, input_shape):
        self.a = self.add_weight(name="gate_a", shape=(), initializer=keras.initializers.Constant(self.a_init))
        self.b = self.add_weight(name="gate_b", shape=(), initializer=keras.initializers.Constant(self.b_init))
        super().build(input_shape)

    def call(self, inputs):
        suppression, level = inputs
        gate = keras.ops.sigmoid(keras.ops.abs(self.a) * (level - self.b))
        return 1.0 - suppression * gate

    def get_config(self):
        config = super().get_config()
        config.update({"a_init": self.a_init, "b_init": self.b_init})
        return config


@keras.saving.register_keras_serializable(package="compressionkit")
class SoftThreshold(keras.layers.Layer):
    """Apply elementwise soft-thresholding to ``(value, threshold)`` inputs."""

    def call(self, inputs):
        value, threshold = inputs
        magnitude = keras.ops.maximum(keras.ops.abs(value) - threshold, 0.0)
        return keras.ops.sign(value) * magnitude


def _causal_ds_block(
    x: keras.KerasTensor,
    filters: int,
    k_w: int,
    dilation: int,
    name: str,
) -> keras.KerasTensor:
    """Causal depthwise-separable conv block on ``(B, 1, T, C)``.

    Causality along the time axis (axis=2) is enforced by left-padding with
    ``dilation * (k_w - 1)`` zeros and using ``padding="valid"``.
    """
    pad = dilation * (k_w - 1)
    x = keras.layers.ZeroPadding2D(padding=((0, 0), (pad, 0)), name=f"{name}_pad")(x)
    x = keras.layers.DepthwiseConv2D(
        (1, k_w),
        dilation_rate=(1, dilation),
        padding="valid",
        name=f"{name}_dw",
    )(x)
    x = keras.layers.Activation("relu", name=f"{name}_dw_act")(x)
    x = keras.layers.Conv2D(filters, (1, 1), padding="same", name=f"{name}_pw")(x)
    x = keras.layers.Activation("relu", name=f"{name}_pw_act")(x)
    return x


def build_wavelet_gain_denoiser(
    frame_size: int = 512,
    in_ch: int = 2,
    width: int = 24,
    k_w: int = 5,
    dilations: tuple[int, ...] = (1, 2, 4),
    identity_bias: float = 4.0,
    gate_b_init: float = 0.3,
    name: str = "wavelet_gain_denoiser",
) -> keras.Model:
    """Build a tiny causal gain denoiser over packed DWT coefficients.

    Args:
        frame_size: Packed coefficient length (equals the signal frame size,
            since the DWT is length-preserving).
        in_ch: Input channels. ``1`` = raw packed coefficients. Additional
            channels (e.g. per-subband sigma, scale id) can be added later.
        width: Channel width of the conv stack.
        k_w: Temporal kernel size.
        dilations: Dilation factors for successive causal blocks.
        identity_bias: Bias initializer for the gain head so the initial gain
            is ``sigmoid(identity_bias) ~ 1`` (near-identity passthrough).
        name: Model name.

    Returns:
        A Keras model mapping ``(B, 1, frame_size, in_ch)`` to a gain tensor
        ``(B, 1, frame_size, 1)`` with values in ``[0, 1]``.
    """
    inp = keras.layers.Input(shape=(1, frame_size, in_ch), name="coeffs_in")
    x = inp
    for i, d in enumerate(dilations):
        x = _causal_ds_block(x, width, k_w=k_w, dilation=d, name=f"blk{i}")
    if in_ch == 2:
        # Noise-gated gain: suppression starts near 0 (bias -identity_bias) and
        # is gated by the per-window noise level so clean input stays identity.
        suppression = keras.layers.Conv2D(
            1,
            (1, 1),
            padding="same",
            activation="sigmoid",
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(-identity_bias),
            name="suppression",
        )(x)
        level = SelectChannel(1, name="level")(inp)
        gain = NoiseGatedGain(b_init=gate_b_init, name="gain")([suppression, level])
    else:
        gain = keras.layers.Conv2D(
            1,
            (1, 1),
            padding="same",
            activation="sigmoid",
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(identity_bias),
            name="gain",
        )(x)
    return keras.Model(inp, gain, name=name)


@keras.saving.register_keras_serializable(package="compressionkit")
class LevelGate(keras.layers.Layer):
    """Noise-level gate ``sigmoid(|a| * (level - b))`` with learnable ``a, b``.

    Closes (``-> 0``) when the per-window noise level falls below ``b`` so a
    gated additive correction vanishes on clean input.
    """

    def __init__(self, a_init: float = 1.0, b_init: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.a_init = float(a_init)
        self.b_init = float(b_init)

    def build(self, input_shape):
        self.a = self.add_weight(name="gate_a", shape=(), initializer=keras.initializers.Constant(self.a_init))
        self.b = self.add_weight(name="gate_b", shape=(), initializer=keras.initializers.Constant(self.b_init))
        super().build(input_shape)

    def call(self, level):
        return keras.ops.sigmoid(keras.ops.abs(self.a) * (level - self.b))

    def get_config(self):
        config = super().get_config()
        config.update({"a_init": self.a_init, "b_init": self.b_init})
        return config


def _residual_ds_block(
    x: keras.KerasTensor,
    filters: int,
    k_w: int,
    dilation: int,
    name: str,
) -> keras.KerasTensor:
    """Causal depthwise-separable block with a residual add when shapes match."""
    y = _causal_ds_block(x, filters, k_w=k_w, dilation=dilation, name=name)
    if x.shape[-1] == filters:
        y = keras.layers.Add(name=f"{name}_res")([x, y])
    return y


def build_wavelet_denoiser_v2(
    frame_size: int = 512,
    in_ch: int = 2,
    width: int = 48,
    k_w: int = 5,
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
    identity_bias: float = 4.0,
    gate_b_init: float = 0.3,
    residual_scale: float = 0.5,
    name: str = "wavelet_denoiser_v2",
) -> keras.Model:
    """Higher-capacity wavelet denoiser with a gated-residual mode of operation.

    Unlike :func:`build_wavelet_gain_denoiser` (gain-only attenuation), this
    model can both attenuate *and* correct coefficients::

        denoised = gain * c  +  gate(level) * (residual_scale * tanh(r))

    where ``gain in [0, 1]`` (near-1 at init), ``r`` is a learned additive
    correction (near-0 at init), and ``gate(level)`` closes on clean input. So
    at initialization and on clean windows the model is identity; under noise it
    can suppress *and* restore morphology, not merely shrink. The additive term
    is bounded and noise-gated to keep hallucination in check (measured by the
    imprint probe).

    Deeper/wider than the gain net (residual DS-conv stack, longer dilation
    schedule) for materially more capacity. All ops are LiteRT/INT8-friendly.

    Returns:
        A model mapping ``(B, 1, frame_size, in_ch)`` to denoised coefficients
        ``(B, 1, frame_size, 1)`` (output layer named ``"denoised"``).
    """
    inp = keras.layers.Input(shape=(1, frame_size, in_ch), name="coeffs_in")
    coeff = SelectChannel(0, name="coeff")(inp)

    x = keras.layers.Conv2D(width, (1, 1), padding="same", name="stem")(inp)
    for i, d in enumerate(dilations):
        x = _residual_ds_block(x, width, k_w=k_w, dilation=d, name=f"blk{i}")

    gain = keras.layers.Conv2D(
        1,
        (1, 1),
        padding="same",
        activation="sigmoid",
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(identity_bias),
        name="gain",
    )(x)
    attenuated = keras.layers.Multiply(name="attenuated")([gain, coeff])

    residual = keras.layers.Conv2D(
        1,
        (1, 1),
        padding="same",
        activation="tanh",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="residual",
    )(x)
    residual = keras.layers.Rescaling(residual_scale, name="residual_scaled")(residual)

    if in_ch == 2:
        level = SelectChannel(1, name="level")(inp)
        gate = LevelGate(b_init=gate_b_init, name="res_gate")(level)
        residual = keras.layers.Multiply(name="gated_residual")([gate, residual])

    denoised = keras.layers.Add(name="denoised")([attenuated, residual])
    return keras.Model(inp, denoised, name=name)


def build_wavelet_noise_predictor(
    frame_size: int = 512,
    in_ch: int = 2,
    width: int = 48,
    k_w: int = 5,
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
    gate_b_init: float = 0.3,
    residual_scale: float = 0.5,
    name: str = "wavelet_noise_predictor",
) -> keras.Model:
    """Predict artifact residual coefficients and subtract them from the input.

    The model's natural identity solution is zero residual:

        denoised = coeff - gate(level) * (residual_scale * tanh(noise_pred))

    so on clean input it is biased toward exact passthrough rather than direct
    signal rewriting. Under noise it learns what to remove instead of what to
    regenerate. The residual is bounded and noise-gated to reduce imprint risk.
    """
    inp = keras.layers.Input(shape=(1, frame_size, in_ch), name="coeffs_in")
    coeff = SelectChannel(0, name="coeff")(inp)

    x = keras.layers.Conv2D(width, (1, 1), padding="same", name="stem")(inp)
    for i, d in enumerate(dilations):
        x = _residual_ds_block(x, width, k_w=k_w, dilation=d, name=f"blk{i}")

    noise_pred = keras.layers.Conv2D(
        1,
        (1, 1),
        padding="same",
        activation="tanh",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="noise_pred",
    )(x)
    noise_pred = keras.layers.Rescaling(residual_scale, name="noise_scaled")(noise_pred)

    if in_ch == 2:
        level = SelectChannel(1, name="level")(inp)
        gate = LevelGate(b_init=gate_b_init, name="noise_gate")(level)
        noise_pred = keras.layers.Multiply(name="gated_noise")([gate, noise_pred])

    denoised = keras.layers.Subtract(name="denoised")([coeff, noise_pred])
    return keras.Model(inp, denoised, name=name)


def build_wavelet_unrolled_shrinkage(
    frame_size: int = 512,
    in_ch: int = 2,
    width: int = 48,
    k_w: int = 5,
    dilations: tuple[int, ...] = (1, 2, 4),
    gate_b_init: float = 0.3,
    residual_scale: float = 0.35,
    max_threshold: float = 0.12,
    stages: int = 4,
    name: str = "wavelet_unrolled_shrinkage",
) -> keras.Model:
    """Fixed-depth unrolled shrinkage denoiser over packed DWT coefficients.

    Each stage performs a bounded learned correction followed by a soft-threshold
    proximal step, which is more conservative than a free-form residual CNN:

        proposal = coeff - gate(level) * delta(x)
        coeff    = soft_threshold(proposal, gate(level) * tau(x))
    """
    inp = keras.layers.Input(shape=(1, frame_size, in_ch), name="coeffs_in")
    coeff = SelectChannel(0, name="coeff_init")(inp)
    level = SelectChannel(1, name="level")(inp) if in_ch == 2 else None
    state = inp

    for stage_idx in range(stages):
        x = keras.layers.Conv2D(width, (1, 1), padding="same", name=f"stage{stage_idx}_stem")(state)
        for block_idx, dilation in enumerate(dilations):
            x = _residual_ds_block(x, width, k_w=k_w, dilation=dilation, name=f"stage{stage_idx}_blk{block_idx}")

        delta = keras.layers.Conv2D(
            1,
            (1, 1),
            padding="same",
            activation="tanh",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name=f"stage{stage_idx}_delta",
        )(x)
        delta = keras.layers.Rescaling(residual_scale, name=f"stage{stage_idx}_delta_scaled")(delta)

        tau = keras.layers.Conv2D(
            1,
            (1, 1),
            padding="same",
            activation="sigmoid",
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(-4.0),
            name=f"stage{stage_idx}_tau",
        )(x)
        tau = keras.layers.Rescaling(max_threshold, name=f"stage{stage_idx}_tau_scaled")(tau)

        if level is not None:
            gate = LevelGate(b_init=gate_b_init, name=f"stage{stage_idx}_gate")(level)
            delta = keras.layers.Multiply(name=f"stage{stage_idx}_gated_delta")([gate, delta])
            tau = keras.layers.Multiply(name=f"stage{stage_idx}_gated_tau")([gate, tau])

        proposal = keras.layers.Subtract(name=f"stage{stage_idx}_proposal")([coeff, delta])

        coeff = SoftThreshold(name=f"stage{stage_idx}_shrink")([proposal, tau])
        state = (
            keras.layers.Concatenate(axis=-1, name=f"stage{stage_idx}_state")([coeff, level])
            if level is not None
            else coeff
        )

    denoised = keras.layers.Identity(name="denoised")(coeff)
    return keras.Model(inp, denoised, name=name)


@dataclass(frozen=True)
class DenoiserMode:
    """Auto-detected calling convention for a trained wavelet-gain denoiser.

    Captured once (from a live Keras model) so it can be persisted into a
    deploy manifest and reused by keras-free consumers (e.g. a LiteRT
    runtime) that can no longer introspect a converted ``.tflite`` model's
    Python layer types or output names.
    """

    start: int
    length: int
    expects_level: bool
    feature_kind: Literal["level", "band_ratio", "none"]
    gain_mode: bool

    def to_dict(self) -> dict[str, int | bool | str]:
        """Serialize for storage in a JSON manifest."""
        return {
            "start": self.start,
            "length": self.length,
            "expects_level": self.expects_level,
            "feature_kind": self.feature_kind,
            "gain_mode": self.gain_mode,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DenoiserMode:
        """Deserialize from a JSON manifest entry produced by :meth:`to_dict`."""
        return cls(
            start=int(data["start"]),
            length=int(data["length"]),
            expects_level=bool(data["expects_level"]),
            feature_kind=data["feature_kind"],
            gain_mode=bool(data["gain_mode"]),
        )

    def compute_feature(self, packed: np.ndarray) -> np.ndarray:
        """Build the ``(1, 1, T, C)`` model input from packed coefficients."""
        arr = np.asarray(packed, dtype=np.float32)
        if self.feature_kind == "band_ratio":
            return band_ratio_feature_np(arr, self.start, self.length)[None, None, :, :]
        if self.feature_kind == "level":
            return level_feature_np(arr, self.start, self.length)[None, None, :, :]
        return arr.reshape(1, 1, -1, 1)


def detect_denoiser_mode(model: keras.Model, *, frame_size: int, wavelet: str, levels: int) -> DenoiserMode:
    """Auto-detect a trained denoiser's input/output calling convention.

    Inspects the model's input width (plain coefficients vs. coefficients +
    a noise-level feature channel), which feature layer it uses (finest-level
    vs. band-ratio), and its output convention (gain-multiply vs. direct
    replacement) so callers don't need to hardcode any of this per model.
    """
    start, length = finest_band_indices(frame_size, wavelet, levels)
    expects_level = int(model.input_shape[-1]) == 2
    try:
        out_name = model.output_names[0]
    except (AttributeError, IndexError):
        out_name = model.layers[-1].name
    gain_mode = "gain" in out_name.lower()
    feature_kind: Literal["level", "band_ratio", "none"] = "none"
    if expects_level:
        layer_types = {type(layer).__name__ for layer in model.layers}
        feature_kind = "band_ratio" if "BandRatioFeature" in layer_types else "level"
    return DenoiserMode(
        start=start,
        length=length,
        expects_level=expects_level,
        feature_kind=feature_kind,
        gain_mode=gain_mode,
    )


def as_coeff_denoiser(model: keras.Model, *, frame_size: int, wavelet: str, levels: int):
    """Adapt a denoiser model into a ``packed_coeffs -> denoised_coeffs`` callable.

    Auto-detects the model's output convention: a *gain* model (output named
    ``"gain"``) multiplies the coefficients by the predicted gain; a *direct*
    model (e.g. :func:`build_wavelet_denoiser_v2`, output named ``"denoised"``)
    returns the denoised coefficients directly. Builds the noise-level feature
    channel when the model expects it. Suitable for
    :class:`LearnedShrinkSpihtCodec`.

    The approximation (DC/baseline) band is always passed through unchanged,
    regardless of what the model predicts for it. It carries the signal's
    low-frequency/DC level, not sensor noise, so it should never be part of a
    denoising correction; a model that assigns it a large gain/replacement
    correction (whether from training imbalance or a noisy label) would
    otherwise introduce a systematic amplitude/baseline shift in the
    reconstructed signal that has nothing to do with actual denoising.
    """
    mode = detect_denoiser_mode(model, frame_size=frame_size, wavelet=wavelet, levels=levels)

    def _denoise(packed: np.ndarray) -> np.ndarray:
        arr = np.asarray(packed, dtype=np.float32)
        feat = mode.compute_feature(arr)
        out = model(feat, training=False)
        out = np.asarray(out).reshape(-1)[: arr.shape[0]]
        denoised = arr * out if mode.gain_mode else out
        denoised = denoised.copy()
        denoised[: mode.start] = arr[: mode.start]  # approx/DC band: always pass through unchanged
        return denoised

    return _denoise
