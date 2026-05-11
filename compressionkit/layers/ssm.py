"""Diagonal structured state-space (SSM) layers for edge AI.

This module implements a clean Keras 3 layer that runs the same computation on
TensorFlow, JAX, and PyTorch backends, while mapping 1-to-1 onto a tiny fixed
memory recurrence suitable for embedded C / LiteRT INT8 deployment.

The layer is a *diagonal* SSM in the LRU / S4D-Lin family:

    For each of N independent state channels, run a damped-rotation recurrence
    in real coordinates:

        [s_r, s_i]_t = decay_n * R(theta_n) @ [s_r, s_i]_{t-1}
                     + [b_r_n, b_i_n] * u_t
        y_t          = sum_n (c_r_n * s_r_{t,n} + c_i_n * s_i_{t,n})
                     + D_skip * u_t

Properties
----------
* No complex tensors — every operator is a scalar add / multiply.
* No associative scan, no FFT, no Cauchy kernels.
* State is a fixed `(B, N, 2)` block; deployment uses one tiny step function.
* `decay_n in (0, 1)` parameterized as `sigmoid(s)` so the system is provably
  stable for every weight value.
* Damped rotations capture periodic structure (e.g. ECG/PPG cardiac cycles)
  that real-only diagonal SSMs cannot represent.
"""

from __future__ import annotations

import math

import keras
import numpy as np
from keras import ops


__all__ = ["DiagonalSSM", "S4DBlock"]


def _split_real_imag(x):
    """Split a final-axis pair into two halves: (..., 2) -> (..., ), (..., )."""
    return x[..., 0], x[..., 1]


class DiagonalSSM(keras.layers.Layer):
    """Diagonal SSM (LRU / S4D-Lin style) with damped-rotation poles.

    Args:
        state_size: Number of complex pole pairs ``N``. Each pair contributes
            two real states.
        output_dim: Output channel count ``H``. Defaults to input channel count.
        max_decay: Upper bound for ``decay_n`` so poles stay strictly inside
            the unit circle (recommended ``0.999``).
        min_decay: Lower bound (recommended ``0.0`` — allows fast-forgetting
            poles that act like one-tap convs).
        max_phase: Upper bound for ``theta_n`` in radians. ``pi`` covers the
            full Nyquist range.
        skip_connection: If ``True``, adds a learned ``D u_t`` feedthrough.
        return_state: If ``True``, returns ``(y, final_state)`` from ``call``.
        initial_state_trainable: If ``True``, learns the initial hidden state;
            otherwise it is zero.

    Input shape:  ``(batch, time, in_dim)``
    Output shape: ``(batch, time, output_dim)``
    """

    def __init__(
        self,
        state_size: int,
        output_dim: int | None = None,
        *,
        max_decay: float = 0.999,
        min_decay: float = 0.4,
        max_phase: float = 3.141592653589793,
        min_phase: float = 0.0,
        skip_connection: bool = True,
        return_state: bool = False,
        initial_state_trainable: bool = False,
        gamma_normalize: bool = True,
        **kwargs,
    ) -> None:
        """Build a DiagonalSSM.

        Default initialization follows Orvieto et al. 2023 (LRU): poles
        sampled uniformly in the disk annulus ``[min_decay, max_decay]``,
        phases uniform in ``[min_phase, max_phase]``, B/C drawn from a
        complex Gaussian, and B is multiplied by ``gamma = sqrt(1 - r^2)``
        so deeply-damped modes still receive useful input energy.
        """
        super().__init__(**kwargs)
        if state_size <= 0:
            raise ValueError("state_size must be positive.")
        if not 0.0 <= min_decay < max_decay < 1.0:
            raise ValueError(
                "Require 0 <= min_decay < max_decay < 1 for stability."
            )
        if max_phase <= 0.0 or min_phase < 0.0 or min_phase >= max_phase:
            raise ValueError("Require 0 <= min_phase < max_phase.")
        self.state_size = int(state_size)
        self.output_dim = None if output_dim is None else int(output_dim)
        self.max_decay = float(max_decay)
        self.min_decay = float(min_decay)
        self.max_phase = float(max_phase)
        self.min_phase = float(min_phase)
        self.skip_connection = bool(skip_connection)
        self.return_state = bool(return_state)
        self.initial_state_trainable = bool(initial_state_trainable)
        self.gamma_normalize = bool(gamma_normalize)

    # ------------------------------------------------------------------ build
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                f"DiagonalSSM expects (batch, time, dim) input; got {input_shape}"
            )
        in_dim = int(input_shape[-1])
        out_dim = self.output_dim if self.output_dim is not None else in_dim
        N = self.state_size
        self._in_dim = in_dim
        self._out_dim = out_dim

        # ----- LRU-style decay parameterization: r = exp(-exp(nu_log))
        # Sample target r uniformly in [min_decay, max_decay], then invert.
        # nu_log = log(-log(r))  -> r = exp(-exp(nu_log)).
        # Use numpy-based initializers so they are concrete during both eager
        # and symbolic Keras builds.
        rng = np.random.default_rng()

        class _NuLogInit(keras.initializers.Initializer):
            def __init__(self, lo: float, hi: float, seed: int):
                self.lo, self.hi, self.seed = lo, hi, seed

            def __call__(self, shape, dtype=None):
                r = np.random.default_rng(self.seed).uniform(
                    self.lo, self.hi, size=tuple(shape),
                ).astype("float32")
                r = np.clip(r, 1e-6, 1.0 - 1e-6)
                return np.log(-np.log(r)).astype("float32")

        class _PhaseLogitInit(keras.initializers.Initializer):
            def __init__(self, seed: int):
                self.seed = seed

            def __call__(self, shape, dtype=None):
                u = np.random.default_rng(self.seed).uniform(
                    1e-4, 1.0 - 1e-4, size=tuple(shape),
                ).astype("float32")
                return np.log(u / (1.0 - u)).astype("float32")

        seed_a = int(rng.integers(0, 2**31 - 1))
        seed_b = int(rng.integers(0, 2**31 - 1))
        self.decay_log = self.add_weight(
            name="decay_log", shape=(N,),
            initializer=_NuLogInit(self.min_decay, self.max_decay, seed_a),
            trainable=True,
        )
        self.phase_logit = self.add_weight(
            name="phase_logit", shape=(N,),
            initializer=_PhaseLogitInit(seed_b),
            trainable=True,
        )

        # ----- Complex Gaussian init for B and C, scaled like LRU.
        # B has variance 1/(2N) per real component (so |B|^2 ~ 1/N).
        # C has variance 1/N per real component, then output is scaled by 1/N
        # implicitly through the recurrence; matches the LRU paper.
        b_std = 1.0 / math.sqrt(2.0 * max(N, 1))
        c_std = 1.0 / math.sqrt(max(N, 1))
        self.B_real = self.add_weight(
            name="B_real", shape=(in_dim, N),
            initializer=keras.initializers.RandomNormal(stddev=b_std),
            trainable=True,
        )
        self.B_imag = self.add_weight(
            name="B_imag", shape=(in_dim, N),
            initializer=keras.initializers.RandomNormal(stddev=b_std),
            trainable=True,
        )
        self.C_real = self.add_weight(
            name="C_real", shape=(N, out_dim),
            initializer=keras.initializers.RandomNormal(stddev=c_std),
            trainable=True,
        )
        self.C_imag = self.add_weight(
            name="C_imag", shape=(N, out_dim),
            initializer=keras.initializers.RandomNormal(stddev=c_std),
            trainable=True,
        )

        # Optional skip
        if self.skip_connection:
            self.D_skip = self.add_weight(
                name="D_skip", shape=(in_dim, out_dim),
                initializer=keras.initializers.GlorotUniform(),
                trainable=True,
            )
        else:
            self.D_skip = None

        # Initial state
        if self.initial_state_trainable:
            self.s0 = self.add_weight(
                name="s0", shape=(N, 2),
                initializer=keras.initializers.Zeros(), trainable=True,
            )
        else:
            self.s0 = None

        super().build(input_shape)

    # ---------------------------------------------------- pole parameterization
    def _pole_params(self):
        """Return (decay, cos_theta, sin_theta, gamma) — each ``(N,)``.

        ``gamma = sqrt(1 - decay^2)`` is the LRU input-normalization factor
        applied to ``B`` so each pole's steady-state variance under white
        input is independent of its decay rate.
        """
        # Stable decay: r = exp(-exp(nu_log)) is in (0, 1) for any real nu.
        decay = ops.exp(-ops.exp(self.decay_log))
        # Hard clip to the configured ceiling for numerical safety.
        decay = ops.clip(decay, 0.0, self.max_decay)
        theta = self.min_phase + (self.max_phase - self.min_phase) * ops.sigmoid(
            self.phase_logit
        )
        if self.gamma_normalize:
            gamma = ops.sqrt(ops.maximum(1.0 - decay * decay, 1e-8))
        else:
            gamma = ops.ones_like(decay)
        return decay, ops.cos(theta), ops.sin(theta), gamma

    # ------------------------------------------------------------------- step
    def step(self, u_t, state):
        """Single-timestep update suitable for streaming inference.

        Args:
            u_t:    ``(batch, in_dim)`` input at time ``t``.
            state:  ``(batch, N, 2)`` previous (s_r, s_i).
        Returns:
            ``(y_t, new_state)`` with ``y_t`` shape ``(batch, out_dim)``.
        """
        decay, cos_t, sin_t, gamma = self._pole_params()  # each (N,)
        s_r = state[..., 0]
        s_i = state[..., 1]

        # Decay then rotate previous state.  Equivalent to multiplying complex
        # state by ``decay * exp(i theta)``.
        ds_r = decay * s_r
        ds_i = decay * s_i
        rot_r = cos_t * ds_r - sin_t * ds_i
        rot_i = sin_t * ds_r + cos_t * ds_i

        # Inject current input through gamma * B = gamma * (B_real + i B_imag).
        b_r = gamma * ops.matmul(u_t, self.B_real)  # (batch, N)
        b_i = gamma * ops.matmul(u_t, self.B_imag)
        new_s_r = rot_r + b_r
        new_s_i = rot_i + b_i

        # Read out via C = C_real + i C_imag (real part only):
        #     y = sum_n (Re(C_n) Re(s_n) - Im(C_n) Im(s_n))
        # but we keep a more flexible bilinear (lets gradients use both) to
        # avoid wasting Im(s) capacity when D_skip is absent.
        y = ops.matmul(new_s_r, self.C_real) + ops.matmul(new_s_i, self.C_imag)
        if self.D_skip is not None:
            y = y + ops.matmul(u_t, self.D_skip)

        new_state = ops.stack([new_s_r, new_s_i], axis=-1)
        return y, new_state

    # ------------------------------------------------------------------- call
    def call(self, u, initial_state=None, return_state=None):
        """Run the recurrence over the time axis.

        Args:
            u:               ``(batch, time, in_dim)``.
            initial_state:   optional ``(batch, N, 2)`` override.
            return_state:    overrides ``self.return_state`` if not ``None``.
        """
        u = ops.convert_to_tensor(u, dtype=self.compute_dtype)
        # Time-major iteration is implemented as an explicit Python loop over
        # ``ops.unstack`` so the same code traces cleanly on TF (autograph),
        # JAX (jit), and PyTorch (eager) — and avoids backend-specific scan
        # quirks (e.g. Keras-on-TF requires y.shape == carry.shape).
        u_steps = ops.unstack(u, axis=1)  # list[ (B, in_dim) ] of length T

        if initial_state is None:
            batch = ops.shape(u_steps[0])[0]
            if self.s0 is not None:
                state = ops.broadcast_to(
                    self.s0, (batch, self.state_size, 2)
                )
            else:
                state = ops.zeros((batch, self.state_size, 2), dtype=u.dtype)
        else:
            state = ops.cast(initial_state, u.dtype)

        ys: list = []
        for u_t in u_steps:
            y_t, state = self.step(u_t, state)
            ys.append(y_t)
        y = ops.stack(ys, axis=1)  # (B, T, out_dim)

        rs = self.return_state if return_state is None else bool(return_state)
        if rs:
            return y, state
        return y

    # ----------------------------------------------------------------- config
    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            state_size=self.state_size,
            output_dim=self.output_dim,
            max_decay=self.max_decay,
            min_decay=self.min_decay,
            max_phase=self.max_phase,
            min_phase=self.min_phase,
            skip_connection=self.skip_connection,
            return_state=self.return_state,
            initial_state_trainable=self.initial_state_trainable,
            gamma_normalize=self.gamma_normalize,
        )
        return cfg


class S4DBlock(keras.layers.Layer):
    """Residual SSM block: LayerNorm → DiagonalSSM → GELU → Dense → +residual.

    This is the recommended building block for stacking SSM layers in an
    encoder/decoder.  Channel count is preserved end-to-end so blocks compose
    without projection bookkeeping.
    """

    def __init__(
        self,
        state_size: int,
        *,
        max_phase: float = 3.141592653589793,
        max_decay: float = 0.999,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.state_size = int(state_size)
        self.max_phase = float(max_phase)
        self.max_decay = float(max_decay)
        self.dropout_rate = float(dropout)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.norm = keras.layers.LayerNormalization(epsilon=1e-5)
        self.ssm = DiagonalSSM(
            state_size=self.state_size,
            output_dim=in_dim,
            max_decay=self.max_decay,
            max_phase=self.max_phase,
            name="diagonal_ssm",
        )
        self.proj = keras.layers.Dense(in_dim, name="proj")
        self.act = keras.layers.Activation("gelu")
        self.drop = (
            keras.layers.Dropout(self.dropout_rate)
            if self.dropout_rate > 0.0 else None
        )
        super().build(input_shape)

    def call(self, x, training=False):
        h = self.norm(x)
        h = self.ssm(h)
        h = self.act(h)
        h = self.proj(h)
        if self.drop is not None:
            h = self.drop(h, training=training)
        return x + h

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            state_size=self.state_size,
            max_phase=self.max_phase,
            max_decay=self.max_decay,
            dropout=self.dropout_rate,
        )
        return cfg
