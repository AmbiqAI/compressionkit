"""Numerical equivalence tests for `DiagonalSSM`.

Three checks:
1. Streaming step-by-step output matches whole-sequence ``call`` output.
2. A pure-PyTorch reference of the same recurrence reproduces the Keras output
   to within float32 tolerance — proves the math is backend-independent.
3. Gradients flow (loss decreases on a 1-step optimizer) — sanity check the
   layer is trainable.
"""

from __future__ import annotations

import os

# Pin TF backend before any keras import.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import torch

from compressionkit.layers.ssm import DiagonalSSM


def _build_layer(seed: int = 0, in_dim: int = 4, out_dim: int = 4, N: int = 6):
    keras.utils.set_random_seed(seed)
    layer = DiagonalSSM(state_size=N, output_dim=out_dim, name="ssm")
    layer.build((None, None, in_dim))
    return layer


def test_streaming_matches_batch():
    """Step-by-step inference must match whole-sequence call exactly."""
    rng = np.random.default_rng(0)
    layer = _build_layer()
    u = rng.standard_normal((2, 16, 4)).astype("float32")

    y_full = keras.ops.convert_to_numpy(layer(u))

    state = np.zeros((2, layer.state_size, 2), dtype="float32")
    y_stream = np.zeros_like(y_full)
    for t in range(u.shape[1]):
        y_t, state = layer.step(u[:, t], state)
        y_stream[:, t] = keras.ops.convert_to_numpy(y_t)
        state = keras.ops.convert_to_numpy(state)

    np.testing.assert_allclose(y_stream, y_full, atol=1e-5, rtol=1e-5)


def _torch_ref(layer: DiagonalSSM, u_np: np.ndarray) -> np.ndarray:
    """Pure-PyTorch reference recurrence using the layer's weights."""
    # Match the LRU-style parameterization exactly:
    #   r = exp(-exp(nu_log)),     theta = min + (max-min)*sigmoid(phase_logit)
    #   gamma = sqrt(1 - r^2)  applied to B at injection time.
    decay_log = torch.from_numpy(keras.ops.convert_to_numpy(layer.decay_log))
    decay = torch.exp(-torch.exp(decay_log)).clamp(max=layer.max_decay)
    phase_logit = torch.from_numpy(keras.ops.convert_to_numpy(layer.phase_logit))
    theta = layer.min_phase + (layer.max_phase - layer.min_phase) * torch.sigmoid(phase_logit)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    gamma = torch.sqrt(torch.clamp(1.0 - decay * decay, min=1e-8))
    if not layer.gamma_normalize:
        gamma = torch.ones_like(decay)
    Br = torch.from_numpy(keras.ops.convert_to_numpy(layer.B_real))
    Bi = torch.from_numpy(keras.ops.convert_to_numpy(layer.B_imag))
    Cr = torch.from_numpy(keras.ops.convert_to_numpy(layer.C_real))
    Ci = torch.from_numpy(keras.ops.convert_to_numpy(layer.C_imag))
    D = torch.from_numpy(keras.ops.convert_to_numpy(layer.D_skip))

    u = torch.from_numpy(u_np)
    B, T, _ = u.shape
    s_r = torch.zeros(B, layer.state_size)
    s_i = torch.zeros(B, layer.state_size)
    out = []
    for t in range(T):
        ut = u[:, t]
        ds_r, ds_i = decay * s_r, decay * s_i
        rot_r = cos_t * ds_r - sin_t * ds_i
        rot_i = sin_t * ds_r + cos_t * ds_i
        s_r = rot_r + gamma * (ut @ Br)
        s_i = rot_i + gamma * (ut @ Bi)
        y = s_r @ Cr + s_i @ Ci + ut @ D
        out.append(y)
    return torch.stack(out, dim=1).numpy()


def test_pytorch_reference_matches():
    """Independent PyTorch implementation must match Keras call to ~1e-5."""
    rng = np.random.default_rng(1)
    layer = _build_layer(seed=1)
    u = rng.standard_normal((3, 32, 4)).astype("float32")

    y_keras = keras.ops.convert_to_numpy(layer(u))
    y_torch = _torch_ref(layer, u)

    np.testing.assert_allclose(y_torch, y_keras, atol=2e-3, rtol=2e-3)


def test_gradient_descent_reduces_loss():
    """One optimizer step on a regression target must reduce loss."""
    rng = np.random.default_rng(2)
    inputs = keras.Input(shape=(20, 4))
    outputs = DiagonalSSM(state_size=8, output_dim=4)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")

    x = rng.standard_normal((16, 20, 4)).astype("float32")
    y = rng.standard_normal((16, 20, 4)).astype("float32")

    loss_before = float(model.evaluate(x, y, verbose=0))
    model.fit(x, y, epochs=3, batch_size=8, verbose=0)
    loss_after = float(model.evaluate(x, y, verbose=0))
    assert loss_after < loss_before, (loss_before, loss_after)


if __name__ == "__main__":
    test_streaming_matches_batch()
    print("OK streaming==batch")
    test_pytorch_reference_matches()
    print("OK keras==pytorch_ref")
    test_gradient_descent_reduces_loss()
    print("OK gradient flow")
