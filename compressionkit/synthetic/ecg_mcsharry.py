"""McSharry-Clifford dynamical ECG model.

Reference:
    McSharry, Clifford, Tarassenko, Smith,
    "A dynamical model for generating synthetic electrocardiogram signals,"
    IEEE Trans. Biomed. Eng. 50(3):289-294, 2003.

The model integrates a 3-D ODE whose ``z`` coordinate is the ECG voltage.
A trajectory rotates around the unit circle in ``(x, y)`` at angular velocity
``omega = 2*pi*HR``. Each fiducial event (P, Q, R, S, T) is a Gaussian "bump"
placed at a specific angle ``theta_i``; as the trajectory sweeps past that
angle the ``z`` coordinate is pushed by ``a_i * exp(-dtheta^2 / (2 b_i^2))``.

Heart-rate variability is injected by modulating ``omega(t)`` with a bimodal
(LF + HF) noise process. Respiration is modelled as a slow sinusoid on the
``z`` baseline.

Everything is fixed-shape NumPy with explicit seeds -- portable, deterministic,
no learned weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Default Lead II-ish fiducial parameters from the original paper
# (theta in radians, a is amplitude, b is angular width).
_DEFAULT_THETA = np.array([-np.pi / 3.0, -np.pi / 12.0, 0.0, np.pi / 12.0, np.pi / 2.0])
_DEFAULT_A = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
_DEFAULT_B = np.array([0.25, 0.1, 0.1, 0.1, 0.4])


@dataclass
class EcgMorphologyParams:
    """Per-fiducial Gaussian parameters for the McSharry ECG model.

    Order is always ``[P, Q, R, S, T]``. Extra entries (e.g. an ST-elevation
    bump or a U-wave) can be appended -- the integrator handles any length.

    Attributes:
        theta: Angular location of each Gaussian, radians on ``[-pi, pi]``.
        a: Amplitude of each Gaussian (mV-ish, arbitrary scale).
        b: Angular width (std) of each Gaussian.
    """

    theta: np.ndarray = field(default_factory=lambda: _DEFAULT_THETA.copy())
    a: np.ndarray = field(default_factory=lambda: _DEFAULT_A.copy())
    b: np.ndarray = field(default_factory=lambda: _DEFAULT_B.copy())

    def __post_init__(self) -> None:
        self.theta = np.asarray(self.theta, dtype=np.float64)
        self.a = np.asarray(self.a, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64)
        if not (self.theta.shape == self.a.shape == self.b.shape):
            raise ValueError("theta, a, b must have the same shape")
        if self.theta.ndim != 1:
            raise ValueError("theta, a, b must be 1-D arrays")


def _rr_process(
    duration_s: float,
    hr_mean: float,
    hr_std: float,
    *,
    lf_hz: float = 0.1,
    hf_hz: float = 0.25,
    lf_hf_ratio: float = 0.5,
    fs_internal: float = 256.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a continuous instantaneous-HR signal with realistic HRV.

    Uses the McSharry RR-spectrum trick: sum of two narrow Gaussians in the
    frequency domain (LF around 0.1 Hz, HF around 0.25 Hz) inverse-FFT'd with
    random phase. The result is added to ``hr_mean`` and scaled to match
    ``hr_std``.

    Returns:
        Array of instantaneous HR in beats/min, length ``duration_s *
        fs_internal``.
    """
    n = int(round(duration_s * fs_internal))
    if n < 2:
        return np.full(max(n, 1), hr_mean, dtype=np.float64)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs_internal)
    # Two Gaussian power peaks in the RR spectrum.
    lf_sigma = 0.01
    hf_sigma = 0.01
    s_lf = np.exp(-0.5 * ((freqs - lf_hz) / lf_sigma) ** 2)
    s_hf = np.exp(-0.5 * ((freqs - hf_hz) / hf_sigma) ** 2)
    spectrum = lf_hf_ratio * s_lf + s_hf
    spectrum[0] = 0.0  # zero DC

    phases = rng.uniform(-np.pi, np.pi, size=freqs.shape)
    coeffs = np.sqrt(spectrum) * np.exp(1j * phases)
    rr_noise = np.fft.irfft(coeffs, n=n).real

    if rr_noise.std() > 1e-12 and hr_std > 0.0:
        rr_noise = rr_noise / rr_noise.std() * hr_std
    else:
        rr_noise = np.zeros_like(rr_noise)

    return hr_mean + rr_noise


def _dz_dt(
    theta: float,
    z: float,
    baseline: float,
    morph: EcgMorphologyParams,
) -> float:
    """RHS of the dz/dt equation: sum of Gaussian pushes + baseline restoring."""
    dtheta = np.remainder(theta - morph.theta + np.pi, 2.0 * np.pi) - np.pi
    bumps = morph.a * dtheta * np.exp(-0.5 * (dtheta / morph.b) ** 2)
    return -float(np.sum(bumps)) - (z - baseline)


def ecg_mcsharry(
    duration_s: float,
    sample_rate: float,
    *,
    hr_mean: float = 60.0,
    hr_std: float = 1.5,
    morphology: EcgMorphologyParams | None = None,
    respiration_hz: float = 0.25,
    respiration_amplitude: float = 0.0,
    fs_internal: float = 512.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Generate a clean synthetic ECG via the McSharry dynamical model.

    Integrates the 3-D ODE with explicit RK4 at ``fs_internal`` Hz, then
    decimates to ``sample_rate`` by simple stride pick. The trajectory rides
    the unit circle in ``(x, y)`` so ``alpha = 1 - sqrt(x^2 + y^2)`` is a
    stable limit-cycle attractor; we initialise on the circle and let the
    integrator handle drift.

    Args:
        duration_s: Output length in seconds.
        sample_rate: Output sample rate (Hz).
        hr_mean: Mean heart rate (beats / min).
        hr_std: Std-dev of the HRV process (beats / min). Set to ``0`` for
            a perfectly periodic ECG.
        morphology: Per-fiducial Gaussian parameters. Defaults to the
            paper's Lead II-like parameters.
        respiration_hz: Respiration frequency for baseline modulation.
        respiration_amplitude: Amplitude of the respiration-induced
            baseline drift (same units as ECG). ``0`` disables.
        fs_internal: Integration sample rate. Should be >= 256 Hz; the paper
            uses 256 Hz.
        seed: RNG seed for the HRV process. ``None`` = nondeterministic.

    Returns:
        1-D float array of length ``round(duration_s * sample_rate)``.
    """
    if morphology is None:
        morphology = EcgMorphologyParams()

    rng = np.random.default_rng(seed)
    n_internal = int(round(duration_s * fs_internal))
    dt = 1.0 / fs_internal

    hr_inst = _rr_process(
        duration_s,
        hr_mean=hr_mean,
        hr_std=hr_std,
        fs_internal=fs_internal,
        rng=rng,
    )
    omega = 2.0 * np.pi * hr_inst / 60.0  # rad/s

    # Initial conditions on the unit circle, just before the P wave.
    x = 1.0
    y = 0.0
    z = 0.0

    z_out = np.empty(n_internal, dtype=np.float64)

    for k in range(n_internal):
        w = omega[k]
        baseline = (
            respiration_amplitude * np.sin(2.0 * np.pi * respiration_hz * k * dt)
            if respiration_amplitude > 0.0
            else 0.0
        )

        # RK4 step on (x, y, z)
        def rhs(xs: float, ys: float, zs: float, theta: float) -> tuple[float, float, float]:
            alpha = 1.0 - np.sqrt(xs * xs + ys * ys)
            dx = alpha * xs - w * ys
            dy = alpha * ys + w * xs
            dz = _dz_dt(theta, zs, baseline, morphology)
            return dx, dy, dz

        theta_now = np.arctan2(y, x)
        k1x, k1y, k1z = rhs(x, y, z, theta_now)
        k2x, k2y, k2z = rhs(
            x + 0.5 * dt * k1x,
            y + 0.5 * dt * k1y,
            z + 0.5 * dt * k1z,
            np.arctan2(y + 0.5 * dt * k1y, x + 0.5 * dt * k1x),
        )
        k3x, k3y, k3z = rhs(
            x + 0.5 * dt * k2x,
            y + 0.5 * dt * k2y,
            z + 0.5 * dt * k2z,
            np.arctan2(y + 0.5 * dt * k2y, x + 0.5 * dt * k2x),
        )
        k4x, k4y, k4z = rhs(
            x + dt * k3x,
            y + dt * k3y,
            z + dt * k3z,
            np.arctan2(y + dt * k3y, x + dt * k3x),
        )

        x += dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        y += dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        z += dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
        z_out[k] = z

    # Decimate to requested sample rate by linear interpolation.
    n_out = int(round(duration_s * sample_rate))
    t_in = np.arange(n_internal) / fs_internal
    t_out = np.arange(n_out) / sample_rate
    return np.interp(t_out, t_in, z_out)
