"""Dynamical PPG model in the McSharry style.

Same scaffold as the ECG model: a 2-D limit-cycle in ``(x, y)`` provides the
cardiac phase, and ``z`` is the PPG voltage produced by a small set of
Gaussian "bumps" placed around the unit circle.

Default morphology uses three Gaussians:

* Systolic peak at ``theta ~= 0`` (large, positive).
* Dicrotic notch at ``theta ~= +pi/3`` (small, negative).
* Diastolic peak at ``theta ~= +pi/2`` (medium, positive).

Respiration is injected as a slow amplitude modulation on the systolic-peak
amplitude (a stand-in for respiratory-induced perfusion variation) plus a
slow baseline drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_DEFAULT_THETA = np.array([0.0, np.pi / 3.0, np.pi / 2.0])
_DEFAULT_A = np.array([12.0, -1.5, 3.0])
_DEFAULT_B = np.array([0.35, 0.18, 0.30])


@dataclass
class PpgMorphologyParams:
    """Per-fiducial Gaussian parameters for the dynamical PPG model.

    Default order is ``[systolic, dicrotic_notch, diastolic]`` but the model
    accepts any length 1-D arrays.
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
    """Bimodal-spectrum HRV generator, same as the ECG model."""
    n = int(round(duration_s * fs_internal))
    if n < 2:
        return np.full(max(n, 1), hr_mean, dtype=np.float64)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs_internal)
    s_lf = np.exp(-0.5 * ((freqs - lf_hz) / 0.01) ** 2)
    s_hf = np.exp(-0.5 * ((freqs - hf_hz) / 0.01) ** 2)
    spectrum = lf_hf_ratio * s_lf + s_hf
    spectrum[0] = 0.0

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
    a_scale: float,
    morph: PpgMorphologyParams,
) -> float:
    dtheta = np.remainder(theta - morph.theta + np.pi, 2.0 * np.pi) - np.pi
    bumps = a_scale * morph.a * dtheta * np.exp(-0.5 * (dtheta / morph.b) ** 2)
    return -float(np.sum(bumps)) - (z - baseline)


def ppg_dynamical(
    duration_s: float,
    sample_rate: float,
    *,
    hr_mean: float = 72.0,
    hr_std: float = 1.0,
    morphology: PpgMorphologyParams | None = None,
    respiration_hz: float = 0.25,
    respiration_amplitude_mod: float = 0.05,
    respiration_baseline_amplitude: float = 0.0,
    fs_internal: float = 256.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Generate a clean synthetic PPG via a McSharry-style dynamical model.

    Args:
        duration_s: Output length (s).
        sample_rate: Output sample rate (Hz).
        hr_mean: Mean heart rate (bpm).
        hr_std: HRV std-dev (bpm).
        morphology: Per-fiducial Gaussian parameters. Defaults to a 3-Gaussian
            (systolic / dicrotic / diastolic) configuration.
        respiration_hz: Respiration frequency (Hz).
        respiration_amplitude_mod: Fractional AC amplitude modulation depth
            from respiration (e.g. ``0.05`` = ±5% systolic-peak swing).
        respiration_baseline_amplitude: Absolute baseline drift amplitude.
        fs_internal: ODE integration rate (Hz).
        seed: RNG seed for HRV. ``None`` = nondeterministic.

    Returns:
        1-D float array of length ``round(duration_s * sample_rate)``.
    """
    if morphology is None:
        morphology = PpgMorphologyParams()

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
    omega = 2.0 * np.pi * hr_inst / 60.0

    x = 1.0
    y = 0.0
    z = 0.0

    z_out = np.empty(n_internal, dtype=np.float64)
    two_pi_fr = 2.0 * np.pi * respiration_hz

    for k in range(n_internal):
        w = omega[k]
        t = k * dt
        a_scale = 1.0 + respiration_amplitude_mod * np.sin(two_pi_fr * t)
        baseline = (
            respiration_baseline_amplitude * np.sin(two_pi_fr * t)
            if respiration_baseline_amplitude > 0.0
            else 0.0
        )

        def rhs(xs: float, ys: float, zs: float, theta: float) -> tuple[float, float, float]:
            alpha = 1.0 - np.sqrt(xs * xs + ys * ys)
            dx = alpha * xs - w * ys
            dy = alpha * ys + w * xs
            dz = _dz_dt(theta, zs, baseline, a_scale, morphology)
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

    n_out = int(round(duration_s * sample_rate))
    t_in = np.arange(n_internal) / fs_internal
    t_out = np.arange(n_out) / sample_rate
    return np.interp(t_out, t_in, z_out)
