"""Calibrated noise harness for synthetic physiology signals.

Given a clean ground-truth waveform, :func:`add_noise` produces a noisy
version at a target SNR with a chosen mixture of physiologically realistic
noise sources:

* ``baseline_wander`` -- low-frequency drift (random walk + slow sinusoid).
* ``emg`` -- bandpassed Gaussian noise in the EMG band (20-150 Hz for ECG,
  high-band for PPG).
* ``motion`` -- intermittent windowed bursts simulating motion artefact.
* ``powerline`` -- pure tone at 50 or 60 Hz with small frequency wobble.
* ``electrode_pop`` -- sparse step + exponential decay events.
* ``gauss`` -- broadband Gaussian (sensor / quantisation floor).

All components are weighted, summed, then rescaled to hit the requested
``snr_db`` measured as ``10*log10(P_signal / P_noise)`` over the full window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.signal import butter, sosfiltfilt


NoiseKind = Literal[
    "baseline_wander",
    "emg",
    "motion",
    "powerline",
    "electrode_pop",
    "gauss",
]


@dataclass
class NoiseSpec:
    """Recipe for a noise mixture.

    Attributes:
        weights: Relative power weights for each noise component. Components
            with weight ``0`` (or absent) are skipped. Weights are normalised
            internally before SNR-scaling.
        powerline_hz: Mains frequency (50.0 or 60.0).
        emg_band_hz: Low/high band edges for the EMG-style component.
        motion_burst_rate_hz: Average rate of motion bursts (events / second).
        motion_burst_duration_s: Mean burst duration.
        electrode_pop_rate_hz: Average rate of electrode-pop events.
    """

    weights: dict[NoiseKind, float] = field(
        default_factory=lambda: {
            "baseline_wander": 1.0,
            "emg": 1.0,
            "powerline": 0.3,
            "gauss": 0.5,
        }
    )
    powerline_hz: float = 60.0
    emg_band_hz: tuple[float, float] = (20.0, 150.0)
    motion_burst_rate_hz: float = 0.2
    motion_burst_duration_s: float = 0.5
    electrode_pop_rate_hz: float = 0.05


def _baseline_wander(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    # Random walk + slow sinusoid at ~0.15 Hz (respiration-ish drift).
    walk = np.cumsum(rng.standard_normal(n))
    if walk.std() > 0:
        walk = walk / walk.std()
    t = np.arange(n) / fs
    sinu = np.sin(2.0 * np.pi * 0.15 * t + rng.uniform(0, 2 * np.pi))
    out = 0.6 * walk + 0.4 * sinu
    out -= out.mean()
    return out


def _emg(
    n: int, fs: float, band: tuple[float, float], rng: np.random.Generator
) -> np.ndarray:
    raw = rng.standard_normal(n)
    nyq = 0.5 * fs
    low = max(band[0], 0.5) / nyq
    high = min(band[1], nyq * 0.95) / nyq
    if low >= high:
        return raw - raw.mean()
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, raw)


def _motion(
    n: int,
    fs: float,
    rate_hz: float,
    duration_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    if rate_hz <= 0.0:
        return out
    total_s = n / fs
    n_events = rng.poisson(rate_hz * total_s)
    for _ in range(int(n_events)):
        centre = int(rng.uniform(0, n))
        dur = max(int(rng.normal(duration_s, 0.25 * duration_s) * fs), max(int(0.05 * fs), 10))
        amp = rng.normal(0.0, 1.0)
        # Half-cosine envelope filled with low-band noise.
        env_x = np.linspace(-1.0, 1.0, dur)
        env = 0.5 * (1.0 + np.cos(np.pi * env_x))
        burst = amp * env * rng.standard_normal(dur)
        # Low-pass the burst content (motion is mostly < 20 Hz).
        nyq = 0.5 * fs
        cutoff = min(20.0, nyq * 0.9) / nyq
        sos = butter(2, cutoff, btype="lowpass", output="sos")
        burst = sosfiltfilt(sos, burst)
        s0 = max(centre - dur // 2, 0)
        s1 = min(s0 + dur, n)
        out[s0:s1] += burst[: s1 - s0]
    return out


def _powerline(n: int, fs: float, f0: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / fs
    # Small frequency wobble for realism.
    wobble = 0.05 * np.sin(2.0 * np.pi * 0.5 * t + rng.uniform(0, 2 * np.pi))
    phi = 2.0 * np.pi * (f0 * t + wobble) + rng.uniform(0, 2 * np.pi)
    return np.sin(phi)


def _electrode_pop(
    n: int, fs: float, rate_hz: float, rng: np.random.Generator
) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    if rate_hz <= 0.0:
        return out
    total_s = n / fs
    n_events = rng.poisson(rate_hz * total_s)
    for _ in range(int(n_events)):
        idx = int(rng.uniform(0, n))
        amp = rng.normal(0.0, 1.0)
        tau = rng.uniform(0.05, 0.3)  # seconds
        decay = np.exp(-np.arange(n - idx) / (tau * fs))
        out[idx:] += amp * decay
    return out


def _gauss(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def _generate_component(
    kind: NoiseKind,
    n: int,
    fs: float,
    spec: NoiseSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    if kind == "baseline_wander":
        return _baseline_wander(n, fs, rng)
    if kind == "emg":
        return _emg(n, fs, spec.emg_band_hz, rng)
    if kind == "motion":
        return _motion(n, fs, spec.motion_burst_rate_hz, spec.motion_burst_duration_s, rng)
    if kind == "powerline":
        return _powerline(n, fs, spec.powerline_hz, rng)
    if kind == "electrode_pop":
        return _electrode_pop(n, fs, spec.electrode_pop_rate_hz, rng)
    if kind == "gauss":
        return _gauss(n, rng)
    raise ValueError(f"Unknown noise kind: {kind!r}")


def add_noise(
    clean: np.ndarray,
    *,
    sample_rate: float,
    snr_db: float,
    spec: NoiseSpec | None = None,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Add a calibrated noise mixture to a clean signal.

    Each requested component is generated independently, scaled to unit
    variance, weighted, summed, then the combined noise is rescaled so that
    ``10*log10(var(clean) / var(noise)) == snr_db``.

    Args:
        clean: 1-D ground-truth signal.
        sample_rate: Sample rate of ``clean`` (Hz).
        snr_db: Target signal-to-noise ratio in dB.
        spec: Mixture recipe. Defaults to a balanced ECG-friendly mix.
        seed: RNG seed. ``None`` = nondeterministic.

    Returns:
        Tuple of ``(noisy, noise)`` -- both same shape as ``clean``.
        ``noisy = clean + noise``.
    """
    if spec is None:
        spec = NoiseSpec()
    rng = np.random.default_rng(seed)

    clean = np.asarray(clean, dtype=np.float64)
    n = clean.shape[0]
    components: list[np.ndarray] = []
    weights: list[float] = []
    for kind, w in spec.weights.items():
        if w <= 0.0:
            continue
        comp = _generate_component(kind, n, sample_rate, spec, rng)
        comp = comp - comp.mean()
        std = comp.std()
        if std > 1e-12:
            comp = comp / std
        components.append(comp)
        weights.append(float(w))

    if not components:
        return clean.copy(), np.zeros_like(clean)

    w_arr = np.asarray(weights, dtype=np.float64)
    w_arr = w_arr / w_arr.sum()
    noise = np.zeros(n, dtype=np.float64)
    for w, comp in zip(w_arr, components):
        noise += np.sqrt(w) * comp  # combine in power, not amplitude

    # Rescale combined noise to hit the SNR target.
    signal_power = float(np.var(clean))
    noise_power = float(np.var(noise))
    if signal_power <= 0.0 or noise_power <= 0.0:
        return clean.copy(), np.zeros_like(clean)

    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise *= np.sqrt(target_noise_power / noise_power)
    return clean + noise, noise
