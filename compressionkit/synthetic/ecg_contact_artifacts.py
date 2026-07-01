"""Synthetic ECG contact-failure artifact generators.

These helpers model artifact-dominated regimes that are not well covered by a
residual-noise bank alone: colored high-impedance noise, mains pickup, motion
bursts, lead-off gating, and near-pure-artifact weak-leak conditions.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_CONTACT_FAMILIES: tuple[str, ...] = (
    "colored",
    "mains",
    "motion",
    "lead_off",
    "weak_leak",
)
DEFAULT_CONTACT_SEVERITIES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


def normalize_window(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    return (out - out.mean()) / (out.std() + 1e-9)


def sample_empirical_residual(
    noise_bank: np.ndarray | None,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_bank is None or len(noise_bank) == 0:
        return np.zeros((length,), dtype=np.float32)
    segment = noise_bank[int(rng.integers(0, len(noise_bank)))].astype(np.float32)
    if segment.shape[0] < length:
        reps = int(np.ceil(length / segment.shape[0]))
        segment = np.tile(segment, reps)[:length]
    elif segment.shape[0] > length:
        start = int(rng.integers(0, segment.shape[0] - length + 1))
        segment = segment[start : start + length]
    return normalize_window(segment)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _colored_noise(length: int, rng: np.random.Generator, *, alpha: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(length)
    spectrum = rng.normal(size=freqs.shape) + 1j * rng.normal(size=freqs.shape)
    scale = np.ones_like(freqs)
    if len(freqs) > 1:
        scale[1:] = 1.0 / np.maximum(freqs[1:], 1.0 / length) ** (alpha / 2.0)
    noise = np.fft.irfft(spectrum * scale, n=length).astype(np.float32)
    return normalize_window(noise)


def _mains_component(length: int, sample_rate: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / float(sample_rate)
    base_freq = float(rng.choice([50.0, 60.0]))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    phase2 = float(rng.uniform(0.0, 2.0 * math.pi))
    drift_hz = float(rng.uniform(0.08, 0.35))
    envelope = 1.0 + 0.35 * np.sin(2.0 * math.pi * drift_hz * t + phase2)
    mains = envelope * (
        np.sin(2.0 * math.pi * base_freq * t + phase) + 0.35 * np.sin(2.0 * math.pi * 2.0 * base_freq * t + 0.3 * phase)
    )
    return normalize_window(mains.astype(np.float32))


def _motion_component(length: int, sample_rate: float, rng: np.random.Generator) -> np.ndarray:
    artifact = 0.4 * _colored_noise(length, rng, alpha=1.7)
    burst_count = int(rng.integers(1, 4))
    for _ in range(burst_count):
        width = int(rng.integers(max(8, length // 40), max(16, length // 8)))
        start = int(rng.integers(0, max(1, length - width)))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        window = np.hanning(width).astype(np.float32)
        artifact[start : start + width] += sign * float(rng.uniform(1.0, 2.4)) * window

    pop_count = int(rng.integers(1, 3))
    for _ in range(pop_count):
        start = int(rng.integers(0, length))
        tau = float(rng.uniform(0.02, 0.12))
        decay = np.exp(-np.arange(length - start, dtype=np.float32) / (tau * sample_rate + 1e-6))
        artifact[start:] += float(rng.uniform(-1.8, 1.8)) * decay

    return normalize_window(artifact.astype(np.float32))


def _lead_off_gate(length: int, severity: float, rng: np.random.Generator) -> np.ndarray:
    gate = np.ones((length,), dtype=np.float32)
    segment_count = 1 if severity < 0.55 else 2
    min_gate = float(np.interp(severity, [0.0, 1.0], [0.55, 0.02]))
    for _ in range(segment_count):
        width = int(np.interp(severity, [0.0, 1.0], [length * 0.18, length * 0.65]))
        width = int(np.clip(width * float(rng.uniform(0.75, 1.2)), 12, length))
        start = int(rng.integers(0, max(1, length - width + 1)))
        edge = max(4, width // 10)
        ramp = _smoothstep(np.linspace(0.0, 1.0, edge, dtype=np.float32))
        hole = np.ones((width,), dtype=np.float32) * min_gate
        hole[:edge] = 1.0 - (1.0 - min_gate) * ramp
        hole[-edge:] = min_gate + (1.0 - min_gate) * ramp
        gate[start : start + width] *= hole
    return gate


def _family_artifact(
    family: str,
    length: int,
    sample_rate: float,
    severity: float,
    rng: np.random.Generator,
    noise_bank: np.ndarray | None,
) -> np.ndarray:
    if family == "colored":
        alpha = float(np.interp(severity, [0.0, 1.0], [0.9, 1.8]))
        artifact = 0.85 * _colored_noise(length, rng, alpha=alpha)
        artifact += 0.35 * sample_empirical_residual(noise_bank, length, rng)
        artifact += 0.15 * normalize_window(rng.normal(size=length).astype(np.float32))
        return normalize_window(artifact)

    if family == "mains":
        artifact = 0.9 * _mains_component(length, sample_rate, rng)
        artifact += 0.45 * _colored_noise(length, rng, alpha=1.4)
        artifact += 0.2 * sample_empirical_residual(noise_bank, length, rng)
        return normalize_window(artifact)

    if family == "motion":
        artifact = _motion_component(length, sample_rate, rng)
        artifact += 0.3 * sample_empirical_residual(noise_bank, length, rng)
        return normalize_window(artifact)

    if family in {"lead_off", "weak_leak", "pure_artifact"}:
        artifact = 0.7 * _colored_noise(length, rng, alpha=1.6)
        artifact += 0.55 * _motion_component(length, sample_rate, rng)
        artifact += 0.45 * _mains_component(length, sample_rate, rng)
        artifact += 0.35 * sample_empirical_residual(noise_bank, length, rng)
        return normalize_window(artifact)

    raise ValueError(f"Unknown contact artifact family: {family}")


# Per-family artifact *power fraction* at severity 0 and severity 1. The fraction
# is the share of total signal power held by the artifact, so severity 1 means the
# window is almost entirely artifact and severity 0 is a mild, clearly ECG-bearing
# contamination. ``lead_off`` and ``weak_leak`` additionally attenuate the ECG
# itself (gated contact / weak leakage), which is their distinguishing character.
_FAMILY_POWER_FRACTION: dict[str, tuple[float, float]] = {
    "colored": (0.15, 0.95),
    "mains": (0.15, 0.95),
    "motion": (0.18, 0.96),
    "lead_off": (0.15, 0.96),
    "weak_leak": (0.20, 0.97),
    "pure_artifact": (1.0, 1.0),
}


def _artifact_power_fraction(family: str, severity: float) -> float:
    lo, hi = _FAMILY_POWER_FRACTION.get(family, (0.15, 0.95))
    return float(np.interp(severity, [0.0, 1.0], [lo, hi]))


def _mix_at_power_fraction(clean_unit: np.ndarray, artifact_unit: np.ndarray, fraction: float) -> np.ndarray:
    """Mix unit-RMS clean and artifact so the artifact holds ``fraction`` of power.

    Both inputs are assumed to be (approximately) unit-RMS. Because the scales are
    ``sqrt(1 - fraction)`` and ``sqrt(fraction)``, the artifact-to-total power ratio
    equals ``fraction`` whenever the two components are roughly uncorrelated. Any
    later normalization rescales both equally and preserves that ratio.
    """
    fraction = float(np.clip(fraction, 0.0, 1.0))
    clean_scale = math.sqrt(max(1e-6, 1.0 - fraction))
    artifact_scale = math.sqrt(fraction)
    return (clean_scale * clean_unit + artifact_scale * artifact_unit).astype(np.float32)


def _weak_leak_gain_envelope(length: int, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Slow, intermittent low gain so the ECG only weakly leaks through."""
    base = float(np.interp(severity, [0.0, 1.0], [0.85, 0.12]))
    n_ctrl = max(2, length // 128)
    ctrl = rng.uniform(0.35, 1.0, size=n_ctrl).astype(np.float32)
    env = np.interp(
        np.linspace(0.0, 1.0, length, dtype=np.float32),
        np.linspace(0.0, 1.0, n_ctrl, dtype=np.float32),
        ctrl,
    ).astype(np.float32)
    env = env / (float(env.max()) + 1e-9)
    return (base * env).astype(np.float32)


def measured_artifact_fraction(clean: np.ndarray, corrupted: np.ndarray) -> float:
    """Empirical artifact power share given the original clean and corrupted windows."""
    clean_unit = normalize_window(clean)
    corrupted_unit = normalize_window(corrupted)
    residual = corrupted_unit - clean_unit
    sig_power = float(np.mean(clean_unit**2)) + 1e-12
    res_power = float(np.mean(residual**2)) + 1e-12
    return float(res_power / (res_power + sig_power))


def synthesize_artifact_waveform(
    family: str,
    length: int,
    *,
    sample_rate: float,
    severity: float,
    rng: np.random.Generator,
    noise_bank: np.ndarray | None = None,
) -> np.ndarray:
    """Return a normalized artifact waveform for ``family`` (no clean signal).

    This exposes the family synthesis used inside
    :func:`simulate_contact_artifact` so that an offline *artifact bank* can be
    precomputed once and sampled cheaply in-graph during training.
    """
    return _family_artifact(family, length, sample_rate, float(severity), rng, noise_bank)


def simulate_contact_artifact(
    clean: np.ndarray,
    *,
    family: str,
    severity: float,
    sample_rate: float,
    rng: np.random.Generator,
    noise_bank: np.ndarray | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Corrupt a clean ECG window with a contact-failure style artifact.

    Severity is calibrated to an artifact *power fraction*: severity 1.0 means the
    window is almost fully dominated by the artifact, while low severities stay
    clearly ECG-bearing. ``lead_off`` and ``weak_leak`` additionally suppress the
    underlying ECG (gated contact and weak intermittent leakage) so their character
    differs from the purely additive families even at matched power fraction.
    """
    clean_arr = np.asarray(clean, dtype=np.float32)
    severity = float(np.clip(severity, 0.0, 1.0))
    clean_unit = normalize_window(clean_arr)
    artifact = _family_artifact(family, clean_arr.shape[0], sample_rate, severity, rng, noise_bank)
    fraction = _artifact_power_fraction(family, severity)

    if family in {"colored", "mains", "motion"}:
        mixed = _mix_at_power_fraction(clean_unit, artifact, fraction)
    elif family == "lead_off":
        gate = _lead_off_gate(clean_arr.shape[0], severity, rng)
        mixed = _mix_at_power_fraction(gate * clean_unit, artifact, fraction)
    elif family == "weak_leak":
        env = _weak_leak_gain_envelope(clean_arr.shape[0], severity, rng)
        mixed = _mix_at_power_fraction(env * clean_unit, artifact, fraction)
    elif family == "pure_artifact":
        mixed = artifact
    else:
        raise ValueError(f"Unknown contact artifact family: {family}")

    return normalize_window(mixed) if normalize else mixed.astype(np.float32)


def simulate_contact_artifact_batch(
    clean: np.ndarray,
    *,
    family: str,
    severity: float,
    sample_rate: float,
    seed: int,
    noise_bank: np.ndarray | None = None,
    normalize: bool = True,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty_like(clean, dtype=np.float32)
    for index, window in enumerate(np.asarray(clean, dtype=np.float32)):
        out[index] = simulate_contact_artifact(
            window,
            family=family,
            severity=severity,
            sample_rate=sample_rate,
            rng=rng,
            noise_bank=noise_bank,
            normalize=normalize,
        )
    return out
