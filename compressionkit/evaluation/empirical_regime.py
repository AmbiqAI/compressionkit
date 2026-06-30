"""Shared primitives for empirical-noise robustness sweeps and evaluations.

These helpers were previously duplicated across the ``scripts/sweep_*`` and
``scripts/eval_*`` robustness tools (ECG and PPG). They are pure NumPy
primitives with no modality-specific assumptions, so they live in the package
to give the release evaluation tooling a single, tested source of truth.

Modality-specific window builders (``build_real_windows``,
``build_clean_windows``) and golden run-directory maps remain in their
respective sweep modules because they depend on dataset loaders and codec
configuration.
"""

from __future__ import annotations

import math

import numpy as np

# Default input-SNR ladder (dB) used by the empirical-noise regime sweeps.
# ``None`` denotes the pristine / clean reference condition.
DEFAULT_SNR_DB: list[float | None] = [None, 24.0, 18.0, 12.0, 8.0, 6.0, 3.0, 0.0, -3.0, -6.0]


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Z-normalize a signal to zero mean and unit standard deviation."""
    return (x - x.mean()) / (x.std() + 1e-9)


def sample_noise_segment(
    noise_bank: np.ndarray, length: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw a ``length``-sample noise segment from ``noise_bank``.

    Tiles short segments and randomly crops long ones so the returned array
    always has exactly ``length`` samples.
    """
    noise = noise_bank[rng.integers(0, len(noise_bank))]
    if len(noise) < length:
        reps = int(np.ceil(length / len(noise)))
        noise = np.tile(noise, reps)[:length]
    elif len(noise) > length:
        start = int(rng.integers(0, len(noise) - length + 1))
        noise = noise[start : start + length]
    return noise.astype(np.float32)


def add_empirical_noise(
    clean: np.ndarray,
    noise_bank: np.ndarray,
    snr_db: float,
    *,
    seed: int,
    normalize: bool = True,
) -> np.ndarray:
    """Inject empirical noise into ``clean`` windows at a target input SNR.

    Args:
        clean: Batch of clean windows, shape ``(n, length)``.
        noise_bank: Pool of real noise/residual segments to sample from.
        snr_db: Target signal-to-noise ratio in dB.
        seed: Seed for the random generator (segment choice + crop offset).
        normalize: If ``True``, z-normalize each noisy window after mixing.

    Returns:
        A float array shaped like ``clean`` containing the noisy windows.
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(clean)
    for i, c in enumerate(clean):
        noise = sample_noise_segment(noise_bank, c.shape[0], rng)
        signal_power = float(np.mean(c**2)) + 1e-10
        target_noise_power = signal_power / (10 ** (snr_db / 10.0))
        current_noise_power = float(np.mean(noise**2)) + 1e-10
        scaled = noise * math.sqrt(target_noise_power / current_noise_power)
        x = c + scaled.astype(np.float32)
        out[i] = normalize_signal(x) if normalize else x
    return out


def snr_label(snr_db: float | None) -> str:
    """Render an SNR level as a compact label (``"clean"`` or ``"6dB"``)."""
    return "clean" if snr_db is None else f"{snr_db:g}dB"


def autocorr_peak(
    recon: np.ndarray,
    sample_rate: float,
    *,
    lo_bpm: float = 40.0,
    hi_bpm: float = 150.0,
) -> np.ndarray:
    """Per-window peak autocorrelation within a heart-rate lag band.

    Used as a rhythmicity probe: a high peak on a pure-noise input indicates a
    codec is imprinting a spurious periodic structure. ``hi_bpm`` maps to the
    shortest lag, ``lo_bpm`` to the longest (ECG: 40-150 bpm; PPG: 40-180 bpm).
    """
    lo_lag = int(round(sample_rate * 60.0 / hi_bpm))
    hi_lag = int(round(sample_rate * 60.0 / lo_bpm))
    out = np.empty(recon.shape[0], dtype=np.float32)
    for i, r in enumerate(recon):
        r = r - r.mean()
        denom = float(np.dot(r, r)) + 1e-12
        ac = np.correlate(r, r, mode="full")[r.shape[0] - 1 :]
        band = ac[lo_lag : hi_lag + 1] / denom
        out[i] = float(band.max()) if band.size else 0.0
    return out


def prd(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Percentage RMS difference between ``ref`` and ``est`` (per window)."""
    num = np.linalg.norm(ref - est, axis=-1)
    den = np.linalg.norm(ref, axis=-1) + 1e-12
    return 100.0 * num / den


def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between ``a`` and ``b`` (per window)."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1)) + 1e-12
    return num / den


def encode_decode_batch(codec, frames: np.ndarray) -> np.ndarray:
    """Round-trip a batch of frames through ``codec`` and z-normalize outputs.

    Some codecs return ``(T,)`` or ``(T, 1)``; outputs are flattened, clipped to
    the input frame length, and z-normalized to match the input convention.
    """
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        enc = codec.encode(f)
        rec = codec.decode(enc)
        rec = np.asarray(rec, dtype=np.float32).reshape(-1)[: frames.shape[1]]
        rec = (rec - rec.mean()) / (rec.std() + 1e-9)
        out[i] = rec
    return out
