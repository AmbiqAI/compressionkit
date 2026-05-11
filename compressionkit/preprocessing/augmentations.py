"""Tier-1 PPG augmentations for improving cross-domain generalization.

These augmentations operate on raw PPG windows and simulate real-world
degradation without requiring generative models. Designed for use in
tf.data pipelines during training.

Augmentations:
    - Baseline wander injection (low-freq sinusoids)
    - Motion artifact injection from accelerometer-correlated noise
    - Per-beat amplitude scaling (perfusion changes)
    - Time warping within physiological bounds
    - Realistic noise injection from empirical noise segments
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as scipy_signal


# ---------------------------------------------------------------------------
# Baseline wander (low-frequency drift)
# ---------------------------------------------------------------------------


def add_baseline_wander(
    ppg: np.ndarray,
    *,
    sample_rate: int = 64,
    freq_range: tuple[float, float] = (0.05, 0.5),
    amplitude_range: tuple[float, float] = (0.05, 0.3),
    num_components: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add realistic baseline wander using sum of sinusoids.

    Models respiration-coupled drift and slow sensor coupling changes.

    Args:
        ppg: Input signal of shape ``(T,)`` or ``(B, T)``.
        sample_rate: Sampling rate in Hz.
        freq_range: Frequency range for wander components in Hz.
        amplitude_range: Amplitude range as fraction of signal std.
        num_components: Number of sinusoidal components to sum.
        rng: Random number generator.

    Returns:
        Signal with baseline wander added, same shape as input.
    """
    if rng is None:
        rng = np.random.default_rng()

    single = ppg.ndim == 1
    if single:
        ppg = ppg[np.newaxis, :]

    batch_size, length = ppg.shape
    t = np.arange(length, dtype=np.float32) / sample_rate
    result = ppg.copy()

    for i in range(batch_size):
        sig_std = ppg[i].std() + 1e-8
        wander = np.zeros(length, dtype=np.float32)
        for _ in range(num_components):
            freq = rng.uniform(freq_range[0], freq_range[1])
            amp = rng.uniform(amplitude_range[0], amplitude_range[1]) * sig_std
            phase = rng.uniform(0, 2 * np.pi)
            wander += amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
        result[i] += wander

    return result[0] if single else result


# ---------------------------------------------------------------------------
# Motion artifact injection (accelerometer-correlated)
# ---------------------------------------------------------------------------


def add_motion_artifact(
    ppg: np.ndarray,
    acc: np.ndarray | None = None,
    *,
    sample_rate: int = 64,
    snr_range: tuple[float, float] = (5.0, 20.0),
    freq_range: tuple[float, float] = (0.5, 8.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add motion artifact noise to PPG signal.

    If accelerometer data is provided, uses it to generate correlated motion
    noise. Otherwise generates synthetic motion-like noise in the PPG band.

    Args:
        ppg: Input signal of shape ``(T,)`` or ``(B, T)``.
        acc: Optional accelerometer magnitude of shape ``(T,)`` or ``(B, T)``.
            If provided, resampled to match ppg length and used as noise envelope.
        sample_rate: Sampling rate in Hz.
        snr_range: Target SNR range in dB for motion corruption.
        freq_range: Frequency range for synthetic motion noise.
        rng: Random number generator.

    Returns:
        PPG signal with motion artifacts added.
    """
    if rng is None:
        rng = np.random.default_rng()

    single = ppg.ndim == 1
    if single:
        ppg = ppg[np.newaxis, :]
        if acc is not None and acc.ndim == 1:
            acc = acc[np.newaxis, :]

    batch_size, length = ppg.shape
    result = ppg.copy()

    for i in range(batch_size):
        sig_power = np.mean(ppg[i] ** 2) + 1e-10
        target_snr_db = rng.uniform(snr_range[0], snr_range[1])
        target_noise_power = sig_power / (10 ** (target_snr_db / 10))

        if acc is not None and i < acc.shape[0]:
            # Use accelerometer as motion envelope
            acc_sig = acc[i]
            if len(acc_sig) != length:
                # Resample acc to match PPG length
                acc_sig = np.interp(
                    np.linspace(0, 1, length),
                    np.linspace(0, 1, len(acc_sig)),
                    acc_sig,
                ).astype(np.float32)
            # Normalize acc and modulate band-limited noise
            acc_envelope = np.abs(acc_sig - acc_sig.mean()) / (acc_sig.std() + 1e-8)
            noise = rng.standard_normal(length).astype(np.float32)
            # Bandpass filter noise to PPG-relevant frequencies
            nyq = sample_rate / 2.0
            low = max(freq_range[0] / nyq, 0.01)
            high = min(freq_range[1] / nyq, 0.99)
            sos = scipy_signal.butter(3, [low, high], btype="bandpass", output="sos")
            noise = scipy_signal.sosfilt(sos, noise).astype(np.float32)
            # Apply motion envelope
            noise = noise * (1.0 + 2.0 * acc_envelope)
        else:
            # Synthetic motion noise: bandpass filtered random signal
            noise = rng.standard_normal(length).astype(np.float32)
            nyq = sample_rate / 2.0
            low = max(freq_range[0] / nyq, 0.01)
            high = min(freq_range[1] / nyq, 0.99)
            sos = scipy_signal.butter(3, [low, high], btype="bandpass", output="sos")
            noise = scipy_signal.sosfilt(sos, noise).astype(np.float32)

        # Scale noise to target power
        current_noise_power = np.mean(noise ** 2) + 1e-10
        noise = noise * np.sqrt(target_noise_power / current_noise_power)
        result[i] += noise

    return result[0] if single else result


# ---------------------------------------------------------------------------
# Per-beat amplitude scaling (perfusion variation)
# ---------------------------------------------------------------------------


def scale_beat_amplitudes(
    ppg: np.ndarray,
    *,
    sample_rate: int = 64,
    scale_range: tuple[float, float] = (0.7, 1.3),
    transition_beats: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly scale individual beat amplitudes to simulate perfusion changes.

    Uses zero-crossings to approximate beat boundaries, then applies smooth
    per-beat amplitude modulation.

    Args:
        ppg: Input signal of shape ``(T,)``.
        sample_rate: Sampling rate in Hz.
        scale_range: Range of per-beat scale factors.
        transition_beats: Smoothing window in beats for scale transitions.
        rng: Random number generator.

    Returns:
        Signal with per-beat amplitude variation.
    """
    if rng is None:
        rng = np.random.default_rng()

    length = len(ppg)
    # Find approximate beat boundaries using peaks
    min_distance = int(0.4 * sample_rate)  # minimum 0.4s between beats (150 bpm max)
    peaks, _ = scipy_signal.find_peaks(ppg, distance=min_distance)

    if len(peaks) < 3:
        # Not enough beats detected, apply global scale
        return ppg * rng.uniform(scale_range[0], scale_range[1])

    # Generate per-beat scale factors
    n_beats = len(peaks)
    scales = rng.uniform(scale_range[0], scale_range[1], size=n_beats).astype(np.float32)

    # Smooth scales for natural transitions
    if transition_beats > 1:
        kernel = np.ones(transition_beats) / transition_beats
        scales = np.convolve(scales, kernel, mode="same").astype(np.float32)

    # Interpolate scale factors to sample level
    sample_scales = np.interp(
        np.arange(length), peaks, scales,
    ).astype(np.float32)

    # Apply around the mean to preserve DC offset
    mean = ppg.mean()
    return ((ppg - mean) * sample_scales + mean).astype(np.float32)


# ---------------------------------------------------------------------------
# Time warping (HR variability simulation)
# ---------------------------------------------------------------------------


def time_warp(
    ppg: np.ndarray,
    *,
    sample_rate: int = 64,
    max_warp_fraction: float = 0.15,
    num_knots: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply smooth time warping to simulate heart rate variability.

    Creates a smooth monotonic warping function that locally stretches/compresses
    the signal, simulating natural HR variability during the window.

    Args:
        ppg: Input signal of shape ``(T,)``.
        sample_rate: Sampling rate in Hz.
        max_warp_fraction: Maximum local time stretch/compression fraction.
        num_knots: Number of control points for the warping function.
        rng: Random number generator.

    Returns:
        Time-warped signal of same length.
    """
    if rng is None:
        rng = np.random.default_rng()

    length = len(ppg)
    # Create warping function using cubic interpolation of random knots
    knot_positions = np.linspace(0, length - 1, num_knots + 2)
    # Random displacements at interior knots
    displacements = np.zeros(num_knots + 2, dtype=np.float32)
    displacements[1:-1] = rng.uniform(
        -max_warp_fraction * length / num_knots,
        max_warp_fraction * length / num_knots,
        size=num_knots,
    ).astype(np.float32)

    # Warped positions (ensure monotonicity)
    warped_knots = knot_positions + displacements
    # Force monotonic by sorting
    warped_knots = np.sort(warped_knots)
    # Clamp to valid range
    warped_knots[0] = 0
    warped_knots[-1] = length - 1

    # Create mapping from output positions to input positions
    output_positions = np.arange(length, dtype=np.float32)
    input_positions = np.interp(output_positions, knot_positions, warped_knots)
    input_positions = np.clip(input_positions, 0, length - 1)

    # Resample signal at warped positions (read from warped input locations)
    return np.interp(input_positions, np.arange(length), ppg).astype(np.float32)


# ---------------------------------------------------------------------------
# Empirical noise injection from real data
# ---------------------------------------------------------------------------


def extract_noise_segments(
    ppg: np.ndarray,
    *,
    sample_rate: int = 64,
    window_size: int = 320,
    noise_threshold_std: float = 2.0,
) -> np.ndarray:
    """Extract high-noise segments from a PPG recording.

    Identifies windows with high residual energy after bandpass filtering,
    which correspond to noise-corrupted segments. These can be used as
    realistic noise templates for augmentation.

    Args:
        ppg: Full recording signal of shape ``(T,)``.
        sample_rate: Sampling rate in Hz.
        window_size: Window size for segment extraction.
        noise_threshold_std: Threshold in std above median noise for selection.

    Returns:
        Array of shape ``(N, window_size)`` containing noisy segments.
    """
    # Bandpass to get clean PPG estimate
    nyq = sample_rate / 2.0
    sos = scipy_signal.butter(3, [0.5 / nyq, 8.0 / nyq], btype="bandpass", output="sos")
    clean_estimate = scipy_signal.sosfilt(sos, ppg).astype(np.float32)

    # Residual = original - clean estimate (contains noise + high-freq artifacts)
    residual = ppg - clean_estimate

    # Window the residual and compute per-window energy
    n_windows = len(ppg) // window_size
    if n_windows == 0:
        return np.empty((0, window_size), dtype=np.float32)

    windows = residual[: n_windows * window_size].reshape(n_windows, window_size)
    energies = np.mean(windows ** 2, axis=1)

    # Select high-noise windows
    median_energy = np.median(energies)
    std_energy = energies.std()
    threshold = median_energy + noise_threshold_std * std_energy
    noisy_mask = energies > threshold

    return windows[noisy_mask].astype(np.float32)


def add_empirical_noise(
    ppg: np.ndarray,
    noise_bank: np.ndarray,
    *,
    snr_range: tuple[float, float] = (10.0, 25.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add noise from an empirical noise bank to clean PPG.

    Args:
        ppg: Clean PPG signal of shape ``(T,)`` or ``(B, T)``.
        noise_bank: Array of noise segments ``(N, W)`` from extract_noise_segments.
        snr_range: Target SNR range in dB.
        rng: Random number generator.

    Returns:
        PPG with empirical noise added.
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(noise_bank) == 0:
        return ppg

    single = ppg.ndim == 1
    if single:
        ppg = ppg[np.newaxis, :]

    batch_size, length = ppg.shape
    noise_len = noise_bank.shape[1]
    result = ppg.copy()

    for i in range(batch_size):
        # Pick random noise segment and tile/crop to match length
        idx = rng.integers(0, len(noise_bank))
        noise = noise_bank[idx]
        if noise_len < length:
            reps = int(np.ceil(length / noise_len))
            noise = np.tile(noise, reps)[:length]
        elif noise_len > length:
            start = rng.integers(0, noise_len - length)
            noise = noise[start : start + length]

        # Scale to target SNR
        sig_power = np.mean(ppg[i] ** 2) + 1e-10
        target_snr_db = rng.uniform(snr_range[0], snr_range[1])
        target_noise_power = sig_power / (10 ** (target_snr_db / 10))
        current_noise_power = np.mean(noise ** 2) + 1e-10
        noise = noise * np.sqrt(target_noise_power / current_noise_power)

        result[i] += noise.astype(np.float32)

    return result[0] if single else result


# ---------------------------------------------------------------------------
# Combined augmentation pipeline
# ---------------------------------------------------------------------------


class PPGAugmenter:
    """Configurable PPG augmentation pipeline for training.

    Applies a random subset of augmentations to each window during training.
    Designed for use with tf.data via tf.numpy_function or offline augmentation.

    Args:
        sample_rate: Signal sampling rate in Hz.
        baseline_wander_prob: Probability of applying baseline wander.
        motion_artifact_prob: Probability of adding motion artifacts.
        beat_scale_prob: Probability of per-beat amplitude scaling.
        time_warp_prob: Probability of time warping.
        empirical_noise_prob: Probability of adding empirical noise.
        noise_bank: Optional precomputed noise bank for empirical noise.
        seed: Random seed.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 64,
        baseline_wander_prob: float = 0.3,
        motion_artifact_prob: float = 0.3,
        motion_snr_range: tuple[float, float] = (12.0, 25.0),
        beat_scale_prob: float = 0.2,
        time_warp_prob: float = 0.2,
        empirical_noise_prob: float = 0.3,
        empirical_snr_range: tuple[float, float] = (10.0, 25.0),
        noise_bank: np.ndarray | None = None,
        seed: int = 42,
    ):
        self.sample_rate = sample_rate
        self.baseline_wander_prob = baseline_wander_prob
        self.motion_artifact_prob = motion_artifact_prob
        self.motion_snr_range = motion_snr_range
        self.beat_scale_prob = beat_scale_prob
        self.time_warp_prob = time_warp_prob
        self.empirical_noise_prob = empirical_noise_prob
        self.empirical_snr_range = empirical_snr_range
        self.noise_bank = noise_bank
        self.rng = np.random.default_rng(seed)

    def augment(self, ppg: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a single window.

        Args:
            ppg: Input of shape ``(T,)``.

        Returns:
            Augmented signal of shape ``(T,)``.
        """
        result = ppg.copy().astype(np.float32)

        if self.rng.random() < self.baseline_wander_prob:
            result = add_baseline_wander(
                result, sample_rate=self.sample_rate, rng=self.rng,
            )

        if self.rng.random() < self.motion_artifact_prob:
            result = add_motion_artifact(
                result, sample_rate=self.sample_rate,
                snr_range=self.motion_snr_range, rng=self.rng,
            )

        if self.rng.random() < self.beat_scale_prob:
            result = scale_beat_amplitudes(
                result, sample_rate=self.sample_rate, rng=self.rng,
            )

        if self.rng.random() < self.time_warp_prob:
            result = time_warp(
                result, sample_rate=self.sample_rate, rng=self.rng,
            )

        if self.noise_bank is not None and self.rng.random() < self.empirical_noise_prob:
            result = add_empirical_noise(
                result, self.noise_bank,
                snr_range=self.empirical_snr_range, rng=self.rng,
            )

        return result

    def augment_batch(self, batch: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a batch of windows.

        Args:
            batch: Input of shape ``(B, T)``.

        Returns:
            Augmented batch of shape ``(B, T)``.
        """
        return np.stack([self.augment(batch[i]) for i in range(len(batch))])


# ---------------------------------------------------------------------------
# Noise bank builder (from DaLiA/WESAD recordings)
# ---------------------------------------------------------------------------


def build_noise_bank_from_h5(
    h5_paths: list[str],
    *,
    target_fs: int = 64,
    window_size: int = 320,
    max_segments: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """Build a noise bank from multiple h5 PPG recordings.

    Extracts noisy segments from wrist-PPG recordings to use as realistic
    noise templates during training augmentation.

    Args:
        h5_paths: Paths to h5 files with ``data`` key containing PPG.
        target_fs: Target sampling rate.
        window_size: Window size for noise segments.
        max_segments: Maximum total noise segments to collect.
        seed: Random seed.

    Returns:
        Noise bank array of shape ``(N, window_size)``.
    """
    import h5py
    from scipy.signal import resample_poly

    rng = np.random.default_rng(seed)
    all_noise: list[np.ndarray] = []

    for path in h5_paths:
        try:
            with h5py.File(path, "r") as hf:
                signal = hf["data"][0].astype(np.float32)
                fs = int(hf.attrs.get("fs", target_fs))
        except (OSError, KeyError):
            continue

        # Resample if needed
        if fs != target_fs:
            gcd = np.gcd(fs, target_fs)
            signal = resample_poly(signal, target_fs // gcd, fs // gcd).astype(np.float32)

        noise_segs = extract_noise_segments(
            signal, sample_rate=target_fs, window_size=window_size,
        )
        if len(noise_segs) > 0:
            all_noise.append(noise_segs)

    if not all_noise:
        return np.empty((0, window_size), dtype=np.float32)

    combined = np.concatenate(all_noise, axis=0)
    if len(combined) > max_segments:
        idx = rng.choice(len(combined), max_segments, replace=False)
        combined = combined[idx]

    return combined


__all__ = [
    "PPGAugmenter",
    "add_baseline_wander",
    "add_empirical_noise",
    "add_motion_artifact",
    "build_noise_bank_from_h5",
    "extract_noise_segments",
    "scale_beat_amplitudes",
    "time_warp",
]
