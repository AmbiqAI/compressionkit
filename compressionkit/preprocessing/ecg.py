"""ECG-specific preprocessing, augmentation, and synthetic data generation.

Uses helia_edge preprocessing layers for the core augmentation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import helia_edge as helia
import keras
import numpy as np
import physiokit as pk
import tensorflow as tf

from compressionkit.configs.ecg_rvq import AugmentationConfig


def build_preprocessor(frame_size: int, epsilon: float = 1e-3) -> keras.layers.Layer:
    """Create preprocessing pipeline: random crop + layer normalization.

    Args:
        frame_size: Number of samples per frame after cropping.
        epsilon: LayerNorm epsilon for numerical stability.
    """
    return helia.layers.preprocessing.AugmentationPipeline(
        layers=[
            helia.layers.preprocessing.RandomCrop1D(duration=frame_size, name="RandomCrop"),
            helia.layers.preprocessing.LayerNormalization1D(epsilon=epsilon, name="LayerNorm"),
        ]
    )


def build_augmenter(
    aug_cfg: AugmentationConfig | None = None,
    *,
    noise_factor: tuple[float, float] | None = None,
    sample_rate: int = 500,
    noise_bank: np.ndarray | None = None,
    artifact_bank: np.ndarray | None = None,
    seed: int = 42,
) -> keras.layers.Layer:
    """Create augmentation pipeline from config.

    Args:
        aug_cfg: Full augmentation config. Takes precedence over noise_factor.
        noise_factor: Legacy fallback ``(min_std, max_std)`` for Gaussian noise.
        noise_bank: Optional empirical residual bank for ``RandomEmpiricalNoise1D``.
        artifact_bank: Optional precomputed contact-artifact bank for
            ``RandomArtifactNoise1D`` (wide, continuous-severity augmentation).
    """
    layers: list[keras.layers.Layer] = []

    if aug_cfg is not None:
        nf = tuple(aug_cfg.gaussian_noise)
        if nf[1] > 0:
            layers.append(
                helia.layers.preprocessing.RandomGaussianNoise1D(
                    factor=nf,
                    name="GaussianNoise",
                )
            )
        empirical_prob = float(aug_cfg.empirical_noise_prob)
        if noise_bank is not None and len(noise_bank) > 0 and empirical_prob > 0.0:
            layers.append(
                RandomEmpiricalNoise1D(
                    noise_bank=noise_bank,
                    prob=empirical_prob,
                    snr_range=tuple(float(v) for v in aug_cfg.empirical_snr_range),
                    seed=seed,
                    name="EmpiricalNoise",
                )
            )
        if getattr(aug_cfg, "artifact_noise_enabled", False) and artifact_bank is not None and len(artifact_bank) > 0:
            layers.append(
                RandomArtifactNoise1D(
                    artifact_bank=artifact_bank,
                    severity_beta=tuple(float(v) for v in aug_cfg.artifact_severity_beta),
                    clean_prob=float(aug_cfg.artifact_clean_prob),
                    clean_severity_max=float(aug_cfg.artifact_clean_severity_max),
                    seed=seed,
                    name="ArtifactNoise",
                )
            )
        if aug_cfg.amplitude_warp:
            layers.append(
                helia.layers.preprocessing.AmplitudeWarp(
                    sample_rate=sample_rate,
                    amplitude=tuple(aug_cfg.amplitude_warp_amplitude),
                    frequency=tuple(aug_cfg.amplitude_warp_frequency),
                    name="AmplitudeWarp",
                )
            )
        # NOTE: random_cutout is handled at the dataset level via paired cutout
        # (_paired_cutout_batch) so that both input and target are zeroed.
    elif noise_factor is not None:
        layers.append(
            helia.layers.preprocessing.RandomGaussianNoise1D(
                factor=noise_factor,
                name="GaussianNoise",
            )
        )
    else:
        layers.append(
            helia.layers.preprocessing.RandomGaussianNoise1D(
                factor=(0.01, 0.1),
                name="GaussianNoise",
            )
        )

    return helia.layers.preprocessing.AugmentationPipeline(layers=layers)


def extract_noise_segments(
    ecg: np.ndarray,
    *,
    sample_rate: int,
    window_size: int,
    noise_threshold_std: float = 2.0,
) -> np.ndarray:
    """Extract high-residual ECG windows suitable for empirical noise augmentation."""
    if ecg.ndim == 2:
        segments = [
            extract_noise_segments(
                ecg[:, channel],
                sample_rate=sample_rate,
                window_size=window_size,
                noise_threshold_std=noise_threshold_std,
            )
            for channel in range(ecg.shape[1])
        ]
        segments = [segment for segment in segments if len(segment) > 0]
        if not segments:
            return np.empty((0, window_size), dtype=np.float32)
        return np.concatenate(segments, axis=0).astype(np.float32)

    if len(ecg) < window_size:
        return np.empty((0, window_size), dtype=np.float32)

    from scipy import signal as scipy_signal

    nyquist = sample_rate / 2.0
    high_hz = min(40.0, nyquist * 0.95)
    if high_hz <= 0.5:
        return np.empty((0, window_size), dtype=np.float32)

    sos = scipy_signal.butter(3, [0.5 / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    clean_estimate = scipy_signal.sosfiltfilt(sos, ecg).astype(np.float32)
    residual = ecg.astype(np.float32) - clean_estimate

    n_windows = len(residual) // window_size
    if n_windows == 0:
        return np.empty((0, window_size), dtype=np.float32)

    windows = residual[: n_windows * window_size].reshape(n_windows, window_size)
    energies = np.mean(windows**2, axis=1)
    median_energy = np.median(energies)
    std_energy = energies.std()
    threshold = median_energy + noise_threshold_std * std_energy
    mask = energies > threshold
    return windows[mask].astype(np.float32)


def build_noise_bank_from_h5(
    file_paths: list[Path],
    *,
    source_sample_rate: int,
    target_sample_rate: int,
    window_size: int,
    lead_index: int = 1,
    leads: list[int] | None = None,
    max_segments: int = 20_000,
    noise_threshold_std: float = 2.0,
) -> np.ndarray | None:
    """Build an ECG empirical noise bank from training H5 files."""
    if not file_paths:
        return None

    from compressionkit.datasets.ecg import _resample, load_ecg_signal

    segments: list[np.ndarray] = []
    for path in file_paths:
        try:
            signal = load_ecg_signal(path, lead_index=lead_index, leads=leads)
        except Exception:
            continue

        if signal.ndim == 1:
            if source_sample_rate != target_sample_rate:
                signal = _resample(signal, source_sample_rate, target_sample_rate)
        else:
            if source_sample_rate != target_sample_rate:
                signal = np.stack(
                    [
                        _resample(signal[:, channel], source_sample_rate, target_sample_rate)
                        for channel in range(signal.shape[1])
                    ],
                    axis=1,
                ).astype(np.float32)

        extracted = extract_noise_segments(
            signal,
            sample_rate=target_sample_rate,
            window_size=window_size,
            noise_threshold_std=noise_threshold_std,
        )
        if len(extracted) > 0:
            segments.append(extracted)

        if sum(len(segment) for segment in segments) >= max_segments:
            break

    if not segments:
        return None

    noise_bank = np.concatenate(segments, axis=0).astype(np.float32)
    if len(noise_bank) > max_segments:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(noise_bank), size=max_segments, replace=False)
        noise_bank = noise_bank[indices]
    return noise_bank


class RandomEmpiricalNoise1D(keras.layers.Layer):
    """Add residual ECG noise sampled from an empirical noise bank."""

    def __init__(
        self,
        *,
        noise_bank: np.ndarray,
        prob: float,
        snr_range: tuple[float, float],
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.noise_bank = np.asarray(noise_bank, dtype=np.float32)
        self.prob = float(prob)
        self.snr_range = (float(snr_range[0]), float(snr_range[1]))
        self.rng = np.random.default_rng(seed)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if training is False or len(self.noise_bank) == 0 or self.prob <= 0.0:
            return inputs

        def _augment_batch(batch: np.ndarray) -> np.ndarray:
            augmented = np.array(batch, copy=True, dtype=np.float32)
            if augmented.ndim != 3:
                return augmented

            batch_size, signal_len, num_channels = augmented.shape
            for batch_idx in range(batch_size):
                if self.rng.random() >= self.prob:
                    continue
                for channel_idx in range(num_channels):
                    noise = self.noise_bank[self.rng.integers(0, len(self.noise_bank))]
                    if len(noise) < signal_len:
                        reps = int(np.ceil(signal_len / len(noise)))
                        noise = np.tile(noise, reps)[:signal_len]
                    elif len(noise) > signal_len:
                        start = int(self.rng.integers(0, len(noise) - signal_len + 1))
                        noise = noise[start : start + signal_len]

                    signal = augmented[batch_idx, :, channel_idx]
                    signal_power = np.mean(signal**2) + 1e-10
                    target_snr_db = self.rng.uniform(self.snr_range[0], self.snr_range[1])
                    target_noise_power = signal_power / (10 ** (target_snr_db / 10))
                    current_noise_power = np.mean(noise**2) + 1e-10
                    scaled_noise = noise * np.sqrt(target_noise_power / current_noise_power)
                    augmented[batch_idx, :, channel_idx] = signal + scaled_noise.astype(np.float32)
            return augmented

        outputs = tf.numpy_function(_augment_batch, [inputs], tf.float32)
        outputs.set_shape(inputs.shape)
        return outputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"prob": self.prob, "snr_range": list(self.snr_range)})
        return config


class RandomArtifactNoise1D(keras.layers.Layer):
    """Mix contact-artifact waveforms from a precomputed bank at continuous severity.

    Unlike :class:`RandomEmpiricalNoise1D` (on/off ``prob`` + narrow SNR band),
    this layer applies a corruption to *every* window with a **continuous**
    artifact power fraction ``f`` drawn from a Beta distribution, plus a small
    continuous near-clean tail — so the training distribution is wide and not
    bimodal. The corruption uses the same power-fraction model as
    :func:`compressionkit.synthetic.ecg_contact_artifacts.simulate_contact_artifact`::

        mixed = sqrt(1 - f) * signal + sqrt(f) * artifact_scaled_to_signal_power

    The expensive family synthesis is amortized into ``artifact_bank`` offline;
    per-step cost is a gather + scale + add inside ``tf.numpy_function``, matching
    the existing empirical-noise augmentation pattern.
    """

    def __init__(
        self,
        *,
        artifact_bank: np.ndarray,
        severity_beta: tuple[float, float] = (0.9, 1.3),
        clean_prob: float = 0.08,
        clean_severity_max: float = 0.05,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.artifact_bank = np.asarray(artifact_bank, dtype=np.float32)
        self.severity_beta = (float(severity_beta[0]), float(severity_beta[1]))
        self.clean_prob = float(clean_prob)
        self.clean_severity_max = float(clean_severity_max)
        self.rng = np.random.default_rng(seed)

    def _sample_fraction(self) -> float:
        if self.rng.random() < self.clean_prob:
            return float(self.rng.uniform(0.0, self.clean_severity_max))
        a, b = self.severity_beta
        return float(self.rng.beta(a, b))

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if training is False or len(self.artifact_bank) == 0:
            return inputs

        def _augment_batch(batch: np.ndarray) -> np.ndarray:
            augmented = np.array(batch, copy=True, dtype=np.float32)
            if augmented.ndim != 3:
                return augmented
            batch_size, signal_len, num_channels = augmented.shape
            bank_len = self.artifact_bank.shape[1]
            for batch_idx in range(batch_size):
                for channel_idx in range(num_channels):
                    artifact = self.artifact_bank[self.rng.integers(0, len(self.artifact_bank))]
                    if bank_len < signal_len:
                        reps = int(np.ceil(signal_len / bank_len))
                        artifact = np.tile(artifact, reps)[:signal_len]
                    elif bank_len > signal_len:
                        start = int(self.rng.integers(0, bank_len - signal_len + 1))
                        artifact = artifact[start : start + signal_len]

                    signal = augmented[batch_idx, :, channel_idx]
                    sig_power = float(np.mean(signal**2)) + 1e-10
                    art_power = float(np.mean(artifact**2)) + 1e-10
                    art = artifact * np.sqrt(sig_power / art_power)
                    f = self._sample_fraction()
                    mixed = np.sqrt(1.0 - f) * signal + np.sqrt(f) * art
                    augmented[batch_idx, :, channel_idx] = mixed.astype(np.float32)
            return augmented

        outputs = tf.numpy_function(_augment_batch, [inputs], tf.float32)
        outputs.set_shape(inputs.shape)
        return outputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "severity_beta": list(self.severity_beta),
                "clean_prob": self.clean_prob,
                "clean_severity_max": self.clean_severity_max,
            }
        )
        return config


# ECG presets suitable for general-purpose training augmentation.
_ECG_PRESETS: list[pk.ecg.EcgPreset] = [
    pk.ecg.EcgPreset.SR,
    pk.ecg.EcgPreset.SR,  # double-weight normal sinus rhythm
    pk.ecg.EcgPreset.AFIB,
    pk.ecg.EcgPreset.LBBB,
    pk.ecg.EcgPreset.LAHB,
    pk.ecg.EcgPreset.LPHB,
    pk.ecg.EcgPreset.ant_STEMI,
    pk.ecg.EcgPreset.high_take_off,
    pk.ecg.EcgPreset.random_morphology,
]


def _sample_uniform_range(
    rng: np.random.Generator,
    value: Any,
    *,
    default_low: float,
    default_high: float,
) -> float:
    """Sample a scalar from either a fixed value or ``[low, high]`` range."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
        if high < low:
            low, high = high, low
        return float(rng.uniform(low, high))
    return float(rng.uniform(default_low, default_high))


def generate_synthetic_ecg_batch(
    *,
    num_segments: int,
    signal_length: int,
    sample_rate: int,
    lead_index: int = 1,
    heart_rate_bpm: list[float] | None = None,
    noise_multiplier: list[float] | None = None,
    impedance: list[float] | None = None,
    seed: int = 1337,
) -> np.ndarray:
    """Generate synthetic ECG segments via physiokit.

    Randomly cycles through rhythm presets (SR, AFIB, LBBB, etc.) to
    create morphological diversity.  For efficiency, generates longer base
    signals and extracts multiple non-overlapping windows from each.

    Args:
        num_segments: Number of synthetic segments to generate.
        signal_length: Samples per segment (at *sample_rate*).
        sample_rate: Sampling rate in Hz.
        lead_index: Which lead to extract (0-based, default 1 = lead II).
        heart_rate_bpm: Range ``[low, high]`` for heart rate in BPM.
        noise_multiplier: Range ``[low, high]`` for noise amplitude scaling.
        impedance: Range ``[low, high]`` for electrode impedance factor.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape ``(num_segments, signal_length)`` with ``float32`` dtype.
    """
    if num_segments <= 0:
        return np.empty((0, signal_length), dtype=np.float32)

    heart_rate_bpm = heart_rate_bpm or [50.0, 100.0]
    noise_multiplier = noise_multiplier or [0.2, 1.0]
    impedance = impedance or [0.5, 1.5]

    # Generate longer signals and slice into windows for efficiency.
    # Each base signal yields ``windows_per_signal`` non-overlapping windows.
    windows_per_signal = max(1, min(64, int(30.0 * sample_rate / signal_length)))
    base_length = windows_per_signal * signal_length
    num_base_signals = int(np.ceil(num_segments / windows_per_signal))

    rng = np.random.default_rng(seed)
    segments: list[np.ndarray] = []
    for i in range(num_base_signals):
        hr = _sample_uniform_range(rng, heart_rate_bpm, default_low=50.0, default_high=100.0)
        nm = _sample_uniform_range(rng, noise_multiplier, default_low=0.2, default_high=1.0)
        imp = _sample_uniform_range(rng, impedance, default_low=0.5, default_high=1.5)
        preset = _ECG_PRESETS[i % len(_ECG_PRESETS)]
        data, _, _ = pk.ecg.synthesize(
            signal_length=int(base_length),
            sample_rate=float(sample_rate),
            leads=12,
            heart_rate=float(hr),
            preset=preset,
            noise_multiplier=float(nm),
            impedance=float(imp),
        )
        lead = np.asarray(data[lead_index], dtype=np.float32)
        for w in range(windows_per_signal):
            if len(segments) >= num_segments:
                break
            start = w * signal_length
            segments.append(lead[start : start + signal_length])
    return np.stack(segments[:num_segments]).astype(np.float32)


__all__ = [
    "build_augmenter",
    "build_noise_bank_from_h5",
    "build_preprocessor",
    "extract_noise_segments",
    "generate_synthetic_ecg_batch",
]
