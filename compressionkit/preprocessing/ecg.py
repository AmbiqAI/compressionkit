"""ECG-specific preprocessing, augmentation, and synthetic data generation.

Uses helia_edge preprocessing layers for the core augmentation pipeline.
"""

from __future__ import annotations

from typing import Any

import helia_edge as helia
import keras
import numpy as np
import physiokit as pk

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
) -> keras.layers.Layer:
    """Create augmentation pipeline from config.

    Args:
        aug_cfg: Full augmentation config. Takes precedence over noise_factor.
        noise_factor: Legacy fallback ``(min_std, max_std)`` for Gaussian noise.
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
        if aug_cfg.amplitude_warp:
            layers.append(
                helia.layers.preprocessing.AmplitudeWarp(
                    sample_rate=sample_rate,
                    amplitude=tuple(aug_cfg.amplitude_warp_amplitude),
                    frequency=tuple(aug_cfg.amplitude_warp_frequency),
                    name="AmplitudeWarp",
                )
            )
        if aug_cfg.random_cutout:
            layers.append(
                helia.layers.preprocessing.RandomCutout1D(
                    factor=tuple(aug_cfg.cutout_factor),
                    name="RandomCutout",
                )
            )
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
    "build_preprocessor",
    "generate_synthetic_ecg_batch",
]
