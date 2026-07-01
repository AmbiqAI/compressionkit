"""PPG-specific preprocessing, augmentation, and synthetic data generation.

Uses helia_edge preprocessing layers for the core augmentation pipeline.
"""

from __future__ import annotations

from typing import Any

import helia_edge as helia
import keras
import numpy as np
import physiokit as pk

from compressionkit.configs.ppg_rvq import AugmentationConfig


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
    noise_factor: tuple[float, float] = (0.01, 0.1),
    *,
    aug_cfg: AugmentationConfig | None = None,
) -> keras.layers.Layer:
    """Create augmentation pipeline.

    Always includes Gaussian noise injection. Paired null augmentation is not
    applied here; it is handled at the dataset layer so both input and target
    are zeroed together, which trains abstention rather than inpainting.

    Args:
        noise_factor: Range ``(min_std, max_std)`` for Gaussian noise.
        aug_cfg: Optional PPG augmentation config. The paired null-augmentation
            knobs are consumed by dataset builders, not the Keras augmenter.
    """
    layers: list[keras.layers.Layer] = [
        helia.layers.preprocessing.RandomGaussianNoise1D(factor=tuple(noise_factor), name="GaussianNoise"),
    ]
    # Paired null augmentation is handled at the dataset layer so both input
    # and target are zeroed under the same mask.
    return helia.layers.preprocessing.AugmentationPipeline(layers=layers)


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


def generate_synthetic_ppg_batch(
    *,
    num_segments: int,
    signal_length: int,
    sample_rate: int,
    heart_rate_bpm: list[float] | None = None,
    frequency_modulation: list[float] | None = None,
    ibi_randomness: list[float] | None = None,
    seed: int = 1337,
) -> np.ndarray:
    """Generate synthetic PPG segments via physiokit.

    Args:
        num_segments: Number of synthetic segments to generate.
        signal_length: Samples per segment.
        sample_rate: Sampling rate in Hz.
        heart_rate_bpm: Range ``[low, high]`` for heart rate in BPM.
        frequency_modulation: Range ``[low, high]`` for frequency modulation.
        ibi_randomness: Range ``[low, high]`` for inter-beat-interval randomness.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape ``[num_segments, signal_length]`` with ``float32`` dtype.
    """
    if num_segments <= 0:
        return np.empty((0, signal_length), dtype=np.float32)

    heart_rate_bpm = heart_rate_bpm or [50.0, 120.0]
    frequency_modulation = frequency_modulation or [0.1, 0.5]
    ibi_randomness = ibi_randomness or [0.02, 0.2]

    rng = np.random.default_rng(seed)
    segments: list[np.ndarray] = []
    for _ in range(num_segments):
        hr = _sample_uniform_range(rng, heart_rate_bpm, default_low=50.0, default_high=120.0)
        fm = _sample_uniform_range(rng, frequency_modulation, default_low=0.1, default_high=0.5)
        ir = _sample_uniform_range(rng, ibi_randomness, default_low=0.02, default_high=0.2)
        signal, _, _ = pk.ppg.synthesize(
            signal_length=int(signal_length),
            sample_rate=float(sample_rate),
            heart_rate=float(hr),
            frequency_modulation=float(fm),
            ibi_randomness=float(ir),
        )
        segments.append(np.asarray(signal, dtype=np.float32).reshape(-1)[:signal_length])
    return np.stack(segments).astype(np.float32)


__all__ = [
    "build_augmenter",
    "build_preprocessor",
    "generate_synthetic_ppg_batch",
]
