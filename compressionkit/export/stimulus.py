"""Generate synthetic stimulus data for deployment packages.

Creates license-safe synthetic physiological signal samples using physiokit,
suitable for inclusion in HuggingFace releases and deployment packages
without redistributing restricted clinical data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def generate_stimulus(
    *,
    modality: str,
    num_samples: int = 10,
    frame_size: int = 256,
    sample_rate: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic stimulus signals for a given modality.

    Args:
        modality: Signal type — ``"ecg"`` or ``"ppg"``.
        num_samples: Number of stimulus windows to generate.
        frame_size: Samples per window.
        sample_rate: Sampling rate in Hz.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape ``(num_samples, frame_size)`` with float32 dtype.
    """
    if modality.lower() == "ecg":
        from compressionkit.preprocessing.ecg import generate_synthetic_ecg_batch

        return generate_synthetic_ecg_batch(
            num_segments=num_samples,
            signal_length=frame_size,
            sample_rate=sample_rate,
            seed=seed,
        )
    elif modality.lower() == "ppg":
        from compressionkit.preprocessing.ppg import generate_synthetic_ppg_batch

        return generate_synthetic_ppg_batch(
            num_segments=num_samples,
            signal_length=frame_size,
            sample_rate=sample_rate,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported modality: {modality!r}. Use 'ecg' or 'ppg'.")


def export_stimulus_npz(
    *,
    modality: str,
    output_path: str | Path,
    num_samples: int = 10,
    frame_size: int = 256,
    sample_rate: int = 256,
    seed: int = 42,
) -> Path:
    """Generate and save synthetic stimulus to a .npz file.

    Args:
        modality: Signal type — ``"ecg"`` or ``"ppg"``.
        output_path: Path for the output ``.npz`` file.
        num_samples: Number of stimulus windows.
        frame_size: Samples per window.
        sample_rate: Sampling rate in Hz.
        seed: Random seed.

    Returns:
        Path to the written ``.npz`` file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stimulus = generate_stimulus(
        modality=modality,
        num_samples=num_samples,
        frame_size=frame_size,
        sample_rate=sample_rate,
        seed=seed,
    )
    np.savez(
        output_path,
        stimulus=stimulus,
        modality=np.array(modality),
        sample_rate=np.array(sample_rate),
    )
    logger.info("Exported %d synthetic %s stimulus frames to %s", num_samples, modality, output_path)
    return output_path
