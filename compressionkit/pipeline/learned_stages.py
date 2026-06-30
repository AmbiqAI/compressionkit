"""Learned/AI implementations of the pipeline stages.

These wrap trained Keras building blocks behind the same stage protocols as
:mod:`compressionkit.pipeline.dsp_stages`, so an AI component can be dropped
into any slot of the 4-stage pipeline without touching the others.

Phase 2 entry: :class:`LearnedDenoisePreprocessor` — a wavelet-gain learned
denoiser in the *preprocess* slot, the AI counterpart to the DSP
:class:`~compressionkit.pipeline.dsp_stages.BandpassPreprocessor`. It reuses
the existing wavelet-gain denoiser building block, so the rest of the chain
(DWT -> dead-zone -> deflate) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from compressionkit.dsp.wavelet import WaveletCoeffs, dwt_forward, dwt_inverse


@dataclass
class LearnedDenoisePreprocessor:
    """Wavelet-gain learned denoise in the preprocess slot (lossy; inverse is identity).

    Applies a trained ``packed_coeffs -> denoised_coeffs`` callable in the DWT
    domain, mirroring
    :class:`~compressionkit.evaluation.codec.LearnedShrinkSpihtCodec` but exposed
    as a swappable pipeline stage. The denoise is a front-end conditioning step,
    so :meth:`inverse` is the identity (matching the DSP bandpass stage).

    Attributes:
        coeff_denoiser: Callable mapping a packed DWT coefficient vector to a
            denoised vector of the same length (e.g. from
            :func:`compressionkit.models.wavelet_denoiser.as_coeff_denoiser`).
        frame_size: Expected frame length in samples.
        wavelet: Wavelet name used for the analysis/synthesis transform.
        levels: Number of DWT levels.
    """

    coeff_denoiser: Any
    frame_size: int
    wavelet: str = "bior4.4"
    levels: int = 6
    name: str = "learned_denoise"

    def forward(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        arr = np.asarray(frame, dtype=np.float32)
        coeffs = dwt_forward(arr, levels=self.levels, wavelet=self.wavelet)
        sizes = [len(coeffs.approx)] + [len(d) for d in coeffs.details]
        packed = np.concatenate([coeffs.approx, *coeffs.details]).astype(np.float32)
        denoised_packed = np.asarray(self.coeff_denoiser(packed), dtype=np.float32)
        if denoised_packed.shape != packed.shape:
            raise ValueError(
                f"coeff_denoiser must preserve length {packed.shape}, got {denoised_packed.shape}"
            )
        offsets = np.cumsum(sizes)
        approx_d = denoised_packed[: offsets[0]]
        details_d = [denoised_packed[offsets[i - 1] : offsets[i]] for i in range(1, len(sizes))]
        recon = dwt_inverse(WaveletCoeffs(approx=approx_d, details=details_d), wavelet=self.wavelet)
        return np.asarray(recon, dtype=np.float32)[: self.frame_size], {}

    def inverse(self, signal: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        return np.asarray(signal, dtype=np.float32)


def load_wavelet_gain_preprocessor(
    model_path: str | Path,
    *,
    frame_size: int,
    wavelet: str = "bior4.4",
    levels: int = 6,
) -> LearnedDenoisePreprocessor:
    """Load a trained wavelet-gain model as a :class:`LearnedDenoisePreprocessor`.

    Args:
        model_path: Path to the saved ``gain_model.keras``.
        frame_size: Frame length the model was trained for.
        wavelet: Wavelet name (must match training).
        levels: DWT levels (must match training).

    Returns:
        A ready-to-use preprocess stage.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Learned denoiser weights not found at {path}. "
            "Train one via experiments/scripts/train_wavelet_denoiser_ecg.py."
        )

    import keras

    from compressionkit.models.wavelet_denoiser import (
        BandRatioFeature,
        FinestLevelFeature,
        LevelGate,
        NoiseGatedGain,
        SelectChannel,
        SoftThreshold,
        as_coeff_denoiser,
    )

    custom_objects = {
        "BandRatioFeature": BandRatioFeature,
        "FinestLevelFeature": FinestLevelFeature,
        "LevelGate": LevelGate,
        "NoiseGatedGain": NoiseGatedGain,
        "SelectChannel": SelectChannel,
        "SoftThreshold": SoftThreshold,
    }
    model = keras.models.load_model(path, compile=False, custom_objects=custom_objects)
    denoiser = as_coeff_denoiser(model, frame_size=frame_size, wavelet=wavelet, levels=levels)
    return LearnedDenoisePreprocessor(
        coeff_denoiser=denoiser,
        frame_size=frame_size,
        wavelet=wavelet,
        levels=levels,
    )


__all__ = [
    "LearnedDenoisePreprocessor",
    "load_wavelet_gain_preprocessor",
]
