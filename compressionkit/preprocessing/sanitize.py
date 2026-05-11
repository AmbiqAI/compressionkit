"""Window-level signal sanitization for physiological time-series.

Provides cheap, embedded-portable checks that flag corrupted PPG / ECG windows
before they reach a model. All operations are pure NumPy and assume the input
window is shaped ``(channels, samples)`` float32.

Intended call site is the ``tf.data`` pipeline (wrapped in ``tf.numpy_function``
or applied at the producer side before stacking into TFRecords). Each check is
deliberately stateless and uses only fixed-size buffers so the same logic can
be ported to embedded C without dynamic allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SanitizeConfig:
    """Threshold configuration for :func:`is_clean_window`.

    Defaults are tuned for PPG/ECG windows but apply to any zero-mean, bounded
    physiological signal.

    Attributes:
        min_std: Minimum per-channel std-dev. Below this the window is flat.
        max_saturation_frac: Reject when more than this fraction of samples
            equals the channel min or max (clipping/rail saturation).
        max_abs_z: Outlier z-score (computed via median + MAD) above which a
            sample is considered an outlier.
        max_outlier_frac: Reject when more than this fraction of samples are
            outliers.
        epsilon: Floor used in MAD / std denominators for numerical safety.
    """

    min_std: float = 1e-4
    max_saturation_frac: float = 0.10
    max_abs_z: float = 8.0
    max_outlier_frac: float = 0.02
    epsilon: float = 1e-8


@dataclass(frozen=True)
class SanitizeReport:
    """Why a window was rejected. ``ok`` is True when all checks passed."""

    ok: bool
    reason: str = ""


def _has_nonfinite(window: np.ndarray) -> bool:
    return not np.all(np.isfinite(window))


def _is_flat(window: np.ndarray, min_std: float) -> bool:
    # Per-channel std; reject if any channel is completely flat.
    std = np.std(window, axis=-1)
    return bool(np.any(std < min_std))


def _saturation_frac(window: np.ndarray) -> float:
    # Fraction of samples that hit the per-channel min or max value.
    chan_min = np.min(window, axis=-1, keepdims=True)
    chan_max = np.max(window, axis=-1, keepdims=True)
    at_rail = (window == chan_min) | (window == chan_max)
    return float(np.mean(at_rail))


def _outlier_frac(window: np.ndarray, max_abs_z: float, epsilon: float) -> float:
    # Robust z via median + MAD; constant 1.4826 makes MAD ~ std for Gaussians.
    med = np.median(window, axis=-1, keepdims=True)
    mad = np.median(np.abs(window - med), axis=-1, keepdims=True)
    z = np.abs(window - med) / (1.4826 * mad + epsilon)
    return float(np.mean(z > max_abs_z))


def is_clean_window(
    window: np.ndarray,
    config: SanitizeConfig | None = None,
) -> SanitizeReport:
    """Return a :class:`SanitizeReport` for ``window``.

    Args:
        window: Array shaped ``(channels, samples)``. 1-D inputs are treated as
            single-channel.
        config: Threshold overrides; defaults from :class:`SanitizeConfig`.
    """
    cfg = config or SanitizeConfig()
    if window.ndim == 1:
        window = window[np.newaxis, :]

    if _has_nonfinite(window):
        return SanitizeReport(False, "nonfinite")
    if _is_flat(window, cfg.min_std):
        return SanitizeReport(False, "flat")
    if _saturation_frac(window) > cfg.max_saturation_frac:
        return SanitizeReport(False, "saturated")
    if _outlier_frac(window, cfg.max_abs_z, cfg.epsilon) > cfg.max_outlier_frac:
        return SanitizeReport(False, "outliers")
    return SanitizeReport(True)


def normalize_window(
    window: np.ndarray,
    *,
    epsilon: float = 1e-6,
    clip_z: float | None = 6.0,
) -> np.ndarray:
    """Robust per-channel z-score using median + MAD.

    Args:
        window: ``(channels, samples)`` float array. 1-D treated as single-channel.
        epsilon: MAD floor for numerical safety.
        clip_z: Optional symmetric clip after normalization. ``None`` disables.

    Returns:
        Float32 array in the same shape as the input.
    """
    if window.ndim == 1:
        window = window[np.newaxis, :]
    med = np.median(window, axis=-1, keepdims=True)
    mad = np.median(np.abs(window - med), axis=-1, keepdims=True)
    out = (window - med) / (1.4826 * mad + epsilon)
    if clip_z is not None:
        out = np.clip(out, -clip_z, clip_z)
    return out.astype(np.float32, copy=False)


__all__ = [
    "SanitizeConfig",
    "SanitizeReport",
    "is_clean_window",
    "normalize_window",
]
