"""Loss function builders for physiological signal compression."""

from compressionkit.losses.derivative import build_derivative_loss
from compressionkit.losses.dwt import build_dwt_loss
from compressionkit.losses.filtered_mse import build_filtered_mse_loss
from compressionkit.losses.spectral import build_multi_scale_spectral_loss

__all__ = [
    "build_derivative_loss",
    "build_dwt_loss",
    "build_filtered_mse_loss",
    "build_multi_scale_spectral_loss",
]
