"""Fixed-filter DWT/iDWT Keras layers for 1-D signal compression.

These layers implement multilevel periodic discrete wavelet transforms using
precomputed transform matrices derived from PyWavelets. The layers operate
on tensors of shape ``(B, 1, T, 1)`` (the codebase's standard 2-D temporal
convention) and produce packed coefficient vectors of the same total length T.

Packing order: ``[cA_L | dL | d(L-1) | ... | d1]`` — coarsest to finest,
matching the SPIHT convention used elsewhere in the codebase.

The matrix approach is correct by construction and GPU-friendly (single
matmul). For deployment on edge, a conv-based implementation can replace
these layers later.
"""

from __future__ import annotations

import keras
import numpy as np

try:
    import pywt
except ImportError:
    pywt = None


def _build_dwt_matrix(signal_len: int, wavelet: str, levels: int) -> np.ndarray:
    """Build the full DWT transform matrix W such that coeffs = W @ signal.

    Returns shape (signal_len, signal_len) float32 matrix.
    """
    if pywt is None:
        raise ImportError("pywt is required for wavelet matrix construction")
    W = np.zeros((signal_len, signal_len), dtype=np.float32)
    for i in range(signal_len):
        # Compute DWT of the i-th basis vector
        e_i = np.zeros(signal_len, dtype=np.float32)
        e_i[i] = 1.0
        coeffs = pywt.wavedec(e_i, wavelet, level=levels, mode="periodization")
        packed = np.concatenate(coeffs, dtype=np.float32)
        W[:, i] = packed
    return W


def _build_idwt_matrix(signal_len: int, wavelet: str, levels: int) -> np.ndarray:
    """Build the full iDWT transform matrix W_inv such that signal = W_inv @ coeffs.

    Returns shape (signal_len, signal_len) float32 matrix.
    """
    if pywt is None:
        raise ImportError("pywt is required for wavelet matrix construction")
    # Get subband sizes by running a dummy transform
    dummy = np.zeros(signal_len, dtype=np.float32)
    coeffs = pywt.wavedec(dummy, wavelet, level=levels, mode="periodization")
    sizes = [len(c) for c in coeffs]

    W_inv = np.zeros((signal_len, signal_len), dtype=np.float32)
    for i in range(signal_len):
        # Compute iDWT of the i-th basis vector in coefficient space
        e_i = np.zeros(signal_len, dtype=np.float32)
        e_i[i] = 1.0
        # Unpack into subband structure
        offset = 0
        coeff_list = []
        for s in sizes:
            coeff_list.append(e_i[offset : offset + s])
            offset += s
        signal = pywt.waverec(coeff_list, wavelet, mode="periodization")
        W_inv[:, i] = signal[:signal_len].astype(np.float32)
    return W_inv


@keras.saving.register_keras_serializable(package="compressionkit")
class DWT1D(keras.layers.Layer):
    """Multilevel 1-D DWT via precomputed transform matrix.

    Input:  ``(B, 1, T, 1)``
    Output: ``(B, 1, T, 1)`` — packed wavelet coefficients.
    """

    def __init__(self, wavelet: str = "bior4.4", levels: int = 6, signal_len: int = 512, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.wavelet = wavelet
        self.levels = levels
        self.signal_len = signal_len
        self._matrix = _build_dwt_matrix(signal_len, wavelet, levels)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="dwt_matrix",
            shape=(self.signal_len, self.signal_len),
            initializer=keras.initializers.Constant(self._matrix),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        """Forward DWT. x: (B, 1, T, 1) → (B, 1, T, 1) packed coefficients."""
        # x: (B, 1, T, 1) → squeeze to (B, T) → matmul → reshape back
        x_flat = x[:, 0, :, 0]  # (B, T)
        coeffs = keras.ops.matmul(x_flat, keras.ops.transpose(self.W))  # (B, T)
        return coeffs[:, None, :, None]  # (B, 1, T, 1)

    def get_config(self):
        config = super().get_config()
        config.update({"wavelet": self.wavelet, "levels": self.levels, "signal_len": self.signal_len})
        return config


@keras.saving.register_keras_serializable(package="compressionkit")
class iDWT1D(keras.layers.Layer):
    """Multilevel 1-D inverse DWT via precomputed transform matrix.

    Input:  ``(B, 1, T, 1)`` — packed wavelet coefficients.
    Output: ``(B, 1, T, 1)`` — reconstructed signal.
    """

    def __init__(self, wavelet: str = "bior4.4", levels: int = 6, signal_len: int = 512, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.wavelet = wavelet
        self.levels = levels
        self.signal_len = signal_len
        self._matrix = _build_idwt_matrix(signal_len, wavelet, levels)

    def build(self, input_shape):
        self.W_inv = self.add_weight(
            name="idwt_matrix",
            shape=(self.signal_len, self.signal_len),
            initializer=keras.initializers.Constant(self._matrix),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        """Inverse DWT. x: (B, 1, T, 1) → (B, 1, T, 1) reconstructed signal."""
        x_flat = x[:, 0, :, 0]  # (B, T)
        signal = keras.ops.matmul(x_flat, keras.ops.transpose(self.W_inv))  # (B, T)
        return signal[:, None, :, None]  # (B, 1, T, 1)

    def get_config(self):
        config = super().get_config()
        config.update({"wavelet": self.wavelet, "levels": self.levels, "signal_len": self.signal_len})
        return config


# ---------------------------------------------------------------------------
# Subband normalization
# ---------------------------------------------------------------------------


def _compute_subband_boundaries(signal_len: int, wavelet: str, levels: int) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each DWT subband.

    Order: [cA_L, dL, d(L-1), ..., d1] matching the packing convention.
    """
    if pywt is None:
        raise ImportError("pywt is required")
    dummy = np.zeros(signal_len, dtype=np.float32)
    coeffs = pywt.wavedec(dummy, wavelet, level=levels, mode="periodization")
    boundaries = []
    offset = 0
    for c in coeffs:
        boundaries.append((offset, offset + len(c)))
        offset += len(c)
    return boundaries


def _compute_subband_scales(signal_len: int, wavelet: str, levels: int) -> np.ndarray:
    """Compute per-coefficient normalization scales from the DWT matrix.

    Uses a synthetic broadband signal (white noise) to estimate subband
    energy. For real signals the mismatch is data-dependent; use
    ``_compute_empirical_subband_scales`` when training data is available.
    """
    W = _build_dwt_matrix(signal_len, wavelet, levels)
    # Per-coefficient expected std for unit-variance input
    coeff_stds = np.sqrt(np.sum(W**2, axis=1))  # (signal_len,)
    # Average within each subband for a cleaner normalization
    boundaries = _compute_subband_boundaries(signal_len, wavelet, levels)
    scales = np.zeros(signal_len, dtype=np.float32)
    for start, end in boundaries:
        subband_scale = np.mean(coeff_stds[start:end])
        scales[start:end] = subband_scale
    # Avoid division by zero
    scales = np.maximum(scales, 1e-8)
    return scales


def _compute_empirical_subband_scales(signal_len: int, wavelet: str, levels: int, *, ecg: bool = True) -> np.ndarray:
    """Return per-coefficient scales based on typical ECG/PPG subband energy.

    Hardcoded values derived from PTB-XL lead-1 (ECG, 500 Hz, 512 samples,
    bior4.4, 6 levels). These can be overridden via the SubbandNorm1D's
    ``custom_scales`` parameter.
    """
    # Empirical per-subband std from PTB-XL ECG (lead 1, 1800 frames)
    # wavelet=bior4.4, levels=6, signal_len=512
    ecg_bior44_stds = {
        "cA6": 0.9915,
        "d6": 0.6054,
        "d5": 0.4470,
        "d4": 0.2452,
        "d3": 0.0720,
        "d2": 0.0227,
        "d1": 0.0092,
    }
    boundaries = _compute_subband_boundaries(signal_len, wavelet, levels)
    scales = np.zeros(signal_len, dtype=np.float32)
    names = ["cA6", "d6", "d5", "d4", "d3", "d2", "d1"]
    for (start, end), name in zip(boundaries, names):
        if ecg and name in ecg_bior44_stds:
            scales[start:end] = ecg_bior44_stds[name]
        else:
            # Fallback to equal scale
            scales[start:end] = 1.0
    scales = np.maximum(scales, 1e-8)
    return scales


@keras.saving.register_keras_serializable(package="compressionkit")
class SubbandNorm1D(keras.layers.Layer):
    """Normalize DWT coefficients by per-subband scale factors.

    Divides each coefficient by its subband's expected standard deviation,
    bringing all subbands to approximately the same scale. This allows
    downstream encoder/VQ to treat all positions equally.

    Input/Output: ``(B, 1, T, 1)``
    """

    def __init__(
        self,
        wavelet: str = "bior4.4",
        levels: int = 6,
        signal_len: int = 512,
        use_empirical: bool = True,
        **kwargs,
    ):
        super().__init__(trainable=False, **kwargs)
        self.wavelet = wavelet
        self.levels = levels
        self.signal_len = signal_len
        self.use_empirical = use_empirical
        if use_empirical:
            self._scales = _compute_empirical_subband_scales(signal_len, wavelet, levels)
        else:
            self._scales = _compute_subband_scales(signal_len, wavelet, levels)

    def build(self, input_shape):
        self.scales = self.add_weight(
            name="subband_scales",
            shape=(self.signal_len,),
            initializer=keras.initializers.Constant(self._scales),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        """Normalize: x / scales."""
        x_flat = x[:, 0, :, 0]  # (B, T)
        normed = x_flat / self.scales  # broadcast (T,)
        return normed[:, None, :, None]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "wavelet": self.wavelet,
                "levels": self.levels,
                "signal_len": self.signal_len,
                "use_empirical": self.use_empirical,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="compressionkit")
class InverseSubbandNorm1D(keras.layers.Layer):
    """Denormalize DWT coefficients — inverse of SubbandNorm1D.

    Multiplies each coefficient by its subband's scale factor.

    Input/Output: ``(B, 1, T, 1)``
    """

    def __init__(
        self,
        wavelet: str = "bior4.4",
        levels: int = 6,
        signal_len: int = 512,
        use_empirical: bool = True,
        **kwargs,
    ):
        super().__init__(trainable=False, **kwargs)
        self.wavelet = wavelet
        self.levels = levels
        self.signal_len = signal_len
        self.use_empirical = use_empirical
        if use_empirical:
            self._scales = _compute_empirical_subband_scales(signal_len, wavelet, levels)
        else:
            self._scales = _compute_subband_scales(signal_len, wavelet, levels)

    def build(self, input_shape):
        self.scales = self.add_weight(
            name="subband_scales",
            shape=(self.signal_len,),
            initializer=keras.initializers.Constant(self._scales),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        """Denormalize: x * scales."""
        x_flat = x[:, 0, :, 0]  # (B, T)
        denormed = x_flat * self.scales  # broadcast (T,)
        return denormed[:, None, :, None]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "wavelet": self.wavelet,
                "levels": self.levels,
                "signal_len": self.signal_len,
                "use_empirical": self.use_empirical,
            }
        )
        return config
