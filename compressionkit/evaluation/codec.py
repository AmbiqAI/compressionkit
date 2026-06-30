"""Uniform codec interface for evaluation.

Defines a small Protocol so any codec (classical or neural) can be plugged
into the evaluation harness, scorecard runner, adversarial battery, etc.

Adapters wrap existing implementations without modifying them:

- :class:`IdentityCodec` — passthrough; useful for sanity checks.
- :class:`SpihtAcCodec` — bior4.4 DWT + SPIHT with optional arithmetic coding.
- (later) ``RvqCodec`` — wraps a trained Keras RVQ autoencoder.

The :class:`Codec` protocol is intentionally minimal:

    encoded = codec.encode(frame)
    recon   = codec.decode(encoded)
    bits    = encoded.nbits

Codecs may attach arbitrary side information to :attr:`EncodedFrame.side`
(e.g. token IDs, per-stage residual norms, quantization distances). Neural
codecs should populate this so the QoS module can compute confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from compressionkit.dsp.spiht import spiht_decode, spiht_encode
from compressionkit.dsp.wavelet import WaveletCoeffs, bayes_shrink_signal, dwt_forward, dwt_inverse

__all__ = [
    "BayesShrinkSpihtCodec",
    "Codec",
    "EncodedFrame",
    "FilterSpihtCodec",
    "IdentityCodec",
    "LearnedShrinkSpihtCodec",
    "SpihtAcCodec",
    "compression_ratio",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EncodedFrame:
    """Output of :meth:`Codec.encode`.

    Attributes:
        payload: Opaque encoded representation. Bytes for bitstream codecs;
            arbitrary array (e.g. token IDs) for neural codecs.
        nbits: Actual number of bits used for this frame's payload. This is
            what should be used to compute the *true* compression ratio.
        side: Optional side information attached by the codec. Common keys:
            ``"token_ids"``, ``"quant_distances"``, ``"residual_norms"``,
            ``"metadata"`` (e.g. SPIHT decode metadata).
    """

    payload: Any
    nbits: int
    side: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Codec(Protocol):
    """Uniform codec interface for evaluation."""

    name: str
    """Human-readable codec name (e.g. ``"spiht_ac"``, ``"rvq_64hz_04x"``)."""

    modality: str
    """``"ppg"`` or ``"ecg"``."""

    sample_rate: int
    """Frame sample rate in Hz."""

    frame_size: int
    """Number of samples per frame at the input layer."""

    target_cr: float
    """Nominal target compression ratio (informational; true CR comes from
    :attr:`EncodedFrame.nbits`)."""

    def encode(self, frame: np.ndarray) -> EncodedFrame:
        """Encode a single ``(frame_size,)`` or ``(frame_size, channels)`` frame."""
        ...

    def decode(self, encoded: EncodedFrame) -> np.ndarray:
        """Decode back to a frame with the same shape as the input."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compression_ratio(codec: Codec, encoded: EncodedFrame, bits_per_sample: int | None = None) -> float:
    """True compression ratio for a given encoded frame.

    Args:
        codec: The codec instance (for ``frame_size`` and, if available,
            ``bits_per_sample``).
        encoded: The encoded frame.
        bits_per_sample: Raw bit depth of the source signal. When *None*,
            falls back to ``codec.bits_per_sample`` if the codec exposes
            that attribute, otherwise to 16. Pass an explicit value to
            override.

    Returns:
        Raw bits / encoded bits. Returns ``inf`` when ``nbits == 0``.
    """
    if bits_per_sample is None:
        bits_per_sample = int(getattr(codec, "bits_per_sample", 16))
    raw_bits = codec.frame_size * bits_per_sample
    if encoded.nbits <= 0:
        return math.inf
    return raw_bits / encoded.nbits


# ---------------------------------------------------------------------------
# IdentityCodec — sanity check
# ---------------------------------------------------------------------------


@dataclass
class IdentityCodec:
    """Passthrough codec. Decoded output equals the input exactly."""

    name: str = "identity"
    modality: str = "ppg"
    sample_rate: int = 64
    frame_size: int = 320
    target_cr: float = 1.0

    def encode(self, frame: np.ndarray) -> EncodedFrame:
        arr = np.asarray(frame, dtype=np.float32)
        nbits = int(arr.size) * 32
        return EncodedFrame(payload=arr.copy(), nbits=nbits)

    def decode(self, encoded: EncodedFrame) -> np.ndarray:
        return np.asarray(encoded.payload, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# SpihtAcCodec — bior4.4 DWT + SPIHT (+ AC)
# ---------------------------------------------------------------------------


@dataclass
class SpihtAcCodec:
    """SPIHT-(AC) codec wrapping :mod:`compressionkit.dsp.spiht`.

    The bit budget per frame is derived from ``target_cr`` and ``frame_size``
    at construction time::

        max_bits = floor(frame_size * bits_per_sample / target_cr)

    Single-channel only. For multi-channel inputs, use a per-channel wrapper
    (the comparison code in :mod:`experiments.scripts.compare_12lead` shows the pattern).
    """

    name: str = "spiht_ac"
    modality: str = "ppg"
    sample_rate: int = 64
    frame_size: int = 320
    target_cr: float = 8.0
    wavelet: str = "bior4.4"
    levels: int = 6
    use_ac: bool = True
    bits_per_sample: int = 16

    def __post_init__(self) -> None:
        self.max_bits = int(self.frame_size * self.bits_per_sample / self.target_cr)
        if self.max_bits <= 0:
            raise ValueError(
                f"max_bits computed as {self.max_bits} (frame_size={self.frame_size}, "
                f"bits_per_sample={self.bits_per_sample}, target_cr={self.target_cr})"
            )

    def encode(self, frame: np.ndarray) -> EncodedFrame:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"SpihtAcCodec expects 1-D input, got shape {arr.shape}")
        coeffs = dwt_forward(arr, levels=self.levels, wavelet=self.wavelet)
        bitstream, meta = spiht_encode(
            coeffs.approx,
            coeffs.details,
            max_bits=self.max_bits,
            use_ac=self.use_ac,
        )
        # Use the encoder's exact bit count (``meta['n_bits']``) rather than
        # ``len(bitstream) * 8``: the byte buffer is padded up to a byte
        # boundary (and AC adds a flush tail), which would overstate the
        # real bitrate by up to 7-15 bits/frame.
        actual_nbits = int(meta.get("n_bits", len(bitstream) * 8))
        return EncodedFrame(
            payload=bitstream,
            nbits=actual_nbits,
            side={"metadata": meta},
        )

    def decode(self, encoded: EncodedFrame) -> np.ndarray:
        meta = encoded.side.get("metadata")
        if meta is None:
            raise ValueError("SpihtAcCodec.decode requires metadata in encoded.side")
        approx_r, details_r = spiht_decode(encoded.payload, meta)
        recon = dwt_inverse(
            WaveletCoeffs(approx=approx_r, details=details_r),
            wavelet=self.wavelet,
        )
        return recon[: self.frame_size].astype(np.float32)


@dataclass
class BayesShrinkSpihtCodec:
    """Hybrid baseline: BayesShrink denoise in DWT domain, then SPIHT encode."""

    name: str = "bayes_shrink_spiht"
    modality: str = "ppg"
    sample_rate: int = 64
    frame_size: int = 320
    target_cr: float = 8.0
    wavelet: str = "bior4.4"
    levels: int = 6
    use_ac: bool = True
    bits_per_sample: int = 16

    def __post_init__(self) -> None:
        self._spiht = SpihtAcCodec(
            name=self.name,
            modality=self.modality,
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            target_cr=self.target_cr,
            wavelet=self.wavelet,
            levels=self.levels,
            use_ac=self.use_ac,
            bits_per_sample=self.bits_per_sample,
        )
        self.max_bits = self._spiht.max_bits

    def encode(self, frame: np.ndarray) -> EncodedFrame:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"BayesShrinkSpihtCodec expects 1-D input, got shape {arr.shape}")
        denoised = bayes_shrink_signal(arr, levels=self.levels, wavelet=self.wavelet)[: self.frame_size]
        encoded = self._spiht.encode(denoised)
        encoded.side["pre_denoised"] = True
        return encoded

    def decode(self, encoded: EncodedFrame) -> np.ndarray:
        return self._spiht.decode(encoded)


# ---------------------------------------------------------------------------
# LearnedShrinkSpihtCodec — learned wavelet-domain gain + SPIHT (+ AC)
# ---------------------------------------------------------------------------


@dataclass
class LearnedShrinkSpihtCodec:
    """Hybrid: learned wavelet-domain gain denoise, then SPIHT encode.

    ``coeff_denoiser`` maps a packed DWT coefficient vector to a denoised
    coefficient vector (e.g. a gain-only network adapted via
    :func:`compressionkit.models.wavelet_denoiser.as_coeff_denoiser`). When
    ``None`` the codec is an exact passthrough and reduces to plain SPIHT,
    which makes it a clean baseline anchor.

    The denoise runs in the DWT domain; the result is inverted back to the
    signal and handed to the unchanged SPIHT encoder. On-device the two DWTs
    would be fused, but for evaluation correctness this mirrors
    :class:`BayesShrinkSpihtCodec`.
    """

    name: str = "learned_shrink_spiht"
    modality: str = "ppg"
    sample_rate: int = 64
    frame_size: int = 320
    target_cr: float = 8.0
    wavelet: str = "bior4.4"
    levels: int = 6
    use_ac: bool = True
    bits_per_sample: int = 16
    coeff_denoiser: Any = None

    def __post_init__(self) -> None:
        self._spiht = SpihtAcCodec(
            name=self.name,
            modality=self.modality,
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            target_cr=self.target_cr,
            wavelet=self.wavelet,
            levels=self.levels,
            use_ac=self.use_ac,
            bits_per_sample=self.bits_per_sample,
        )
        self.max_bits = self._spiht.max_bits

    def _denoise_signal(self, arr: np.ndarray) -> np.ndarray:
        if self.coeff_denoiser is None:
            return arr
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
        return recon[: self.frame_size].astype(np.float32)

    def encode(self, frame: np.ndarray) -> EncodedFrame:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"LearnedShrinkSpihtCodec expects 1-D input, got shape {arr.shape}")
        denoised = self._denoise_signal(arr)
        encoded = self._spiht.encode(denoised)
        encoded.side["pre_denoised"] = self.coeff_denoiser is not None
        return encoded

    def decode(self, encoded: EncodedFrame) -> np.ndarray:
        return self._spiht.decode(encoded)


# ---------------------------------------------------------------------------
# FilterSpihtCodec — classical bandpass denoise + SPIHT
# ---------------------------------------------------------------------------


@dataclass
class FilterSpihtCodec:
    """Classical baseline: zero-phase Butterworth bandpass denoise, then SPIHT.

    This is the fully classical control for the learned
    :class:`LearnedShrinkSpihtCodec` ("Hybrid"). It uses the *same* architecture
    -- pre-denoise the signal, then hand it to the unchanged SPIHT encoder -- but
    swaps the trained wavelet-gain network for a fixed IIR bandpass. Butterworth
    second-order sections are embedded-portable (fixed coefficients, no dynamic
    allocation), so this baseline is realizable on-device.

    Note on evaluation: when ``low_hz``/``high_hz`` match the clean-truth proxy
    band used by a sweep, this codec measures the *ceiling* of linear denoising
    against that proxy and is partly favoured by construction. Use a deliberately
    different band to estimate realistic deployment behaviour.
    """

    name: str = "filter_spiht"
    modality: str = "ecg"
    sample_rate: int = 256
    frame_size: int = 512
    target_cr: float = 8.0
    wavelet: str = "bior4.4"
    levels: int = 6
    use_ac: bool = True
    bits_per_sample: int = 16
    low_hz: float = 0.5
    high_hz: float = 40.0
    order: int = 3
    forward_backward: bool = True

    def __post_init__(self) -> None:
        from scipy import signal as scipy_signal

        self._spiht = SpihtAcCodec(
            name=self.name,
            modality=self.modality,
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            target_cr=self.target_cr,
            wavelet=self.wavelet,
            levels=self.levels,
            use_ac=self.use_ac,
            bits_per_sample=self.bits_per_sample,
        )
        self.max_bits = self._spiht.max_bits
        nyq = self.sample_rate / 2.0
        high = min(self.high_hz, nyq * 0.95)
        if not 0.0 < self.low_hz < high:
            raise ValueError(
                f"Invalid bandpass cutoffs: low_hz={self.low_hz}, high_hz={self.high_hz} (nyquist={nyq})"
            )
        self._sos = scipy_signal.butter(
            self.order, [self.low_hz / nyq, high / nyq], btype="bandpass", output="sos"
        )

    def _denoise_signal(self, arr: np.ndarray) -> np.ndarray:
        from scipy import signal as scipy_signal

        if self.forward_backward:
            return scipy_signal.sosfiltfilt(self._sos, arr).astype(np.float32)
        return scipy_signal.sosfilt(self._sos, arr).astype(np.float32)

    def encode(self, frame: np.ndarray) -> EncodedFrame:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"FilterSpihtCodec expects 1-D input, got shape {arr.shape}")
        denoised = self._denoise_signal(arr)
        encoded = self._spiht.encode(denoised)
        encoded.side["pre_filtered"] = True
        return encoded

    def decode(self, encoded: EncodedFrame) -> np.ndarray:
        return self._spiht.decode(encoded)
