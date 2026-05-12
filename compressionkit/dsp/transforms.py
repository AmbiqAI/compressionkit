"""Signal transforms for transform-domain compression pipelines.

Provides forward/inverse STFT and DWT helpers that convert 1-D signals
into fixed-shape representations suitable as autoencoder inputs.

**STFT** produces a 2-D time-frequency representation::

    (num_frames, freq_bins, 2)   # channels: [real, imaginary]

**DWT** produces a flat coefficient vector::

    (T,)   # packed subbands: [approx | detail_coarse → detail_fine]

Both support perfect reconstruction (round-trip error ≈ float32 precision).

Design choices:
    - NumPy-only (for data preprocessing / TFRecord caching).
    - Fixed output shapes given signal length and config — no dynamic allocation.
    - Single-signal API ``(T,)`` for simplicity; batch with ``np.vectorize`` or loops.
    - STFT uses real + imaginary channels (lossless, no phase-wrapping issues).
    - DWT delegates to the ``compressionkit.dsp.wavelet`` module.

Example — STFT round-trip::

    cfg = StftConfig(n_fft=64, hop_length=16)
    spec = stft_forward(ecg_signal, cfg)      # (33, 33, 2) for 512-sample signal
    recon = stft_inverse(spec, cfg, len(ecg_signal))
    assert np.allclose(ecg_signal, recon, atol=1e-6)

Example — DWT round-trip::

    cfg = DwtConfig(levels=4, wavelet="haar")
    packed = dwt_pack(ecg_signal, cfg)         # (512,)
    recon = dwt_unpack(packed, cfg, len(ecg_signal))
    assert np.allclose(ecg_signal, recon, atol=1e-6)
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from compressionkit.dsp.wavelet import WaveletCoeffs, dwt_forward, dwt_inverse

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


class StftConfig(BaseModel):
    """Parameters for the short-time Fourier transform."""

    n_fft: int = Field(default=64, description="FFT window length (samples).")
    hop_length: int = Field(default=16, description="Hop between successive frames (samples).")
    window: str = Field(
        default="hann",
        description="Window function name: hann, hamming, blackman, rectangular.",
    )


class DwtConfig(BaseModel):
    """Parameters for the discrete wavelet transform."""

    levels: int = Field(default=4, description="Number of decomposition levels.")
    wavelet: str = Field(default="haar", description="Wavelet name: haar, db4.")


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

_WINDOW_FNS: dict[str, callable] = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
    "bartlett": np.bartlett,
}


def _get_window(name: str, length: int) -> np.ndarray:
    """Return a symmetric window of the given *length*.

    Args:
        name: Window name.  ``"rectangular"`` returns all ones.
        length: Number of samples.

    Returns:
        Window array of shape ``(length,)`` and dtype ``float32``.
    """
    if name == "rectangular":
        return np.ones(length, dtype=np.float32)
    fn = _WINDOW_FNS.get(name)
    if fn is None:
        raise ValueError(f"Unknown window '{name}'. Choose from: {[*list(_WINDOW_FNS), 'rectangular']}")
    return fn(length).astype(np.float32)


# ---------------------------------------------------------------------------
# STFT — forward
# ---------------------------------------------------------------------------


def stft_num_frames(signal_length: int, cfg: StftConfig) -> int:
    """Number of STFT frames for a given signal length.

    Uses center-padding of ``n_fft // 2`` on each side so the first window
    is centered at sample 0.
    """
    return 1 + signal_length // cfg.hop_length


def stft_output_shape(signal_length: int, cfg: StftConfig) -> tuple[int, int, int]:
    """Output shape of :func:`stft_forward`: ``(num_frames, freq_bins, 2)``.

    The last axis holds ``[real, imaginary]`` components.
    """
    n_frames = stft_num_frames(signal_length, cfg)
    freq_bins = cfg.n_fft // 2 + 1
    return (n_frames, freq_bins, 2)


def stft_forward(x: np.ndarray, cfg: StftConfig) -> np.ndarray:
    """Compute the windowed STFT of a 1-D signal.

    Args:
        x: Real-valued signal of shape ``(T,)``.
        cfg: STFT parameters.

    Returns:
        Real/imaginary spectrogram of shape ``(num_frames, freq_bins, 2)``,
        dtype ``float32``.
    """
    x = np.asarray(x, dtype=np.float32)
    N = cfg.n_fft
    H = cfg.hop_length

    # Center-pad so the first frame is centered at t=0.
    pad_left = N // 2
    pad_right = N // 2
    x_pad = np.pad(x, (pad_left, pad_right), mode="constant")

    # Extract overlapping frames using stride tricks (zero-copy).
    n_frames = 1 + (len(x_pad) - N) // H
    frames = np.lib.stride_tricks.sliding_window_view(x_pad, N)[::H][:n_frames]

    # Windowed FFT (real-input → half-spectrum).
    win = _get_window(cfg.window, N)
    spectrum = np.fft.rfft(frames * win, n=N)  # (n_frames, N//2+1) complex64

    return np.stack([spectrum.real, spectrum.imag], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# STFT — inverse
# ---------------------------------------------------------------------------


def stft_inverse(spec: np.ndarray, cfg: StftConfig, length: int) -> np.ndarray:
    """Reconstruct a signal from its STFT (overlap-add).

    Args:
        spec: Spectrogram of shape ``(num_frames, freq_bins, 2)`` as returned
            by :func:`stft_forward`.
        cfg: STFT parameters (must match those used in the forward pass).
        length: Original signal length to truncate to.

    Returns:
        Reconstructed signal of shape ``(length,)``, dtype ``float32``.
    """
    N = cfg.n_fft
    H = cfg.hop_length

    # Re-assemble complex spectrum and invert FFT.
    spectrum = spec[..., 0] + 1j * spec[..., 1]
    frames = np.fft.irfft(spectrum, n=N).astype(np.float32)  # (n_frames, N)

    n_frames = len(frames)
    padded_len = N + (n_frames - 1) * H

    # Overlap-add the (already windowed) frames + track window sum.
    output = np.zeros(padded_len, dtype=np.float32)
    win_sum = np.zeros(padded_len, dtype=np.float32)
    win = _get_window(cfg.window, N)

    for i in range(n_frames):
        s = i * H
        output[s : s + N] += frames[i]
        win_sum[s : s + N] += win

    # Normalize by the window sum (COLA denominator).
    win_sum = np.maximum(win_sum, 1e-8)
    output /= win_sum

    # Strip center-padding.
    pad = N // 2
    return output[pad : pad + length]


# ---------------------------------------------------------------------------
# STFT — convenience conversions
# ---------------------------------------------------------------------------


def stft_to_mag_phase(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert real/imag spectrogram to magnitude and phase.

    Args:
        spec: ``(num_frames, freq_bins, 2)`` — real, imaginary.

    Returns:
        ``(magnitude, phase)`` each of shape ``(num_frames, freq_bins)``.
    """
    cpx = spec[..., 0] + 1j * spec[..., 1]
    return np.abs(cpx).astype(np.float32), np.angle(cpx).astype(np.float32)


def mag_phase_to_stft(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Convert magnitude and phase back to real/imag spectrogram.

    Args:
        magnitude: ``(num_frames, freq_bins)``.
        phase: ``(num_frames, freq_bins)`` in radians.

    Returns:
        ``(num_frames, freq_bins, 2)`` real/imaginary spectrogram.
    """
    cpx = magnitude * np.exp(1j * phase)
    return np.stack([cpx.real, cpx.imag], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# DWT — packing
# ---------------------------------------------------------------------------


def dwt_band_sizes(signal_length: int, cfg: DwtConfig) -> list[int]:
    """Subband sizes for a multi-level DWT.

    Returns a list ``[approx, detail_coarsest, ..., detail_finest]``.
    For a 512-sample signal with 4 levels: ``[32, 32, 64, 128, 256]``.

    Args:
        signal_length: Length of the input signal (must be divisible by
            ``2**levels``).
        cfg: DWT parameters.

    Returns:
        List of ``levels + 1`` integers summing to *signal_length*.
    """
    if signal_length % (2**cfg.levels) != 0:
        raise ValueError(f"signal_length {signal_length} must be divisible by 2**levels = {2**cfg.levels}")
    sizes: list[int] = []
    remaining = signal_length
    for _ in range(cfg.levels):
        remaining //= 2
        sizes.append(remaining)  # detail at this level
    sizes.append(remaining)  # approximation (same size as coarsest detail)

    # Reverse so order is [approx, detail_coarsest, ..., detail_finest].
    sizes.reverse()
    return sizes


def dwt_output_shape(signal_length: int, cfg: DwtConfig) -> tuple[int]:
    """Output shape of :func:`dwt_pack`: ``(signal_length,)``."""
    return (signal_length,)


def dwt_pack(x: np.ndarray, cfg: DwtConfig) -> np.ndarray:
    """Transform a signal via multi-level DWT and pack coefficients flat.

    Layout: ``[approx | detail_coarsest | … | detail_finest]``

    Args:
        x: Signal of shape ``(T,)``.
        cfg: DWT parameters.

    Returns:
        Packed coefficient vector of shape ``(T,)``, dtype ``float32``.
    """
    x = np.asarray(x, dtype=np.float32)
    coeffs = dwt_forward(x, levels=cfg.levels, wavelet=cfg.wavelet)
    # coeffs.details: [finest, ..., coarsest] (per wavelet.py convention)
    bands = [coeffs.approx, *list(reversed(coeffs.details))]
    return np.concatenate(bands).astype(np.float32)


def dwt_unpack(packed: np.ndarray, cfg: DwtConfig, signal_length: int) -> np.ndarray:
    """Reconstruct a signal from packed DWT coefficients.

    Args:
        packed: Flat coefficient vector from :func:`dwt_pack`.
        cfg: DWT parameters (must match the forward pass).
        signal_length: Original signal length.

    Returns:
        Reconstructed signal of shape ``(signal_length,)``, dtype ``float32``.
    """
    packed = np.asarray(packed, dtype=np.float32)
    sizes = dwt_band_sizes(signal_length, cfg)

    # Split packed vector into bands.
    bands: list[np.ndarray] = []
    offset = 0
    for s in sizes:
        bands.append(packed[offset : offset + s])
        offset += s

    # bands[0] = approx, bands[1] = coarsest detail, ..., bands[-1] = finest
    # wavelet.py expects details ordered [finest, ..., coarsest]
    approx = bands[0]
    details = list(reversed(bands[1:]))
    coeffs = WaveletCoeffs(approx=approx, details=details)
    return dwt_inverse(coeffs, wavelet=cfg.wavelet).astype(np.float32)


# ---------------------------------------------------------------------------
# Shape summary
# ---------------------------------------------------------------------------


def describe_transform(signal_length: int, sample_rate: float, cfg: StftConfig | DwtConfig) -> str:
    """Return a human-readable summary of the transform output.

    Args:
        signal_length: Number of samples in the input signal.
        sample_rate: Sampling rate in Hz.
        cfg: Transform configuration (STFT or DWT).

    Returns:
        Multi-line description string.
    """
    lines: list[str] = []
    duration_s = signal_length / sample_rate

    if isinstance(cfg, StftConfig):
        shape = stft_output_shape(signal_length, cfg)
        freq_res = sample_rate / cfg.n_fft
        time_res_ms = 1000.0 * cfg.hop_length / sample_rate
        max_freq = sample_rate / 2
        lines.append(f"STFT  n_fft={cfg.n_fft}  hop={cfg.hop_length}  window={cfg.window}")
        lines.append(f"  Input:  ({signal_length},)  [{duration_s:.2f}s @ {sample_rate:.0f} Hz]")
        lines.append(f"  Output: {shape}  (frames, freq_bins, real/imag)")
        lines.append(f"  Freq resolution: {freq_res:.1f} Hz  (0 – {max_freq:.0f} Hz)")
        lines.append(f"  Time resolution: {time_res_ms:.1f} ms")
        lines.append(f"  Total coefficients: {shape[0] * shape[1] * shape[2]}")
    elif isinstance(cfg, DwtConfig):
        sizes = dwt_band_sizes(signal_length, cfg)
        lines.append(f"DWT  levels={cfg.levels}  wavelet={cfg.wavelet}")
        lines.append(f"  Input:  ({signal_length},)  [{duration_s:.2f}s @ {sample_rate:.0f} Hz]")
        lines.append(f"  Output: ({signal_length},)  packed flat")
        lines.append("  Band layout [approx, detail_coarse → fine]:")
        band_names = ["approx"] + [f"detail_L{cfg.levels - i}" for i in range(cfg.levels)]
        nyquist = sample_rate / 2
        for name, size in zip(band_names, sizes, strict=False):
            if name == "approx":
                lines.append(f"    {name:14s}  {size:5d} samples  (0 – {nyquist / 2**cfg.levels:.1f} Hz)")
            else:
                level = int(name.split("L")[1])
                lo = nyquist / 2**level
                hi = nyquist / 2 ** (level - 1)
                lines.append(f"    {name:14s}  {size:5d} samples  ({lo:.1f} – {hi:.1f} Hz)")
    else:
        raise TypeError(f"Expected StftConfig or DwtConfig, got {type(cfg)}")

    return "\n".join(lines)


__all__ = [
    "DwtConfig",
    "StftConfig",
    "describe_transform",
    "dwt_band_sizes",
    "dwt_output_shape",
    "dwt_pack",
    "dwt_unpack",
    "mag_phase_to_stft",
    "stft_forward",
    "stft_inverse",
    "stft_num_frames",
    "stft_output_shape",
    "stft_to_mag_phase",
]
