"""Multi-resolution STFT loss."""

from __future__ import annotations

import keras.ops as ops


def build_multi_scale_spectral_loss(
    weight: float,
    fft_sizes: list[int],
) -> callable:
    """Multi-resolution STFT loss combining spectral convergence and log-mag L1.

    For each FFT size *n* the loss computes:

    * **Spectral convergence** -- Frobenius-norm ratio of the magnitude
      difference to the reference magnitude.
    * **Log-magnitude L1** -- mean absolute difference of log-magnitude
      spectra.

    The two terms are summed per scale and the result is averaged over
    scales and weighted by *weight*.

    Input shape is ``(B, 1, T, C)``; the time axis is axis 2.

    Args:
        weight: Scalar multiplier applied to the combined spectral loss.
        fft_sizes: List of FFT sizes to use (e.g. ``[256, 512, 1024]``).

    Returns:
        A callable ``spectral_loss(y_true, y_pred) -> scalar``.
    """

    def _stft_mag(x, fft_length: int, hop_length: int):
        real, imag = ops.stft(
            x,
            sequence_length=fft_length,
            sequence_stride=hop_length,
            fft_length=fft_length,
        )
        return ops.sqrt(ops.square(real) + ops.square(imag) + 1e-8)

    def spectral_loss(y_true, y_pred):
        num_ch = y_true.shape[-1] if y_true.shape[-1] is not None else 1
        total = ops.convert_to_tensor(0.0)
        for ch in range(num_ch):
            yt = y_true[:, 0, :, ch]
            yp = y_pred[:, 0, :, ch]
            for n in fft_sizes:
                hop = n // 4
                st = _stft_mag(yt, n, hop)
                sp = _stft_mag(yp, n, hop)

                diff_sq = ops.sum(ops.square(st - sp), axis=(1, 2))
                ref_sq = ops.sum(ops.square(st), axis=(1, 2))
                sc = ops.mean(ops.sqrt(diff_sq + 1e-8) / (ops.sqrt(ref_sq + 1e-8) + 1e-6))

                log_mag = ops.mean(ops.abs(ops.log(st + 1e-8) - ops.log(sp + 1e-8)))

                total = total + sc + log_mag

        return weight * total / (len(fft_sizes) * num_ch)

    spectral_loss.__name__ = "spectral_loss"
    return spectral_loss
