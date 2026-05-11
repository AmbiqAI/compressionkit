"""Classical DSP primitives for compression baselines and transform-domain pipelines.

This subpackage holds purely-numpy signal-processing building blocks that
stand apart from the learned RVQ autoencoder stack. They are used to:

- Build transform-domain inputs (STFT / DWT) that feed the RVQ pipeline
  when operating in a frequency or wavelet domain instead of raw time.
- Provide DSP baselines (wavelet-thresholding compression, SPIHT bit-stream
  scaffolding) for apples-to-apples comparisons against the learned models.

All modules are CPU-only numpy and safe to import without TensorFlow.
"""

from compressionkit.dsp.spiht import BitReader, BitWriter, spiht_decode, spiht_encode
from compressionkit.dsp.transforms import (
    DwtConfig,
    StftConfig,
    describe_transform,
    dwt_band_sizes,
    dwt_output_shape,
    dwt_pack,
    dwt_unpack,
    mag_phase_to_stft,
    stft_forward,
    stft_inverse,
    stft_num_frames,
    stft_output_shape,
    stft_to_mag_phase,
)
from compressionkit.dsp.wavelet import (
    WaveletCoeffs,
    WaveletCompressed,
    WaveletFilters,
    apply_threshold,
    compress_signal,
    compute_prd,
    compute_step_sizes,
    compute_thresholds,
    decompress_signal,
    dequantize_coeffs,
    dwt_forward,
    dwt_inverse,
    pack_coeffs,
    quantize_coeffs,
    unpack_coeffs,
)

__all__ = [
    # spiht
    "BitReader",
    "BitWriter",
    # transforms
    "DwtConfig",
    "StftConfig",
    # wavelet
    "WaveletCoeffs",
    "WaveletCompressed",
    "WaveletFilters",
    "apply_threshold",
    "compress_signal",
    "compute_prd",
    "compute_step_sizes",
    "compute_thresholds",
    "decompress_signal",
    "dequantize_coeffs",
    "describe_transform",
    "dwt_band_sizes",
    "dwt_forward",
    "dwt_inverse",
    "dwt_output_shape",
    "dwt_pack",
    "dwt_unpack",
    "mag_phase_to_stft",
    "pack_coeffs",
    "quantize_coeffs",
    "spiht_decode",
    "spiht_encode",
    "stft_forward",
    "stft_inverse",
    "stft_num_frames",
    "stft_output_shape",
    "stft_to_mag_phase",
    "unpack_coeffs",
]
