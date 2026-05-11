"""Evaluation modules for compressionkit."""

from compressionkit.evaluation.artifacts import save_sample_artifacts
from compressionkit.evaluation.metrics import (
    PRD,
    TruePRD,
    compute_ecg_hr_hrv,
    compute_ppg_physiokit_metrics,
    compute_signal_metrics,
    summarize_ecg_alignment,
    summarize_physiokit_alignment,
)
from compressionkit.evaluation.noise import (
    estimate_bandpass_residual_noise,
    estimate_ecg_noise_floor,
    estimate_ecg_qrs_snr,
    estimate_hf_noise_power,
    estimate_ppg_noise_floor,
)
from compressionkit.evaluation.overlap_add import (
    evaluate_long_recordings,
    reconstruct_overlap_add,
)
from compressionkit.evaluation.spectral_metrics import (
    ECG_DEFAULT_BANDS,
    ECG_DEFAULT_COHERENCE_BAND,
    ECG_DEFAULT_FREQ_WEIGHTS,
    PPG_DEFAULT_BANDS,
    PPG_DEFAULT_COHERENCE_BAND,
    PPG_DEFAULT_FREQ_WEIGHTS,
    psd_band_error,
    spectral_coherence,
    weighted_freq_prd,
)
from compressionkit.evaluation.stitching import (
    STITCH_METHODS,
    reconstruct_hard_concat,
    reconstruct_linear_crossfade,
    reconstruct_tukey_overlap_add,
    seam_discontinuity_ratio,
    stitch,
)

__all__ = [
    "ECG_DEFAULT_BANDS",
    "ECG_DEFAULT_COHERENCE_BAND",
    "ECG_DEFAULT_FREQ_WEIGHTS",
    "PPG_DEFAULT_BANDS",
    "PPG_DEFAULT_COHERENCE_BAND",
    "PPG_DEFAULT_FREQ_WEIGHTS",
    "PRD",
    "STITCH_METHODS",
    "TruePRD",
    "compute_ecg_hr_hrv",
    "compute_ppg_physiokit_metrics",
    "compute_signal_metrics",
    "estimate_bandpass_residual_noise",
    "estimate_ecg_noise_floor",
    "estimate_ecg_qrs_snr",
    "estimate_hf_noise_power",
    "estimate_ppg_noise_floor",
    "evaluate_long_recordings",
    "psd_band_error",
    "reconstruct_hard_concat",
    "reconstruct_linear_crossfade",
    "reconstruct_overlap_add",
    "reconstruct_tukey_overlap_add",
    "save_sample_artifacts",
    "seam_discontinuity_ratio",
    "spectral_coherence",
    "stitch",
    "summarize_ecg_alignment",
    "summarize_physiokit_alignment",
    "weighted_freq_prd",
]
