"""Shared deploy/HuggingFace artifact names.

Keep this module small: it defines the common filenames and NPZ array keys
that must agree across export, publishing, runtime loading, docs, and tests.
The package manifests remain flexible JSON objects; these constants guard the
stable release-contract surface where string drift is costly.
"""

from __future__ import annotations

from enum import StrEnum


class ArtifactFile(StrEnum):
    """Common deploy and HuggingFace bundle filenames."""

    CHECKSUMS = "checksums.json"
    CODEBOOK_HEADER = "codebook.h"
    CODEBOOK_NPZ = "codebook.npz"
    CODEC_SPEC = "codec_spec.json"
    DECODER_FLOAT32_HEADER = "_decoder_float32.h"
    DECODER_FLOAT32_TFLITE = "decoder_float32.tflite"
    DECODER_HEADER = "decoder.h"
    DECODER_INT8_HF_TFLITE = "decoder_int8.tflite"
    DECODER_KERAS = "decoder.keras"
    DECODER_TFLITE = "decoder.tflite"
    DEMO_RECORDINGS = "demo_recordings.npz"
    DEMO_RECORDINGS_MANIFEST = "demo_recordings_manifest.json"
    DENOISER_GAIN_MODEL = "denoiser_gain_model.keras"
    DENOISER_HEADER = "denoiser_gain_model.h"
    DENOISER_TFLITE = "denoiser_gain_model.tflite"
    DENOISER_TRAIN_CONFIG = "denoiser_train_config.json"
    DEPLOY_MANIFEST = "deploy_manifest.json"
    ENCODER_HEADER = "encoder.h"
    ENCODER_FLOAT32_HEADER = "_encoder_float32.h"
    ENCODER_FLOAT32_TFLITE = "encoder_float32.tflite"
    ENCODER_FP16_HEADER = "encoder_fp16.h"
    ENCODER_FP16_TFLITE = "encoder_fp16.tflite"
    ENCODER_INT16X8_HEADER = "encoder_int16x8.h"
    ENCODER_INT16X8_TFLITE = "encoder_int16x8.tflite"
    ENCODER_INT8_HF_TFLITE = "encoder_int8.tflite"
    ENCODER_KERAS = "encoder.keras"
    ENCODER_TFLITE = "encoder.tflite"
    HF_CONFIG = "config.json"
    HYBRID_MANIFEST = "hybrid_manifest.json"
    MODEL_CARD = "model_card.json"
    PRIOR_INT8_HEADER = "prior_int8.h"
    PRIOR_INT8_TFLITE = "prior_int8.tflite"
    PRIOR_MANIFEST = "prior_manifest.json"
    QUALITY_SCORECARD = "quality_scorecard.json"
    QUANTIZATION_REPORT = "quantization_report.json"
    REFERENCE_VECTORS = "reference_vectors.npz"
    SAMPLE_DATA = "sample_data.npz"
    SAMPLE_STIMULUS = "sample_stimulus.npz"
    SCORECARD = "scorecard.json"
    SPIHT_APP_CONFIG_HEADER = "spiht_app_config.h"
    SPIHT_CONFIG = "spiht_config.json"


class SampleArray(StrEnum):
    """Common arrays in sample/reference NPZ artifacts."""

    INPUTS = "inputs"
    INPUT_FRAMES = "input_frames"
    INDICES = "indices"
    RECONSTRUCTIONS = "reconstructions"
    STIMULUS = "stimulus"
    TARGETS = "targets"


class DemoArray(StrEnum):
    """Stable arrays in the real-recordings browser-demo NPZ artifact."""

    SAMPLE_RATE = "sample_rate"
    SIGNALS = "signals"
    SOURCE_RECORDS = "source_records"
    SOURCE_SAMPLE_RATES = "source_sample_rates"
    START_SECONDS = "start_seconds"


RVQ_HF_FILE_RENAMES: tuple[tuple[ArtifactFile, ArtifactFile], ...] = (
    (ArtifactFile.ENCODER_TFLITE, ArtifactFile.ENCODER_INT8_HF_TFLITE),
    (ArtifactFile.ENCODER_HEADER, ArtifactFile.ENCODER_HEADER),
    (ArtifactFile.ENCODER_FLOAT32_TFLITE, ArtifactFile.ENCODER_FLOAT32_TFLITE),
    (ArtifactFile.ENCODER_FP16_TFLITE, ArtifactFile.ENCODER_FP16_TFLITE),
    (ArtifactFile.ENCODER_FP16_HEADER, ArtifactFile.ENCODER_FP16_HEADER),
    (ArtifactFile.ENCODER_INT16X8_TFLITE, ArtifactFile.ENCODER_INT16X8_TFLITE),
    (ArtifactFile.ENCODER_INT16X8_HEADER, ArtifactFile.ENCODER_INT16X8_HEADER),
    (ArtifactFile.ENCODER_KERAS, ArtifactFile.ENCODER_KERAS),
    (ArtifactFile.DECODER_FLOAT32_TFLITE, ArtifactFile.DECODER_FLOAT32_TFLITE),
    (ArtifactFile.DECODER_TFLITE, ArtifactFile.DECODER_INT8_HF_TFLITE),
    (ArtifactFile.DECODER_INT8_HF_TFLITE, ArtifactFile.DECODER_INT8_HF_TFLITE),
    (ArtifactFile.DECODER_HEADER, ArtifactFile.DECODER_HEADER),
    (ArtifactFile.DECODER_KERAS, ArtifactFile.DECODER_KERAS),
    (ArtifactFile.CODEBOOK_NPZ, ArtifactFile.CODEBOOK_NPZ),
    (ArtifactFile.CODEBOOK_HEADER, ArtifactFile.CODEBOOK_HEADER),
    (ArtifactFile.CODEC_SPEC, ArtifactFile.CODEC_SPEC),
    (ArtifactFile.SAMPLE_DATA, ArtifactFile.SAMPLE_STIMULUS),
    (ArtifactFile.SAMPLE_STIMULUS, ArtifactFile.SAMPLE_STIMULUS),
    (ArtifactFile.DEMO_RECORDINGS, ArtifactFile.DEMO_RECORDINGS),
    (ArtifactFile.DEMO_RECORDINGS_MANIFEST, ArtifactFile.DEMO_RECORDINGS_MANIFEST),
    (ArtifactFile.QUANTIZATION_REPORT, ArtifactFile.QUANTIZATION_REPORT),
    (ArtifactFile.DEPLOY_MANIFEST, ArtifactFile.HF_CONFIG),
    (ArtifactFile.MODEL_CARD, ArtifactFile.MODEL_CARD),
    (ArtifactFile.PRIOR_INT8_TFLITE, ArtifactFile.PRIOR_INT8_TFLITE),
    (ArtifactFile.PRIOR_INT8_HEADER, ArtifactFile.PRIOR_INT8_HEADER),
    (ArtifactFile.PRIOR_MANIFEST, ArtifactFile.PRIOR_MANIFEST),
)

SPIHT_HF_FILE_RENAMES: tuple[tuple[ArtifactFile, ArtifactFile], ...] = (
    (ArtifactFile.DEPLOY_MANIFEST, ArtifactFile.HF_CONFIG),
    (ArtifactFile.SPIHT_CONFIG, ArtifactFile.SPIHT_CONFIG),
    (ArtifactFile.CODEC_SPEC, ArtifactFile.CODEC_SPEC),
    (ArtifactFile.SAMPLE_STIMULUS, ArtifactFile.SAMPLE_STIMULUS),
    (ArtifactFile.REFERENCE_VECTORS, ArtifactFile.REFERENCE_VECTORS),
    (ArtifactFile.MODEL_CARD, ArtifactFile.MODEL_CARD),
    (ArtifactFile.SCORECARD, ArtifactFile.SCORECARD),
    (ArtifactFile.SPIHT_APP_CONFIG_HEADER, ArtifactFile.SPIHT_APP_CONFIG_HEADER),
    (ArtifactFile.CHECKSUMS, ArtifactFile.CHECKSUMS),
)

HYBRID_HF_FILE_RENAMES: tuple[tuple[ArtifactFile, ArtifactFile], ...] = (
    *SPIHT_HF_FILE_RENAMES,
    (ArtifactFile.DENOISER_GAIN_MODEL, ArtifactFile.DENOISER_GAIN_MODEL),
    (ArtifactFile.DENOISER_TFLITE, ArtifactFile.DENOISER_TFLITE),
    (ArtifactFile.DENOISER_HEADER, ArtifactFile.DENOISER_HEADER),
    (ArtifactFile.DENOISER_TRAIN_CONFIG, ArtifactFile.DENOISER_TRAIN_CONFIG),
    (ArtifactFile.HYBRID_MANIFEST, ArtifactFile.HYBRID_MANIFEST),
    (ArtifactFile.SPIHT_APP_CONFIG_HEADER, ArtifactFile.SPIHT_APP_CONFIG_HEADER),
    (ArtifactFile.CHECKSUMS, ArtifactFile.CHECKSUMS),
)
