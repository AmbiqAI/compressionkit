"""Tests for shared deploy/HuggingFace artifact names."""

from __future__ import annotations

from compressionkit.export.artifact_contract import (
    HYBRID_HF_FILE_RENAMES,
    RVQ_HF_FILE_RENAMES,
    SPIHT_HF_FILE_RENAMES,
    ArtifactFile,
    SampleArray,
)


def test_rvq_hf_contract_covers_manifest_and_sample_aliases() -> None:
    renames = dict(RVQ_HF_FILE_RENAMES)

    assert renames[ArtifactFile.DEPLOY_MANIFEST] == ArtifactFile.HF_CONFIG
    assert renames[ArtifactFile.SAMPLE_DATA] == ArtifactFile.SAMPLE_STIMULUS
    assert renames[ArtifactFile.ENCODER_TFLITE] == ArtifactFile.ENCODER_INT8_HF_TFLITE
    assert renames[ArtifactFile.DECODER_TFLITE] == ArtifactFile.DECODER_INT8_HF_TFLITE


def test_spiht_and_hybrid_hf_contracts_share_manifest_alias() -> None:
    spiht_renames = dict(SPIHT_HF_FILE_RENAMES)
    hybrid_renames = dict(HYBRID_HF_FILE_RENAMES)

    assert spiht_renames[ArtifactFile.DEPLOY_MANIFEST] == ArtifactFile.HF_CONFIG
    assert hybrid_renames[ArtifactFile.DEPLOY_MANIFEST] == ArtifactFile.HF_CONFIG
    assert ArtifactFile.DENOISER_GAIN_MODEL in hybrid_renames
    assert ArtifactFile.HYBRID_MANIFEST in hybrid_renames


def test_published_sample_npz_uses_inputs_key() -> None:
    assert SampleArray.INPUTS == "inputs"
    assert SampleArray.STIMULUS == "stimulus"
    assert SampleArray.INPUTS != SampleArray.STIMULUS
