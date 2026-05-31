"""End-to-end smoke test for RVQ deploy export and validation."""

from __future__ import annotations

import keras
import numpy as np

from compressionkit.experiments import cli as golden_cli
from compressionkit.export import export_for_deployment, validate_deploy_package

FRAME = 16
LATENT_TOKENS = 4
EMBED_DIM = 8
NUM_EMBEDDINGS = 16


def _tiny_encoder() -> keras.Model:
    inputs = keras.Input(shape=(1, FRAME, 1))
    outputs = keras.layers.Conv2D(
        EMBED_DIM,
        kernel_size=(1, 4),
        strides=(1, 4),
        padding="valid",
        activation="relu",
    )(inputs)
    return keras.Model(inputs, outputs, name="smoke_encoder")


def _tiny_decoder() -> keras.Model:
    inputs = keras.Input(shape=(1, LATENT_TOKENS, EMBED_DIM))
    outputs = keras.layers.Conv2DTranspose(
        1,
        kernel_size=(1, 4),
        strides=(1, 4),
        padding="valid",
    )(inputs)
    return keras.Model(inputs, outputs, name="smoke_decoder")


def _build_smoke_rvq_deploy(tmp_path):
    encoder = _tiny_encoder()
    decoder = _tiny_decoder()

    rep_dataset = np.random.randn(4, 1, FRAME, 1).astype(np.float32)
    sample_inputs = rep_dataset[:2]
    sample_targets = sample_inputs.copy()
    sample_reconstructions = decoder.predict(encoder.predict(sample_inputs, verbose=0), verbose=0)
    rvq_weights = [np.random.randn(NUM_EMBEDDINGS, EMBED_DIM).astype(np.float32)]

    deploy_dir = tmp_path / "deploy"
    artifacts = export_for_deployment(
        encoder,
        decoder,
        rvq_weights,
        rep_dataset=rep_dataset,
        output_dir=deploy_dir,
        sample_inputs=sample_inputs,
        sample_targets=sample_targets,
        sample_reconstructions=sample_reconstructions,
        quantization="FP32",
        io_type="float32",
        model_name="ppg_rvq_smoke",
        model_card_info={
            "run_name": "ppg_rvq_64hz_04x_golden",
            "modality": "ppg",
            "sample_rate": 64,
            "compression_ratio": 4,
            "license": "other",
        },
        scorecard_summary={
            "time_domain": {"prd_percent": {"mean": 2.5, "n": 2}},
            "spectral": {"band_total_rel_error": {"mean": 0.1, "n": 2}},
        },
    )
    return deploy_dir, artifacts


def test_export_for_deployment_smoke_validates_strict_release(tmp_path) -> None:
    deploy_dir, artifacts = _build_smoke_rvq_deploy(tmp_path)

    assert artifacts.manifest.exists()
    assert artifacts.codec_spec.exists()
    assert artifacts.scorecard.exists()
    assert artifacts.reference_vectors.exists()
    assert artifacts.readme.exists()

    result = validate_deploy_package(
        deploy_dir,
        strict_release=True,
        max_vectors=1,
    )

    assert result.ok, result.errors
    assert result.warnings == []
    assert "scorecard.json" in result.checked_files
    assert "reference_vectors.npz" in result.checked_files


def test_validate_deploy_cli_smoke(tmp_path, capsys) -> None:
    deploy_dir, _artifacts = _build_smoke_rvq_deploy(tmp_path)

    rc = golden_cli.main(["validate-deploy", str(deploy_dir), "--strict-release", "--max-vectors", "1"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "family: rvq" in out
    assert "validation: ok" in out
    assert "scorecard.json" in out
