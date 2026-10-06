"""End-to-end smoke test for RVQ deploy export and validation."""

from __future__ import annotations

import json

import keras
import numpy as np
import pytest

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


def _build_smoke_rvq_deploy(
    tmp_path, *, use_ema=False, export_decoder_float32=True, export_decoder_int8=True, quantization="FP32"
):
    encoder = _tiny_encoder()
    decoder = _tiny_decoder()

    rep_dataset = np.random.randn(4, 1, FRAME, 1).astype(np.float32)
    sample_inputs = rep_dataset[:2]
    sample_targets = sample_inputs.copy()
    sample_reconstructions = decoder.predict(encoder.predict(sample_inputs, verbose=0), verbose=0)
    rvq_weights = [np.random.randn(NUM_EMBEDDINGS, EMBED_DIM).astype(np.float32)]
    if use_ema:
        from compressionkit.layers.ema_residual_vector_quantizer import EmaResidualVectorQuantizer

        vq = EmaResidualVectorQuantizer(num_levels=2, num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBED_DIM)
        vq.build(encoder.output_shape)
        for cb, count, total in zip(vq._codebooks, vq._ema_counts, vq._ema_weights):
            cb.assign(np.random.randn(NUM_EMBEDDINGS, EMBED_DIM).astype(np.float32))
            count.assign(np.full(NUM_EMBEDDINGS, 10.0, np.float32))
            total.assign(np.full((NUM_EMBEDDINGS, EMBED_DIM), 50.0, np.float32))
        rvq_weights = vq.get_weights()

    deploy_dir = tmp_path / "deploy"
    artifacts = export_for_deployment(
        encoder,
        decoder,
        rvq_weights,
        rvq_num_levels=2 if use_ema else 1,
        rvq_use_ema=use_ema,
        rep_dataset=rep_dataset,
        output_dir=deploy_dir,
        sample_inputs=sample_inputs,
        sample_targets=sample_targets,
        sample_reconstructions=sample_reconstructions,
        quantization=quantization,
        io_type="int8" if quantization == "INT8" else "float32",
        quantization_validation_dataset=np.random.default_rng(75)
        .uniform(-0.1, 0.1, rep_dataset.shape)
        .astype(np.float32),
        export_decoder_float32=export_decoder_float32,
        export_decoder_int8=export_decoder_int8,
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


@pytest.mark.parametrize("use_ema", [False, True])
def test_export_for_deployment_smoke_validates_strict_release(tmp_path, use_ema) -> None:
    deploy_dir, artifacts = _build_smoke_rvq_deploy(tmp_path, use_ema=use_ema)

    assert artifacts.manifest.exists()
    assert artifacts.codec_spec.exists()
    assert artifacts.encoder_float32_tflite.exists()
    assert artifacts.encoder_fp16_tflite.exists()
    assert artifacts.encoder_fp16_header.exists()
    assert artifacts.encoder_int16x8_tflite.exists()
    assert artifacts.encoder_int16x8_header.exists()
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

    manifest = json.loads(artifacts.manifest.read_text())
    assert manifest["codebook"]["num_levels"] == (2 if use_ema else 1)
    with np.load(artifacts.reference_vectors) as refs:
        assert refs["source_indices"].shape[-1] == (2 if use_ema else 1)
        with np.load(artifacts.sample_data_npz) as samples:
            np.testing.assert_array_equal(samples["reconstructions"], refs["source_reconstructions"])
    assert manifest["encoder"]["default_variant"] == "float32"
    assert manifest["encoder"]["default_tflite"] == "encoder_float32.tflite"
    assert manifest["encoder"]["float32_tflite"] == "encoder_float32.tflite"
    assert manifest["encoder"]["fp16_tflite"] == "encoder_fp16.tflite"
    assert manifest["encoder"]["int16x8_tflite"] == "encoder_int16x8.tflite"

    from compressionkit.runtime._litert import Interpreter

    interpreter = Interpreter(model_path=str(artifacts.encoder_float32_tflite))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    assert input_detail["dtype"] == np.float32
    assert output_detail["dtype"] == np.float32
    sample = np.load(deploy_dir / "sample_data.npz")["inputs"][:1]
    interpreter.set_tensor(input_detail["index"], sample)
    interpreter.invoke()
    assert interpreter.get_tensor(output_detail["index"]).dtype == np.float32

    if use_ema:
        from compressionkit.export.release import write_checksums

        with np.load(artifacts.reference_vectors) as refs:
            payload = {key: refs[key] for key in refs.files}
        source_latents = payload["source_latents"].copy()
        payload["source_latents"] = source_latents + 0.01
        np.savez_compressed(artifacts.reference_vectors, **payload)
        write_checksums(deploy_dir)
        result = validate_deploy_package(deploy_dir, strict_release=True, max_vectors=1)
        assert any("trained-encoder latent mismatch" in error for error in result.errors)
        payload["source_latents"] = source_latents
        # The deployed references still agree with themselves. An independent
        # training token mismatch must nevertheless reject the package.
        payload["source_indices"] = (payload["source_indices"] + 1) % NUM_EMBEDDINGS
        np.savez_compressed(artifacts.reference_vectors, **payload)
        write_checksums(deploy_dir)
        result = validate_deploy_package(deploy_dir, strict_release=True, max_vectors=1)
        assert any("trained-quantizer index mismatch" in error for error in result.errors)
        for key in list(payload):
            if key.startswith("source_"):
                payload.pop(key)
        np.savez_compressed(artifacts.reference_vectors, **payload)
        write_checksums(deploy_dir)
        result = validate_deploy_package(deploy_dir, strict_release=True, max_vectors=1)
        assert any("lack independent trained-quantizer references" in error for error in result.errors)


def test_validate_deploy_cli_smoke(tmp_path, capsys) -> None:
    deploy_dir, _artifacts = _build_smoke_rvq_deploy(tmp_path)

    rc = golden_cli.main(["validate-deploy", str(deploy_dir), "--strict-release", "--max-vectors", "1"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "family: rvq" in out
    assert "validation: ok" in out
    assert "scorecard.json" in out


def test_int8_only_decoder_preserves_independent_float_parity(tmp_path) -> None:
    from compressionkit.export.release import write_checksums

    deploy_dir, artifacts = _build_smoke_rvq_deploy(tmp_path, export_decoder_float32=False, quantization="INT8")
    assert not (deploy_dir / "decoder_float32.tflite").exists()
    assert artifacts.decoder_int8_tflite.exists()
    result = validate_deploy_package(deploy_dir, strict_release=True, max_vectors=2)
    assert result.ok, result.errors
    assert result.warnings == []

    # Float training references and INT8 deployed references both remain
    # checked, rather than weakening the float tolerance for quantization.
    with np.load(artifacts.reference_vectors) as refs:
        payload = {key: refs[key] for key in refs.files}
    original = payload["source_reconstructions"].copy()
    payload["source_reconstructions"] += 0.1
    np.savez_compressed(artifacts.reference_vectors, **payload)
    write_checksums(deploy_dir)
    result = validate_deploy_package(deploy_dir, strict_release=True, max_vectors=2)
    assert any("trained-decoder reconstruction mismatch" in error for error in result.errors)

    payload["source_reconstructions"] = original
    payload["reconstructions"] += 0.1
    np.savez_compressed(artifacts.reference_vectors, **payload)
    write_checksums(deploy_dir)
    result = validate_deploy_package(deploy_dir, strict_release=True, max_vectors=2)
    assert any("RVQ reference reconstruction mismatch" in error for error in result.errors)


def test_int8_encoder_export_supports_keras_only_decoder(tmp_path) -> None:
    from pathlib import Path

    deploy_dir, artifacts = _build_smoke_rvq_deploy(
        tmp_path, quantization="INT8", export_decoder_float32=False, export_decoder_int8=False
    )
    assert artifacts.decoder_keras.exists()
    assert not (deploy_dir / "decoder.tflite").exists()
    assert not (deploy_dir / "decoder_float32.tflite").exists()
    assert artifacts.reference_vectors == Path()
    report = json.loads(artifacts.quantization_report.read_text())
    assert report["metrics"]["frames_checked"] == 4
    assert report["passed"]
