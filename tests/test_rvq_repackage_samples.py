"""Real calibration frames must never become redistributed release samples."""

from __future__ import annotations

import json
from types import SimpleNamespace

import keras
import numpy as np

from compressionkit.experiments import repackage
from compressionkit.export import deploy, quantization, stimulus
from compressionkit.export.family_registry import get_family_spec
from scripts.publish_to_huggingface import _stage_deploy_files


def test_repackage_publishes_synthetic_samples_with_real_calibration(tmp_path, monkeypatch):
    for name in ("encoder.keras", "decoder.keras"):
        (tmp_path / name).touch()
    np.savez(tmp_path / "rvq_weights.npz", np.zeros((4, 2), np.float32))
    (tmp_path / "config.json").write_text(
        json.dumps({"model": {"num_levels": 1, "use_ema": False, "latent_width": 4, "embedding_dim": 2}})
    )
    model = SimpleNamespace(input_shape=(None, 1, 16, 1), predict=lambda x, **kw: x)
    monkeypatch.setattr(keras.models, "load_model", lambda _: model)
    calibration = np.full((4, 1, 16, 1), 123.0, np.float32)
    holdout = calibration + 1
    contract = {"normalization": {"epsilon": 0.001}}
    monkeypatch.setattr(
        repackage, "collect_release_quantization_frames", lambda *args: (calibration, holdout, contract)
    )
    monkeypatch.setattr(repackage, "finalize_release_metadata", lambda *args: None)
    monkeypatch.setattr(quantization, "refresh_rvq_encoder_precision_reports", lambda *args: None, raising=False)
    monkeypatch.setattr(
        stimulus,
        "generate_stimulus",
        lambda **kw: np.tile(np.arange(kw["frame_size"], dtype=np.float32), (kw["num_samples"], 1)),
    )
    calls = []

    def export(**kw):
        calls.append(kw)
        output = kw["output_dir"]
        output.mkdir()
        np.savez(output / "sample_data.npz", inputs=kw["sample_inputs"], targets=kw["sample_targets"])
        (output / "quantization_report.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "metrics": {"reconstruction_prd_percent_p90": 0, "encoder_input_saturation_fraction_max": 0},
                }
            )
        )
        return SimpleNamespace(as_dict=lambda: {})

    monkeypatch.setattr(deploy, "export_for_deployment", export)
    experiment = SimpleNamespace(
        method="rvq",
        experiment_id="test",
        run_name="test",
        modality="ppg",
        sample_rate=64,
        compression_ratio=8,
        hf_version="v1.1",
    )
    repackage.repackage_rvq_golden(experiment, run_dir=tmp_path, num_stimulus=2)
    assert calls[0]["rep_dataset"] is calibration
    assert calls[0]["quantization_validation_dataset"] is holdout
    assert calls[0]["model_version"] == "1.1"
    samples = calls[0]["sample_inputs"]
    assert samples.shape == (2, 1, 16, 1)
    assert np.allclose(samples.mean(axis=(1, 2, 3)), 0, atol=1e-6)
    assert not np.any(np.isin(samples, [123.0, 124.0]))
    staging = tmp_path / "staging"
    staging.mkdir()
    _stage_deploy_files(get_family_spec("rvq"), tmp_path / "deploy", staging, None)
    with np.load(staging / "sample_data.npz") as published:
        np.testing.assert_array_equal(published["inputs"], samples)
        assert not np.any(np.isin(published["inputs"], [123.0, 124.0]))
    with np.load(staging / "sample_stimulus.npz") as published:
        assert not np.any(np.isin(published["stimulus"], [123.0, 124.0]))
