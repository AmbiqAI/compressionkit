"""Repackaging must honor the saved training layout rather than registry defaults."""

from __future__ import annotations

import json
from types import SimpleNamespace

import keras
import numpy as np
import pytest

from compressionkit.experiments import repackage
from compressionkit.export.codebook import extract_codebooks
from compressionkit.layers.ema_residual_vector_quantizer import EmaResidualVectorQuantizer


@pytest.mark.parametrize("bad_config", [False, True])
def test_repackage_restores_saved_ema_weights(tmp_path, monkeypatch, bad_config):
    vq = EmaResidualVectorQuantizer(num_levels=4, num_embeddings=8, embedding_dim=3, kmeans_init=True)
    vq.build((None, 1, 4, 3))
    for i, cb in enumerate(vq._codebooks):
        cb.assign(np.full((8, 3), (i + 1) / 10, np.float32))
    np.savez(tmp_path / "rvq_weights.npz", *vq.get_weights())
    config = {
        "model": {
            "num_levels": 2 if bad_config else 4,
            "latent_width": 8,
            "embedding_dim": 3,
            "use_ema": True,
            "kmeans_init": True,
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name in ("encoder.keras", "decoder.keras"):
        (tmp_path / name).touch()
    monkeypatch.setattr(keras.models, "load_model", lambda _: SimpleNamespace(input_shape=(None, 1, 16, 1)))
    frames = np.ones((2, 1, 16, 1), np.float32)
    monkeypatch.setattr(
        repackage,
        "collect_release_quantization_frames",
        lambda *args: (frames, frames + 1, {"normalization": {"epsilon": 0.001}}),
    )
    monkeypatch.setattr(repackage, "finalize_release_metadata", lambda *args: None)
    from compressionkit.export import deploy, quantization, stimulus

    monkeypatch.setattr(stimulus, "export_stimulus_npz", lambda **kwargs: None)
    monkeypatch.setattr(stimulus, "generate_normalized_stimulus", lambda **kwargs: frames.copy())
    monkeypatch.setattr(quantization, "refresh_rvq_encoder_precision_reports", lambda *args: None)
    calls = []

    def export(**kwargs):
        calls.append(kwargs)
        books = extract_codebooks(
            kwargs["rvq_weights"],
            num_levels=kwargs["rvq_num_levels"],
            use_ema=kwargs["rvq_use_ema"],
            kmeans_init=kwargs["rvq_kmeans_init"],
        )
        for cb, expected in zip(books, vq._codebooks):
            np.testing.assert_array_equal(cb, keras.ops.convert_to_numpy(expected))
        output = kwargs["output_dir"]
        output.mkdir()
        (output / "quantization_report.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "metrics": {"reconstruction_prd_percent_p90": 0.0, "encoder_input_saturation_fraction_max": 0.0},
                }
            )
        )
        return SimpleNamespace(as_dict=lambda: {})

    monkeypatch.setattr(deploy, "export_for_deployment", export)
    experiment = SimpleNamespace(
        method="rvq", experiment_id="test", run_name="test", modality="ecg", sample_rate=256, compression_ratio=8
    )
    if bad_config:
        with pytest.raises(ValueError, match=r"Expected .* RVQ weight arrays"):
            repackage.repackage_rvq_golden(experiment, run_dir=tmp_path)
        assert calls == []
    else:
        repackage.repackage_rvq_golden(experiment, run_dir=tmp_path)
        assert len(calls) == 1
