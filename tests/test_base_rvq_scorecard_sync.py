"""Tests for RVQ deploy scorecard wiring in BaseRVQTrainer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compressionkit.export.deploy import DeploymentArtifacts
from compressionkit.recipes import base_rvq


class _DummyModel:
    def __init__(self) -> None:
        self.encoder = object()
        self.decoder = object()
        self.vq = SimpleNamespace(get_weights=lambda: [np.zeros((1, 1), dtype=np.float32)])

    def fit(self, *args, **kwargs):
        return SimpleNamespace(history={"loss": [1.0], "val_loss": [0.5], "val_mse": [0.4]})


class _DummyTrainer(base_rvq.BaseRVQTrainer[SimpleNamespace]):
    @property
    def logger(self):
        return SimpleNamespace(
            info=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        )

    def build_preprocessor(self):
        return object()

    def build_augmenter(self):
        return object()

    def build_datasets(self, preprocessor, augmenter):
        return object(), object(), 1, {"mode": "test"}

    def build_model(self):
        return _DummyModel()

    def compile_model(self, model, *, learning_rate):
        return None

    def run_evaluation(self, *, model, val_ds, run_dir, validation_steps):
        return {
            "samples": {},
            "sample_inputs": np.zeros((1, 4, 1), dtype=np.float32),
            "sample_targets": np.zeros((1, 4, 1), dtype=np.float32),
            "sample_reconstructions": np.zeros((1, 4, 1), dtype=np.float32),
        }

    def build_compression_stats(self):
        return {"compression_ratio": 4}

    def assemble_summary(self, **kwargs):
        return {"artifacts": {"deploy": kwargs["deploy_artifacts"]}}


def test_base_rvq_trainer_syncs_built_scorecard_into_deploy(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "ppg_rvq_64hz_04x_golden"
    deploy_dir = run_dir / "deploy"
    deploy_dir.mkdir(parents=True)

    cfg = SimpleNamespace(
        run_name="ppg_rvq_64hz_04x_golden",
        model_dump=lambda: {"run_name": "ppg_rvq_64hz_04x_golden"},
        data=SimpleNamespace(sampling_rate=64, steps_per_epoch=1, epochs=1),
        model=SimpleNamespace(kmeans_init=False),
        evaluation=SimpleNamespace(
            int8_calibration_frames=1,
            int8_validation_frames=1,
            int8_sampling_pool_frames=2,
        ),
        training=SimpleNamespace(selection_metric="val_loss"),
        output=SimpleNamespace(
            results_root=tmp_path, log_file="train.log", wandb=SimpleNamespace(artifact_summary_only=False)
        ),
    )
    trainer = _DummyTrainer(cfg)

    written_summaries: list[dict] = []
    synced_payloads: list[dict] = []

    monkeypatch.setattr(base_rvq, "setup_run_dir", lambda results_root, run_name: run_dir)
    monkeypatch.setattr(base_rvq, "setup_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(base_rvq, "save_config_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(base_rvq, "init_wandb_run", lambda **kwargs: None)
    monkeypatch.setattr(base_rvq, "build_learning_rate", lambda cfg, steps_per_epoch: 1.0e-3)
    monkeypatch.setattr(base_rvq, "build_callbacks", lambda *args, **kwargs: [])
    monkeypatch.setattr(base_rvq, "reload_best_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(base_rvq, "save_model_artifacts", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        base_rvq,
        "collect_disjoint_quantization_datasets",
        lambda *args, **kwargs: (np.zeros((1, 4, 1), dtype=np.float32), np.ones((1, 4, 1), dtype=np.float32)),
    )
    monkeypatch.setattr(
        base_rvq,
        "export_for_deployment",
        lambda *args, **kwargs: DeploymentArtifacts(output_dir=deploy_dir, checksums=deploy_dir / "checksums.json"),
    )
    monkeypatch.setattr(
        base_rvq,
        "extract_history_metrics",
        lambda history, selection_metric: (
            0,
            {selection_metric: 0.5},
            {"final_loss": 1.0, "final_val_loss": 0.5, "final_val_mse": 0.4},
            selection_metric,
        ),
    )
    monkeypatch.setattr(
        base_rvq,
        "write_summary",
        lambda summary, run_dir: written_summaries.append(summary.copy()) or run_dir / "summary.json",
    )
    monkeypatch.setattr(
        base_rvq, "write_long_recording_eval", lambda payload, run_dir: run_dir / "long_recording_eval.json"
    )
    monkeypatch.setattr(base_rvq, "finalize_wandb_run", lambda **kwargs: None)

    scorecard = {"time_domain": {"prd_percent": {"mean": 2.5}}}

    def _write_quality_scorecard(run_dir_arg, *, modality, sample_rate, **kwargs):
        path = run_dir_arg / "quality_scorecard.json"
        path.write_text(json.dumps(scorecard))
        return path

    monkeypatch.setattr(base_rvq, "write_quality_scorecard", _write_quality_scorecard)
    monkeypatch.setattr(
        base_rvq,
        "sync_scorecard_to_deploy",
        lambda output_dir, scorecard_summary: (
            synced_payloads.append(scorecard_summary) or (Path(output_dir) / "scorecard.json")
        ),
    )

    summary = trainer.train()

    assert synced_payloads == [scorecard]
    assert len(written_summaries) == 2
    assert written_summaries[-1]["artifacts"]["deploy"]["scorecard"] == "scorecard.json"
    assert summary["artifacts"]["deploy"]["scorecard"] == "scorecard.json"
