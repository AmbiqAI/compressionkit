"""Helpers for backfilling release-complete golden deploy packages."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from compressionkit.experiments.registry import GoldenExperiment


def load_scorecard_payload(scorecard_path: Path | None) -> dict[str, object] | None:
    """Load a frozen scorecard JSON payload when present."""
    if scorecard_path is None or not scorecard_path.is_file():
        return None
    with scorecard_path.open() as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object in {scorecard_path}, got {type(payload).__name__}")
    return dict(payload)


def resolve_scorecard_path(run_dir: Path, explicit_scorecard: Path | None) -> Path | None:
    """Return the explicit scorecard path or the run's default frozen scorecard."""
    if explicit_scorecard is not None:
        return explicit_scorecard
    default_path = run_dir / "quality_scorecard.json"
    return default_path if default_path.is_file() else None


def finalize_release_metadata(output_dir: Path, scorecard_payload: dict[str, object] | None) -> None:
    """Refresh deploy metadata after backfilling artifacts."""
    from compressionkit.export.deploy import sync_scorecard_to_deploy
    from compressionkit.export.release import write_checksums

    if scorecard_payload is not None:
        sync_scorecard_to_deploy(output_dir, scorecard_payload)
        return
    write_checksums(output_dir)


def collect_release_quantization_frames(
    experiment: GoldenExperiment,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Rebuild disjoint real, training-preprocessed INT8 frame partitions.

    The release repackage path must not calibrate a model trained on normalized
    physiological windows with raw synthetic waveforms. This helper reloads the
    golden's saved Pydantic config and delegates to its normal dataset builder,
    so resampling, framing, and normalization remain identical to training.
    """
    from compressionkit.trainers.common import collect_disjoint_quantization_datasets

    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing saved golden config: {config_path}")
    config_payload = config_path.read_text()

    if experiment.modality == "ppg":
        from compressionkit.configs.ppg_rvq import PpgRvqConfig
        from compressionkit.preprocessing.ppg import build_augmenter, build_preprocessor
        from compressionkit.trainers.ppg_rvq import build_datasets

        cfg = PpgRvqConfig.model_validate_json(config_payload)
        preprocessor = build_preprocessor(
            frame_size=cfg.data.frame_size,
            epsilon=cfg.data.epsilon,
            seed=cfg.data.shuffle_seed,
        )
        augmenter = build_augmenter(tuple(cfg.data.gaussian_noise), aug_cfg=cfg.data.augmentation)
        _, val_ds, _, _ = build_datasets(cfg, preprocessor, augmenter)
        sample_rate = cfg.data.sampling_rate
    elif experiment.modality == "ecg":
        from compressionkit.configs.ecg_rvq import EcgRvqConfig
        from compressionkit.preprocessing.ecg import build_augmenter, build_preprocessor
        from compressionkit.trainers.ecg_rvq import build_datasets

        cfg = EcgRvqConfig.model_validate_json(config_payload)
        preprocessor = build_preprocessor(
            frame_size=cfg.data.frame_size,
            epsilon=cfg.data.epsilon,
            seed=cfg.data.shuffle_seed,
        )
        augmenter = build_augmenter(
            aug_cfg=cfg.data.augmentation,
            sample_rate=cfg.data.effective_sample_rate,
            seed=cfg.data.shuffle_seed,
        )
        _, val_ds, _, _ = build_datasets(cfg, preprocessor, augmenter)
        sample_rate = cfg.data.effective_sample_rate
    else:
        raise ValueError(f"Unsupported RVQ modality {experiment.modality!r}")

    calibration_frames, validation_frames = collect_disjoint_quantization_datasets(
        val_ds,
        calibration_frames=cfg.evaluation.int8_calibration_frames,
        validation_frames=cfg.evaluation.int8_validation_frames,
        sampling_pool_frames=cfg.evaluation.int8_sampling_pool_frames,
        seed=cfg.data.shuffle_seed,
    )
    contract: dict[str, object] = {
        "format_version": 1,
        "sample_rate_hz": sample_rate,
        "frame_size": cfg.data.frame_size,
        "input_shape": [1, 1, cfg.data.frame_size, 1],
        "normalization": {
            "kind": "per_frame_layer_norm",
            "mean": "mean over all samples in each frame",
            "variance": "mean squared deviation over all samples in each frame",
            "epsilon": cfg.data.epsilon,
            "inverse_for_display": "raw = normalized * sqrt(variance + epsilon) + mean",
        },
        "int8_quantization": {
            "calibration_frames": int(calibration_frames.shape[0]),
            "validation_frames": int(validation_frames.shape[0]),
            "sampling_pool_frames": cfg.evaluation.int8_sampling_pool_frames,
            "sampling_method": "seeded reservoir sample; disjoint calibration and validation partitions",
        },
    }
    return calibration_frames, validation_frames, contract


def repackage_rvq_golden(
    experiment: GoldenExperiment,
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    num_stimulus: int = 10,
    export_decoder_int8: bool = True,
    scorecard_path: Path | None = None,
) -> dict[str, object]:
    """Re-export an existing RVQ golden run into a release-complete deploy package."""
    if experiment.method != "rvq":
        raise ValueError(f"repackage_rvq_golden only supports RVQ experiments, got {experiment.method!r}")

    import keras

    from compressionkit.export.codebook import extract_codebooks, load_rvq_weights
    from compressionkit.export.deploy import export_for_deployment
    from compressionkit.export.quantization import refresh_rvq_encoder_precision_reports
    from compressionkit.export.stimulus import export_stimulus_npz

    output_dir = output_dir or (run_dir / "deploy")
    resolved_scorecard_path = resolve_scorecard_path(run_dir, scorecard_path)
    scorecard_payload = load_scorecard_payload(resolved_scorecard_path)

    encoder_path = run_dir / "encoder.keras"
    decoder_path = run_dir / "decoder.keras"
    rvq_weights_path = run_dir / "rvq_weights.npz"
    missing = [p for p in [encoder_path, decoder_path, rvq_weights_path] if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing golden artifacts for {experiment.experiment_id!r}: {missing_str}")

    encoder = keras.models.load_model(encoder_path)
    decoder = keras.models.load_model(decoder_path)
    rvq_weights = load_rvq_weights(rvq_weights_path)
    # Use the saved run contract, not the current registry/config defaults.
    saved_model = json.loads((run_dir / "config.json").read_text())["model"]
    rvq_num_levels = saved_model["num_levels"]
    rvq_use_ema = saved_model.get("use_ema", False)
    rvq_kmeans_init = saved_model.get("kmeans_init", False)
    codebooks = extract_codebooks(
        rvq_weights, num_levels=rvq_num_levels, use_ema=rvq_use_ema, kmeans_init=rvq_kmeans_init
    )
    sizes = saved_model.get("codebook_sizes") or [saved_model["latent_width"]] * rvq_num_levels
    if [cb.shape for cb in codebooks] != [(size, saved_model["embedding_dim"]) for size in sizes]:
        raise ValueError("Saved RVQ codebook shapes do not match the training config")

    input_shape = encoder.input_shape
    if len(input_shape) in (3, 4):
        frame_size = input_shape[-2]
    else:
        frame_size = input_shape[-1]

    rep_dataset, quantization_validation_dataset, preprocessing_contract = collect_release_quantization_frames(
        experiment, run_dir
    )
    if tuple(rep_dataset.shape[1:]) != tuple(input_shape[1:]):
        raise ValueError(
            f"Calibration frame shape {rep_dataset.shape[1:]} does not match encoder input {input_shape[1:]}"
        )

    model_card_info: dict[str, object] = {
        "experiment_id": experiment.experiment_id,
        "run_name": experiment.run_name,
        "modality": experiment.modality,
        "sample_rate": experiment.sample_rate,
        "compression_ratio": experiment.compression_ratio,
        "license": "other",
        "preprocessing_contract": preprocessing_contract,
    }
    if scorecard_payload is not None:
        model_card_info["scorecard_summary"] = scorecard_payload

    sample_inputs = rep_dataset[:10]

    artifacts = export_for_deployment(
        encoder=encoder,
        decoder=decoder,
        rvq_weights=rvq_weights,
        rvq_num_levels=rvq_num_levels,
        rvq_use_ema=rvq_use_ema,
        rvq_kmeans_init=rvq_kmeans_init,
        rep_dataset=rep_dataset,
        quantization_validation_dataset=quantization_validation_dataset,
        output_dir=output_dir,
        sample_inputs=sample_inputs,
        sample_targets=sample_inputs,
        model_name=experiment.run_name,
        export_decoder_float32=True,
        export_decoder_int8=export_decoder_int8,
        model_card_info=model_card_info,
    )
    quantization_report = json.loads((output_dir / "quantization_report.json").read_text())
    report_metrics = quantization_report["metrics"]
    # Historical reports may have used different (including faulty) codebooks.
    refresh_rvq_encoder_precision_reports(output_dir, quantization_validation_dataset)
    export_stimulus_npz(
        modality=experiment.modality,
        output_path=output_dir / "sample_stimulus.npz",
        num_samples=num_stimulus,
        frame_size=frame_size,
        sample_rate=experiment.sample_rate,
    )
    finalize_release_metadata(output_dir, scorecard_payload)

    return {
        "experiment_id": experiment.experiment_id,
        "run_dir": str(run_dir),
        "deploy_dir": str(output_dir),
        "scorecard_path": str(resolved_scorecard_path) if resolved_scorecard_path is not None else None,
        "artifacts": artifacts.as_dict(),
        "quantization_report": {
            "passed": quantization_report["passed"],
            "reconstruction_prd_percent_p90": report_metrics["reconstruction_prd_percent_p90"],
            "encoder_input_saturation_fraction_max": report_metrics["encoder_input_saturation_fraction_max"],
        },
    }


__all__ = [
    "collect_release_quantization_frames",
    "finalize_release_metadata",
    "load_scorecard_payload",
    "repackage_rvq_golden",
    "resolve_scorecard_path",
]
