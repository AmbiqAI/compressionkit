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

    from compressionkit.export.deploy import export_for_deployment
    from compressionkit.export.stimulus import export_stimulus_npz, generate_stimulus

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
    rvq_npz = np.load(rvq_weights_path)
    rvq_weights = [rvq_npz[k] for k in sorted(rvq_npz.files)]

    input_shape = encoder.input_shape
    if len(input_shape) in (3, 4):
        frame_size = input_shape[-2]
    else:
        frame_size = input_shape[-1]

    rep_dataset = generate_stimulus(
        modality=experiment.modality,
        num_samples=100,
        frame_size=frame_size,
        sample_rate=experiment.sample_rate,
        seed=42,
    )
    if len(input_shape) == 4:
        rep_dataset = rep_dataset.reshape(-1, 1, frame_size, 1)
    elif len(input_shape) == 3:
        rep_dataset = rep_dataset[..., np.newaxis]

    model_card_info: dict[str, object] = {
        "experiment_id": experiment.experiment_id,
        "run_name": experiment.run_name,
        "modality": experiment.modality,
        "sample_rate": experiment.sample_rate,
        "compression_ratio": experiment.compression_ratio,
        "license": "other",
    }
    if scorecard_payload is not None:
        model_card_info["scorecard_summary"] = scorecard_payload

    sample_inputs = rep_dataset[:10]
    latents = encoder.predict(sample_inputs, verbose=0)
    sample_reconstructions = decoder.predict(latents, verbose=0)
    if isinstance(sample_reconstructions, dict):
        sample_reconstructions = sample_reconstructions.get("reconstruction", sample_reconstructions.get("output"))

    artifacts = export_for_deployment(
        encoder=encoder,
        decoder=decoder,
        rvq_weights=rvq_weights,
        rep_dataset=rep_dataset,
        output_dir=output_dir,
        sample_inputs=sample_inputs,
        sample_targets=sample_inputs,
        sample_reconstructions=np.asarray(sample_reconstructions, dtype=np.float32),
        model_name=experiment.run_name,
        export_decoder_float32=True,
        export_decoder_int8=export_decoder_int8,
        model_card_info=model_card_info,
    )
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
    }


__all__ = [
    "finalize_release_metadata",
    "load_scorecard_payload",
    "repackage_rvq_golden",
    "resolve_scorecard_path",
]
