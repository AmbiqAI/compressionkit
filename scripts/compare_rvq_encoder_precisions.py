"""Compare RVQ encoder precision variants on disjoint real validation data.

This research script never alters a golden deploy directory. For every selected
RVQ golden, it exports temporary INT8, FP16, and INT16x8 encoders using the
same 4,096-frame calibration partition, then compares each with the float32
encoder through the shared FP32 RVQ codebook and decoder on a separate
2,048-frame holdout.

Example:
    scripts/devcontainer.sh exec -- uv run python scripts/compare_rvq_encoder_precisions.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import keras

from compressionkit.experiments.registry import GoldenExperiment, list_goldens
from compressionkit.experiments.repackage import collect_release_quantization_frames
from compressionkit.export.quantization import evaluate_rvq_encoder_quantization
from compressionkit.export.tflite import export_encoder_tflite
from compressionkit.runtime._litert import Interpreter

_PRECISIONS = ("INT8", "FP16", "INT16X8")


def _experiment_output(
    experiment: GoldenExperiment,
    *,
    results_root: Path,
    precisions: tuple[str, ...],
) -> dict[str, object]:
    """Export and score selected encoder precisions for one golden experiment."""
    run_dir = results_root / experiment.run_name
    deploy_dir = run_dir / "deploy"
    if not deploy_dir.is_dir():
        raise FileNotFoundError(f"Missing deploy directory: {deploy_dir}")
    calibration, holdout, contract = collect_release_quantization_frames(experiment, run_dir)
    encoder = keras.models.load_model(run_dir / "encoder.keras")

    variants: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix=f"{experiment.experiment_id}_precision_") as temp_root:
        temp_root_path = Path(temp_root)
        for precision in precisions:
            candidate_dir = temp_root_path / precision.lower()
            shutil.copytree(deploy_dir, candidate_dir)
            encoder_path, _ = export_encoder_tflite(
                encoder,
                rep_dataset=calibration,
                output_dir=candidate_dir,
                tflite_name="encoder.tflite",
                header_name="encoder.h",
                c_array_name="encoder",
                quantization=precision,
                io_type="float32" if precision != "INT8" else "int8",
            )
            interpreter = Interpreter(model_path=str(encoder_path))
            interpreter.allocate_tensors()
            report = evaluate_rvq_encoder_quantization(
                candidate_dir,
                holdout,
                max_frames=None,
            )
            variants[precision] = {
                "encoder_bytes": encoder_path.stat().st_size,
                "input_dtype": str(interpreter.get_input_details()[0]["dtype"]),
                "output_dtype": str(interpreter.get_output_details()[0]["dtype"]),
                "report": asdict(report),
            }

    return {
        "modality": experiment.modality,
        "sample_rate_hz": experiment.sample_rate,
        "compression_ratio": experiment.compression_ratio,
        "preprocessing_contract": contract,
        "variants": variants,
    }


def main() -> None:
    """Run precision exports and write the comparison JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("all", "ppg", "ecg"), default="all")
    parser.add_argument(
        "--experiment",
        action="append",
        help="Golden experiment ID to evaluate; repeat to select several.",
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rvq_encoder_precision_comparison.json"),
        help="JSON path for the comparison results.",
    )
    parser.add_argument(
        "--precision",
        action="append",
        choices=_PRECISIONS,
        help="Precision to test; repeat to select a subset (default: all).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge selected variants into an existing output report.",
    )
    args = parser.parse_args()

    precisions = tuple(args.precision or _PRECISIONS)
    goldens = [
        golden
        for golden in list_goldens(method="rvq")
        if golden.structure == "codec"
        and (args.modality == "all" or golden.modality == args.modality)
        and (not args.experiment or golden.experiment_id in args.experiment)
    ]
    if args.experiment:
        unknown_ids = set(args.experiment) - {golden.experiment_id for golden in goldens}
        if unknown_ids:
            parser.error(f"Unknown or non-RVQ golden IDs: {', '.join(sorted(unknown_ids))}")
    results: dict[str, object] = {
        "format_version": 1,
        "precisions": list(precisions),
        "experiments": {},
    }
    if args.merge and args.output.is_file():
        existing = json.loads(args.output.read_text())
        existing_experiments = existing.get("experiments")
        if not isinstance(existing_experiments, dict):
            parser.error(f"Existing report has invalid experiments field: {args.output}")
        existing_precisions = existing.get("precisions", [])
        if not isinstance(existing_precisions, list):
            parser.error(f"Existing report has invalid precisions field: {args.output}")
        results = existing
        results["precisions"] = list(dict.fromkeys([*existing_precisions, *precisions]))
    for golden in goldens:
        print(f"Evaluating {golden.experiment_id}...", flush=True)
        current = _experiment_output(
            golden,
            results_root=args.results_root,
            precisions=precisions,
        )
        if args.merge and golden.experiment_id in results["experiments"]:
            existing = results["experiments"][golden.experiment_id]
            if not isinstance(existing, dict):
                parser.error(f"Existing report has invalid experiment entry: {golden.experiment_id}")
            existing_variants = existing.get("variants")
            current_variants = current["variants"]
            if not isinstance(existing_variants, dict) or not isinstance(current_variants, dict):
                parser.error(f"Existing report has invalid variants: {golden.experiment_id}")
            current["variants"] = {**existing_variants, **current_variants}
        results["experiments"][golden.experiment_id] = current

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
