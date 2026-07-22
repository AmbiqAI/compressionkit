"""Attach real, quality-gated ECG and PPG recordings to RVQ golden deploys.

This is the manual release step for browser demo inputs. It creates the same
10-recording bundle for every compression rate of one modality, then updates
each package manifest and checksum list. It does not train or re-export models.

Examples:
    uv run python scripts/attach_rvq_demo_recordings.py --modality ppg
    uv run python scripts/attach_rvq_demo_recordings.py --modality ecg
    uv run python scripts/attach_rvq_demo_recordings.py --modality all --duration-seconds 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

from compressionkit.configs.paths import default_datasets_dir
from compressionkit.experiments.registry import list_goldens
from compressionkit.export.demo_ecg import generate_ecg_demo_clips
from compressionkit.export.demo_ppg import generate_ppg_demo_clips
from compressionkit.export.demo_recordings import attach_demo_recordings_to_deploy, export_demo_recordings
from compressionkit.export.validate import validate_deploy_package

_SOURCES = {
    "ecg": {
        "dataset": "MIT-BIH Arrhythmia Database v1.0.0",
        "license": "ODC-By-1.0",
        "license_url": "https://opendatacommons.org/licenses/by/1-0/",
        "url": "https://physionet.org/content/mitdb/1.0.0/",
        "citation": "Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol. 2001;20(3):45-50.",
    },
    "ppg": {
        "dataset": "BIDMC PPG and Respiration Dataset v1.0.0",
        "license": "ODC-By-1.0",
        "license_url": "https://opendatacommons.org/licenses/by/1-0/",
        "url": "https://physionet.org/content/bidmc/1.0.0/",
        "citation": "Pimentel MAF, et al. Toward a robust estimation of respiratory rate from pulse oximeters. IEEE Trans Biomed Eng. 2017;64(8):1914-1923.",
    },
}


def _build_clips(modality: str, dataset_root: Path, *, num_clips: int, duration_seconds: float, seed: int):
    """Generate the modality's shared demo selection at its golden sample rate."""
    if modality == "ecg":
        return generate_ecg_demo_clips(
            dataset_root / "mitdb",
            num_clips=num_clips,
            duration_seconds=duration_seconds,
            sample_rate=256,
            seed=seed,
        )
    return generate_ppg_demo_clips(
        dataset_root / "bidmc",
        num_clips=num_clips,
        duration_seconds=duration_seconds,
        sample_rate=64,
        seed=seed,
    )


def attach_modality(
    modality: str,
    *,
    results_root: Path,
    dataset_root: Path,
    num_clips: int,
    duration_seconds: float,
    seed: int,
    experiment_ids: set[str] | None = None,
) -> list[Path]:
    """Attach one shared real-recordings bundle to selected local RVQ deploys."""
    sample_rate = 256 if modality == "ecg" else 64
    clips = _build_clips(
        modality,
        dataset_root,
        num_clips=num_clips,
        duration_seconds=duration_seconds,
        seed=seed,
    )
    updated: list[Path] = []
    for golden in list_goldens(modality=modality, method="rvq"):
        if golden.structure != "codec":
            continue
        if experiment_ids is not None and golden.experiment_id not in experiment_ids:
            continue
        deploy_dir = results_root / golden.run_name / "deploy"
        if not deploy_dir.is_dir():
            raise FileNotFoundError(f"Missing local deploy package for {golden.experiment_id}: {deploy_dir}")
        bundle = export_demo_recordings(
            deploy_dir,
            clips=clips,
            modality=modality,
            sample_rate=sample_rate,
            source=_SOURCES[modality],
            seed=seed,
        )
        attach_demo_recordings_to_deploy(deploy_dir, bundle)
        result = validate_deploy_package(deploy_dir, strict_release=True)
        if not result.ok:
            raise RuntimeError(f"Updated {deploy_dir} failed validation: {'; '.join(result.errors)}")
        updated.append(deploy_dir)
    return updated


def main() -> None:
    """Parse arguments, attach bundles, and print publish-ready deploy paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=["ppg", "ecg", "all"], default="all")
    parser.add_argument(
        "--experiment-id",
        action="append",
        default=None,
        help="Optional golden experiment ID to update; repeat to update a release subset.",
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--datasets-root", type=Path, default=Path(default_datasets_dir()))
    parser.add_argument("--num-clips", type=int, default=10)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    modalities = ("ppg", "ecg") if args.modality == "all" else (args.modality,)
    experiment_ids = set(args.experiment_id) if args.experiment_id else None
    for modality in modalities:
        updated = attach_modality(
            modality,
            results_root=args.results_root,
            dataset_root=args.datasets_root,
            num_clips=args.num_clips,
            duration_seconds=args.duration_seconds,
            seed=args.seed,
            experiment_ids=experiment_ids,
        )
        print(f"Updated {len(updated)} {modality.upper()} RVQ deploy packages:")
        for deploy_dir in updated:
            print(f"  {deploy_dir}")


if __name__ == "__main__":
    main()
