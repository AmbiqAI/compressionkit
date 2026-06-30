"""Publish a deployment package to HuggingFace Hub.

Usage::

    # Publish from a golden deploy directory
    python scripts/publish_to_huggingface.py \\
        --deploy-dir results/ppg_rvq_64hz_04x_golden/deploy \\
        --repo-id Ambiq/compressionkit-ppg-4x

    # With a quality scorecard
    python scripts/publish_to_huggingface.py \\
        --deploy-dir results/ppg_rvq_64hz_04x_golden/deploy \\
        --repo-id Ambiq/compressionkit-ppg-4x \\
        --scorecard results/ppg_rvq_64hz_04x_golden/quality_scorecard.json

    # Dry run (generate model card only, don't upload)
    python scripts/publish_to_huggingface.py \\
        --deploy-dir results/ppg_rvq_64hz_04x_golden/deploy \\
        --repo-id Ambiq/compressionkit-ppg-4x \\
        --dry-run

Requires ``HF_TOKEN`` environment variable or ``huggingface-cli login``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Files from an RVQ deploy directory to upload, mapped to their HF repo names.
# If a file doesn't exist, it is silently skipped.
_DEPLOY_FILE_MAP: dict[str, str] = {
    "encoder.tflite": "encoder_int8.tflite",
    "encoder.h": "encoder.h",
    "decoder_float32.tflite": "decoder_float32.tflite",
    "decoder.tflite": "decoder_int8.tflite",
    "decoder_int8.tflite": "decoder_int8.tflite",
    "decoder.h": "decoder.h",
    "codebook.npz": "codebook.npz",
    "codebook.h": "codebook.h",
    "codec_spec.json": "codec_spec.json",
    "sample_data.npz": "sample_stimulus.npz",
    "sample_stimulus.npz": "sample_stimulus.npz",
    "deploy_manifest.json": "config.json",
    "model_card.json": "model_card.json",
    # Two-stage prior artifacts (only present for two_stage golden runs, #27).
    "prior_int8.tflite": "prior_int8.tflite",
    "prior_int8.h": "prior_int8.h",
    "prior_manifest.json": "prior_manifest.json",
}

# Files from a SPIHT (DSP-only) deploy directory. The vendored C sources
# live in a ``c_sources/`` subdirectory and are uploaded recursively.
_SPIHT_FILE_MAP: dict[str, str] = {
    "deploy_manifest.json": "config.json",
    "spiht_config.json": "spiht_config.json",
    "codec_spec.json": "codec_spec.json",
    "sample_stimulus.npz": "sample_stimulus.npz",
    "reference_vectors.npz": "reference_vectors.npz",
    "model_card.json": "model_card.json",
    "scorecard.json": "scorecard.json",
}


def _detect_family(deploy_dir: Path) -> str:
    manifest_path = deploy_dir / "deploy_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"deploy_manifest.json not found in {deploy_dir}")
    with manifest_path.open() as f:
        manifest = json.load(f)
    # Legacy RVQ manifests didn't carry a family field.
    return str(manifest.get("family", "rvq"))


def _stage_files(deploy_dir: Path, staging_dir: Path, scorecard_path: Path | None) -> list[str]:
    """Copy RVQ deploy artifacts to a staging directory with HF naming."""
    staged: list[str] = []

    for src_name, dst_name in _DEPLOY_FILE_MAP.items():
        src = deploy_dir / src_name
        dst = staging_dir / dst_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            staged.append(dst_name)
            logger.info("Staged: %s → %s", src_name, dst_name)

    # Copy scorecard if provided
    if scorecard_path and scorecard_path.exists():
        dst = staging_dir / "quality_scorecard.json"
        shutil.copy2(scorecard_path, dst)
        staged.append("quality_scorecard.json")
        logger.info("Staged: scorecard → quality_scorecard.json")

    # Include the custom model weights license if it exists
    license_file = Path(__file__).resolve().parent.parent / "LICENSE-MODEL-WEIGHTS.md"
    if license_file.exists():
        dst = staging_dir / "LICENSE-MODEL-WEIGHTS.md"
        shutil.copy2(license_file, dst)
        staged.append("LICENSE-MODEL-WEIGHTS.md")
        logger.info("Staged: LICENSE-MODEL-WEIGHTS.md")

    return staged


def _stage_files_spiht(deploy_dir: Path, staging_dir: Path, scorecard_path: Path | None) -> list[str]:
    """Copy SPIHT (DSP-only) deploy artifacts to a staging directory.

    Includes the ``c_sources/`` subtree verbatim so consumers get the
    portable C99 reference plus the generated ``spiht_app_config.h``.
    """
    staged: list[str] = []

    for src_name, dst_name in _SPIHT_FILE_MAP.items():
        src = deploy_dir / src_name
        dst = staging_dir / dst_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            staged.append(dst_name)
            logger.info("Staged: %s → %s", src_name, dst_name)

    # Copy the c_sources/ subtree if present.
    c_src = deploy_dir / "c_sources"
    if c_src.is_dir():
        dst_dir = staging_dir / "c_sources"
        shutil.copytree(c_src, dst_dir, dirs_exist_ok=True)
        for p in sorted(dst_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(staging_dir)
                staged.append(str(rel))
                logger.info("Staged: c_sources/%s", p.name)

    # Optional external scorecard.
    if scorecard_path and scorecard_path.exists():
        dst = staging_dir / "quality_scorecard.json"
        shutil.copy2(scorecard_path, dst)
        staged.append("quality_scorecard.json")
        logger.info("Staged: scorecard → quality_scorecard.json")

    return staged


def publish(
    deploy_dir: str | Path,
    repo_id: str,
    scorecard_path: str | Path | None = None,
    license_id: str = "other",
    private: bool = False,
    dry_run: bool = False,
) -> Path | None:
    """Stage and publish deployment artifacts to HuggingFace Hub.

    Args:
        deploy_dir: Path to deployment directory with ``deploy_manifest.json``.
        repo_id: HuggingFace repo ID (e.g. ``Ambiq/compressionkit-ppg-4x``).
        scorecard_path: Optional path to ``quality_scorecard.json``.
        license_id: SPDX license ID for the model card.
        private: Whether to create a private repo.
        dry_run: If *True*, stage files and generate model card but don't upload.

    Returns:
        Path to staging directory (useful for dry-run inspection), or *None*.

    Raises:
        FileNotFoundError: If ``deploy_manifest.json`` is not found.
        ImportError: If ``huggingface_hub`` is not installed and dry_run is False.
        ValueError: If no files are found to stage.
    """
    deploy_dir = Path(deploy_dir)
    if not (deploy_dir / "deploy_manifest.json").exists():
        raise FileNotFoundError(f"deploy_manifest.json not found in {deploy_dir}")

    family = _detect_family(deploy_dir)
    _KNOWN_FAMILIES = ("rvq", "spiht")
    if family not in _KNOWN_FAMILIES:
        raise ValueError(f"Unknown deploy family {family!r}; expected one of {_KNOWN_FAMILIES!r}")
    logger.info("Detected deploy family: %s", family)

    # Validate HuggingFace availability before allocating any resources (skip for dry runs)
    if not dry_run:
        try:
            from huggingface_hub import HfApi as _HfApi  # noqa: F401 — import check only
        except ImportError as exc:
            raise ImportError("huggingface_hub not installed. Install with: uv sync --extra hf") from exc

    # Create staging directory
    staging_dir = Path(tempfile.mkdtemp(prefix="hf_release_"))
    logger.info("Staging directory: %s", staging_dir)

    sc_path = Path(scorecard_path) if scorecard_path else None
    if sc_path is None:
        # Point the release at the corrected scorecard (carries the v1 headline
        # block with the paired clean-truth / noise-regime view) rather than any
        # frozen deploy-time copy.
        candidate = deploy_dir.parent / "quality_scorecard.json"
        if candidate.exists():
            sc_path = candidate
            logger.info("Using corrected scorecard: %s", sc_path)
    # Stage and generate card; clean up on any error
    try:
        if family == "spiht":
            staged = _stage_files_spiht(deploy_dir, staging_dir, sc_path)
        else:
            staged = _stage_files(deploy_dir, staging_dir, sc_path)
        if not staged:
            raise ValueError("No files staged — check deploy directory contents")

        if family == "spiht":
            from compressionkit.export.model_card import generate_spiht_model_card

            # DSP-only SPIHT has no proprietary weights — default to Apache-2.0
            # unless the caller explicitly overrode license_id.
            spiht_license = "apache-2.0" if license_id == "other" else license_id
            card_text = generate_spiht_model_card(
                deploy_dir=deploy_dir,
                scorecard_path=sc_path,
                license_id=spiht_license,
            )
        else:
            from compressionkit.export.model_card import generate_model_card

            card_text = generate_model_card(
                deploy_dir=deploy_dir,
                scorecard_path=sc_path,
                license_id=license_id,
            )
        readme_path = staging_dir / "README.md"
        readme_path.write_text(card_text)
        staged.append("README.md")
        logger.info("Generated README.md model card (%d chars)", len(card_text))
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    logger.info("Staged %d files for %s", len(staged), repo_id)
    for name in sorted(staged):
        size = (staging_dir / name).stat().st_size
        logger.info("  %s (%d bytes)", name, size)

    if dry_run:
        logger.info("Dry run — files staged at %s", staging_dir)
        return staging_dir

    # Upload to HuggingFace; staging dir is always removed after this point
    from huggingface_hub import HfApi  # already verified importable above

    api = HfApi()
    try:
        # Create repo if it doesn't exist
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        logger.info("Repo ready: %s", repo_id)

        # Upload all staged files
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Release deployment artifacts from CompressionKit",
        )
        logger.info("Published to https://huggingface.co/%s", repo_id)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish compressionkit deployment artifacts to HuggingFace Hub.")
    parser.add_argument(
        "--deploy-dir",
        type=Path,
        required=True,
        help="Path to deploy directory (must contain deploy_manifest.json).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g. Ambiq/compressionkit-ppg-4x).",
    )
    parser.add_argument(
        "--scorecard",
        type=Path,
        default=None,
        help="Path to quality_scorecard.json.",
    )
    parser.add_argument(
        "--license",
        type=str,
        default="other",
        help="SPDX license ID for model card (default: other — Ambiq silicon only).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage files and generate model card without uploading.",
    )
    args = parser.parse_args()

    try:
        publish(
            deploy_dir=args.deploy_dir,
            repo_id=args.repo_id,
            scorecard_path=args.scorecard,
            license_id=args.license,
            private=args.private,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
