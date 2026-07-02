"""Family-agnostic codec loader.

Reads ``deploy_manifest.json`` from a local directory or HuggingFace
repo, inspects the ``family`` field, and returns the matching codec
runtime. Lets callers do::

    from compressionkit.runtime import load_codec

    codec = load_codec("Ambiq/compressionkit-ppg-spiht-4x-v1.0")
    recon = codec.decompress(codec.compress(frame))

without having to know whether the underlying method is DSP-only,
AI-only, or hybrid.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from compressionkit.export.artifact_contract import ArtifactFile
from compressionkit.runtime.base import Codec  # used as return type annotation

logger = logging.getLogger(__name__)

__all__ = ["load_codec", "resolve_deploy_dir"]


def resolve_deploy_dir(repo_or_dir: str | Path) -> Path:
    """Return a local deploy directory, downloading from HF if needed.

    Args:
        repo_or_dir: Either a local path containing ``deploy_manifest.json``
            or a HuggingFace repo id (``"Ambiq/compressionkit-ppg-4x-v1.0"``).

    Returns:
        Local filesystem path to the deploy directory.
    """
    p = Path(str(repo_or_dir))
    if p.is_dir() and ((p / ArtifactFile.DEPLOY_MANIFEST).exists() or (p / ArtifactFile.HF_CONFIG).exists()):
        return p

    # Treat as HF repo id
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - import-time guard
        raise ImportError(
            "huggingface_hub is required to load codecs by repo id. Install with: uv sync --extra hf"
        ) from exc

    local_dir = snapshot_download(repo_id=str(repo_or_dir), repo_type="model")
    return Path(local_dir)


def _read_manifest(deploy_dir: Path) -> dict:
    manifest_path = deploy_dir / ArtifactFile.DEPLOY_MANIFEST
    if not manifest_path.exists():
        # HF repos historically used config.json
        alt = deploy_dir / ArtifactFile.HF_CONFIG
        if alt.exists():
            manifest_path = alt
        else:
            raise FileNotFoundError(
                f"No {ArtifactFile.DEPLOY_MANIFEST} (or {ArtifactFile.HF_CONFIG}) under {deploy_dir}"
            )
    with manifest_path.open() as f:
        return json.load(f)


def _ensure_deploy_manifest(deploy_dir: Path) -> None:
    """Ensure ``deploy_manifest.json`` exists, symlinking from ``config.json`` if needed.

    HuggingFace snapshots stage the manifest as ``config.json``; the
    family-specific codec constructors look for ``deploy_manifest.json``.
    """
    manifest_path = deploy_dir / ArtifactFile.DEPLOY_MANIFEST
    config_path = deploy_dir / ArtifactFile.HF_CONFIG
    if not manifest_path.exists() and config_path.exists():
        try:
            manifest_path.symlink_to(config_path.name)
        except OSError:
            # Symlinks unavailable (e.g. Windows w/o privilege); copy instead.
            import shutil

            shutil.copyfile(config_path, manifest_path)


def _ensure_alias(deploy_dir: Path, source_name: ArtifactFile, alias_name: ArtifactFile) -> None:
    source_path = deploy_dir / source_name
    alias_path = deploy_dir / alias_name
    if not source_path.exists() or alias_path.exists():
        return
    try:
        alias_path.symlink_to(source_path.name)
    except OSError:
        import shutil

        shutil.copyfile(source_path, alias_path)


def _ensure_rvq_hf_aliases(deploy_dir: Path) -> None:
    """Create local deploy names for RVQ HuggingFace snapshots when needed."""
    _ensure_alias(deploy_dir, ArtifactFile.ENCODER_INT8_HF_TFLITE, ArtifactFile.ENCODER_TFLITE)
    _ensure_alias(deploy_dir, ArtifactFile.DECODER_INT8_HF_TFLITE, ArtifactFile.DECODER_TFLITE)
    _ensure_alias(deploy_dir, ArtifactFile.SAMPLE_STIMULUS, ArtifactFile.SAMPLE_DATA)


def load_codec(repo_or_dir: str | Path) -> Codec:
    """Hydrate a codec from a local deploy directory or HF repo id.

    Dispatches on the manifest ``family`` field via the shared
    :data:`~compressionkit.export.family_registry.FAMILY_REGISTRY`. Unknown
    families raise ``ValueError`` with the list of known families.

    Args:
        repo_or_dir: Local deploy directory or HuggingFace repo id.

    Returns:
        A concrete codec instance implementing :class:`Codec`.
    """
    from compressionkit.export.family_registry import get_family_spec

    deploy_dir = resolve_deploy_dir(repo_or_dir)
    manifest = _read_manifest(deploy_dir)
    family: str = str(manifest.get("family", "rvq"))  # legacy manifests are RVQ
    _ensure_deploy_manifest(deploy_dir)

    spec = get_family_spec(family)
    if family == "rvq":
        _ensure_rvq_hf_aliases(deploy_dir)
    return spec.loader(deploy_dir)
