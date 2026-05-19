"""Family-agnostic codec loader.

Reads ``deploy_manifest.json`` from a local directory or HuggingFace
repo, inspects the ``family`` field, and returns the matching codec
runtime. Lets callers do::

    from compressionkit.runtime import load_codec

    codec = load_codec("Ambiq/compressionkit-ppg-spiht-4x")
    recon = codec.decompress(codec.compress(frame))

without having to know whether the underlying method is DSP-only,
AI-only, or hybrid.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from compressionkit.runtime.base import Codec

logger = logging.getLogger(__name__)

__all__ = ["load_codec", "resolve_deploy_dir"]


def resolve_deploy_dir(repo_or_dir: str | Path) -> Path:
    """Return a local deploy directory, downloading from HF if needed.

    Args:
        repo_or_dir: Either a local path containing ``deploy_manifest.json``
            or a HuggingFace repo id (``"Ambiq/compressionkit-ppg-4x"``).

    Returns:
        Local filesystem path to the deploy directory.
    """
    p = Path(str(repo_or_dir))
    if p.exists() and p.is_dir():
        return p

    # Treat as HF repo id
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - import-time guard
        raise ImportError(
            "huggingface_hub is required to load codecs by repo id. "
            "Install with: uv sync --extra hf"
        ) from exc

    local_dir = snapshot_download(repo_id=str(repo_or_dir), repo_type="model")
    return Path(local_dir)


def _read_manifest(deploy_dir: Path) -> dict:
    manifest_path = deploy_dir / "deploy_manifest.json"
    if not manifest_path.exists():
        # HF repos historically used config.json
        alt = deploy_dir / "config.json"
        if alt.exists():
            manifest_path = alt
        else:
            raise FileNotFoundError(
                f"No deploy_manifest.json (or config.json) under {deploy_dir}"
            )
    with manifest_path.open() as f:
        return json.load(f)


def load_codec(repo_or_dir: str | Path) -> Codec:
    """Hydrate a codec from a local deploy directory or HF repo id.

    Dispatches on the manifest ``family`` field. Unknown families raise
    ``ValueError`` with the list of known families.

    Args:
        repo_or_dir: Local deploy directory or HuggingFace repo id.

    Returns:
        A concrete codec instance implementing :class:`Codec`.
    """
    deploy_dir = resolve_deploy_dir(repo_or_dir)
    manifest = _read_manifest(deploy_dir)
    family = manifest.get("family", "rvq")  # legacy manifests are RVQ

    if family == "rvq":
        from compressionkit.runtime.codec import RVQCodec

        return RVQCodec(deploy_dir)
    if family == "spiht":
        from compressionkit.runtime.spiht import SpihtCodec

        return SpihtCodec.from_deploy_dir(deploy_dir)

    raise ValueError(
        f"Unknown codec family {family!r} in {deploy_dir}. "
        "Known families: 'rvq', 'spiht'."
    )
