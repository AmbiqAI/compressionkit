"""Download + extract utilities for dataset ingestion scripts.

Used by ``scripts/datasets/*.py`` to pull public physio-signal datasets onto
disk before converting them to the project's canonical h5 layout.

Design goals:
- Resumable HTTP/HTTPS downloads with progress bar (tqdm).
- Optional sha256 checksum verification.
- Convenience extractors for ``.zip`` and ``.tar(.gz|.bz2|.xz)``.
- Thin PhysioNet helper that knows the standard layout
  ``https://physionet.org/files/<slug>/<version>/...``.
- S3 upload helper for re-publishing sanitized h5 bundles.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PHYSIONET_BASE = "https://physionet.org/files"

_CHUNK = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# HTTP download
# ---------------------------------------------------------------------------


def http_download(
    url: str,
    dst: str | os.PathLike,
    *,
    sha256: str | None = None,
    resume: bool = True,
    timeout: float = 60.0,
    chunk_size: int = _CHUNK,
    show_progress: bool = True,
) -> Path:
    """Download a URL to ``dst`` with optional resume + checksum.

    Args:
        url: HTTP/HTTPS URL.
        dst: Destination file path. Parent dirs are created.
        sha256: If provided, verified after a fresh download. Mismatched
            files are removed.
        resume: If True and the partial file exists, send a Range header.
        timeout: Per-request timeout in seconds.
        chunk_size: Streaming chunk size.
        show_progress: Whether to render a tqdm bar.

    Returns:
        Final path written.

    Raises:
        RuntimeError: On HTTP errors or checksum mismatch.
    """
    import requests
    from tqdm import tqdm

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Quick skip when an existing file matches the expected checksum.
    if dst_path.exists() and sha256 and _file_sha256(dst_path) == sha256:
        logger.info("✓ %s already present and checksum matches.", dst_path.name)
        return dst_path

    headers: dict[str, str] = {}
    mode = "wb"
    initial = 0
    if resume and dst_path.exists():
        initial = dst_path.stat().st_size
        if initial > 0:
            headers["Range"] = f"bytes={initial}-"
            mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=timeout) as resp:
        if resp.status_code in (200, 206):
            total = int(resp.headers.get("Content-Length", 0))
            if resp.status_code == 200:
                # Server didn't honour Range — start over.
                mode = "wb"
                initial = 0
            total_size = total + initial
            with (
                open(dst_path, mode) as fh,
                tqdm(
                    total=total_size or None,
                    initial=initial,
                    unit="B",
                    unit_scale=True,
                    desc=dst_path.name,
                    disable=not show_progress,
                ) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    bar.update(len(chunk))
        elif resp.status_code == 416 and dst_path.exists():
            # Already complete on disk.
            logger.info("✓ %s already complete.", dst_path.name)
        else:
            raise RuntimeError(f"HTTP {resp.status_code} downloading {url}: {resp.text[:200]}")

    if sha256:
        actual = _file_sha256(dst_path)
        if actual != sha256:
            dst_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch for {dst_path.name}: expected {sha256}, got {actual}",
            )
    return dst_path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_archive(
    archive: str | os.PathLike,
    dst_dir: str | os.PathLike,
    *,
    delete_after: bool = False,
) -> Path:
    """Extract a ``.zip`` / ``.tar`` / ``.tar.gz`` / ``.tgz`` / ``.tar.bz2`` /
    ``.tar.xz`` archive into ``dst_dir``.

    Args:
        archive: Path to the compressed archive.
        dst_dir: Destination directory (created if needed).
        delete_after: Remove the archive once extraction succeeds.

    Returns:
        ``dst_dir`` as a :class:`Path`.
    """
    archive_path = Path(archive)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    logger.info("Extracting %s → %s", archive_path.name, dst)
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dst)
    elif (
        name.endswith(".tar")
        or name.endswith(".tar.gz")
        or name.endswith(".tgz")
        or name.endswith(".tar.bz2")
        or name.endswith(".tar.xz")
    ):
        with tarfile.open(archive_path) as tf:
            tf.extractall(dst, filter="data")
    else:
        raise ValueError(f"Unsupported archive type: {archive_path.name}")
    if delete_after:
        archive_path.unlink(missing_ok=True)
    return dst


# ---------------------------------------------------------------------------
# PhysioNet helpers
# ---------------------------------------------------------------------------


def physionet_url(slug: str, version: str, relpath: str) -> str:
    """Build a PhysioNet ``files`` URL.

    Example::

        physionet_url("mitdb", "1.0.0", "100.dat")
        # → https://physionet.org/files/mitdb/1.0.0/100.dat
    """
    relpath = relpath.lstrip("/")
    return f"{PHYSIONET_BASE}/{slug}/{version}/{relpath}"


def download_physionet_files(
    slug: str,
    version: str,
    relpaths: Iterable[str],
    dst_dir: str | os.PathLike,
    *,
    show_progress: bool = True,
) -> list[Path]:
    """Download a list of files from a PhysioNet database.

    Args:
        slug: Database short name, e.g. ``"mitdb"``.
        version: Version string from the PhysioNet database page.
        relpaths: Relative paths under the database root.
        dst_dir: Local destination directory; sub-paths inside are preserved.
        show_progress: Whether to render per-file tqdm bars.

    Returns:
        List of downloaded local paths in the same order as *relpaths*.
    """
    dst = Path(dst_dir)
    out: list[Path] = []
    for rp in relpaths:
        url = physionet_url(slug, version, rp)
        local = dst / rp
        http_download(url, local, show_progress=show_progress)
        out.append(local)
    return out


# ---------------------------------------------------------------------------
# Ambiq AI S3 bundle helpers (matches heartkit convention)
# ---------------------------------------------------------------------------

AMBIQ_S3_BUCKET = "ambiq-ai-datasets"


def download_ambiq_s3_zip(
    slug: str,
    dst_dir: str | os.PathLike,
    *,
    bucket: str = AMBIQ_S3_BUCKET,
    key: str | None = None,
    extract: bool = True,
    delete_zip: bool = False,
    force: bool = False,
) -> Path:
    """Pull ``s3://<bucket>/<slug>/<slug>.zip`` and optionally extract it.

    This mirrors the convention used by AmbiqAI/heartkit so the same bundles
    can be reused (PTB-XL, LSAD, LUDB, QTDB, icentia_mini are already
    published there).

    Args:
        slug: Dataset short name (e.g. ``"ludb"``); used for both the S3 prefix
            and the local zip filename.
        dst_dir: Destination directory; created if missing. The extracted h5
            files (or whatever is in the zip) are placed directly here.
        bucket: S3 bucket name. Defaults to the Ambiq AI public bucket.
        key: Override the S3 key. Defaults to ``"<slug>/<slug>.zip"``.
        extract: If True, extract the archive after download.
        delete_zip: Remove the zip after successful extraction.
        force: Re-download even if the local zip is present and matches.

    Returns:
        Path to the local zip file.
    """
    import helia_edge as helia

    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    s3_key = key or f"{slug}/{slug}.zip"
    zip_path = dst / Path(s3_key).name

    logger.info("Fetching s3://%s/%s → %s", bucket, s3_key, zip_path)
    did_download = helia.utils.download_s3_file(
        key=s3_key,
        dst=zip_path,
        bucket=bucket,
        checksum="size",
    )
    if force and not did_download:
        # User asked for force; remove and retry.
        zip_path.unlink(missing_ok=True)
        helia.utils.download_s3_file(
            key=s3_key,
            dst=zip_path,
            bucket=bucket,
            checksum="size",
        )

    if extract and zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dst)
        logger.info("Extracted %s into %s", zip_path.name, dst)
    if delete_zip:
        zip_path.unlink(missing_ok=True)
    return zip_path


def download_ambiq_s3_prefix(
    slug: str,
    dst_dir: str | os.PathLike,
    *,
    bucket: str = AMBIQ_S3_BUCKET,
    prefix: str | None = None,
    num_workers: int | None = None,
    max_objects: int | None = None,
    anonymous: bool = True,
) -> Path:
    """Pull objects under ``s3://<bucket>/<prefix>/`` to ``dst_dir``.

    For prefixes with thousands of objects (e.g. ``icentia11k`` at ~330 GB
    total), pass ``max_objects`` to bound the pull. We list first, then issue
    one ``GetObject`` per file via :mod:`boto3`. Files already on disk with
    matching size are skipped.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    pfx = prefix or slug
    if not pfx.endswith("/"):
        pfx = pfx + "/"
    logger.info(
        "Listing s3://%s/%s (max_objects=%s) …",
        bucket,
        pfx,
        max_objects,
    )

    cfg = Config(signature_version=UNSIGNED) if anonymous else None
    client = boto3.client("s3", config=cfg)

    paginator = client.get_paginator("list_objects_v2")
    keys: list[tuple[str, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=pfx):
        for obj in page.get("Contents", []):
            keys.append((obj["Key"], obj["Size"]))
            if max_objects is not None and len(keys) >= max_objects:
                break
        if max_objects is not None and len(keys) >= max_objects:
            break
    logger.info("Will fetch %d objects → %s", len(keys), dst)

    try:
        from tqdm import tqdm

        bar: Any = tqdm(total=len(keys), unit="file", desc=slug)
    except ImportError:  # pragma: no cover
        bar = None

    for key, size in keys:
        rel = key[len(pfx) :] if key.startswith(pfx) else Path(key).name
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size == size:
            if bar is not None:
                bar.update(1)
            continue
        client.download_file(bucket, key, str(out))
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return dst


# ---------------------------------------------------------------------------
# S3 upload (sanitized re-publishing)
# ---------------------------------------------------------------------------


def s3_upload_zip(
    src_dir: str | os.PathLike,
    *,
    bucket: str,
    key: str,
    workdir: str | os.PathLike | None = None,
    glob: str = "*",
    dry_run: bool = False,
) -> Path | None:
    """Zip a directory and upload it to S3.

    Mirrors the convention used by :class:`PtbxlDataset.download` so future
    consumers can pull the sanitized bundle back the same way.

    Args:
        src_dir: Directory containing the canonical h5 files (or whatever
            should be archived).
        bucket: Target S3 bucket.
        key: Target S3 key (e.g. ``"mitdb/mitdb_v1.zip"``).
        workdir: Directory to place the zip in. Defaults to ``src_dir.parent``.
        glob: Pattern to include from *src_dir* (default: everything).
        dry_run: If True, only build the zip locally and skip the upload.

    Returns:
        Path to the local zip (so callers can keep / delete it).
    """
    src = Path(src_dir).resolve()
    work = Path(workdir).resolve() if workdir else src.parent
    work.mkdir(parents=True, exist_ok=True)
    zip_path = work / Path(key).name
    logger.info("Building %s from %s …", zip_path.name, src)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for fp in sorted(src.rglob(glob)):
            if fp.is_file():
                zf.write(fp, arcname=fp.relative_to(src))
    size_mb = zip_path.stat().st_size / 2**20
    logger.info("Built %s (%.1f MiB)", zip_path.name, size_mb)
    if dry_run:
        logger.info("dry_run=True → skipping S3 upload.")
        return zip_path

    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - boto3 only in publish env
        raise RuntimeError(
            "boto3 is required for S3 upload. Install it in your publishing environment (`pip install boto3`).",
        ) from exc

    logger.info("Uploading to s3://%s/%s …", bucket, key)
    boto3.client("s3").upload_file(str(zip_path), bucket, key)
    logger.info("Uploaded.")
    return zip_path


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def safe_rmtree(path: str | os.PathLike) -> None:
    """``shutil.rmtree`` that does not raise if the directory is missing."""
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)


__all__ = [
    "AMBIQ_S3_BUCKET",
    "PHYSIONET_BASE",
    "download_ambiq_s3_prefix",
    "download_ambiq_s3_zip",
    "download_physionet_files",
    "extract_archive",
    "http_download",
    "physionet_url",
    "s3_upload_zip",
    "safe_rmtree",
]
