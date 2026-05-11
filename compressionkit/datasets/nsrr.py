"""National Sleep Research Resource (NSRR) download helpers.

Provides authenticated download of restricted-access datasets hosted on
the NSRR platform (https://sleepdata.org).  Users must first request
access to each dataset via the NSRR website, then set the ``NSRR_TOKEN``
environment variable to their personal API token.

Adapted from the pattern used in AmbiqAI/sleepkit.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_NSRR_API = "https://sleepdata.org/api/v1"


def authenticate_nsrr(token: str | None = None) -> str:
    """Validate an NSRR token and return it.

    Args:
        token: NSRR API token.  If ``None``, reads from the ``NSRR_TOKEN``
            environment variable.

    Returns:
        The validated token string.

    Raises:
        EnvironmentError: If no token is provided or found in the env.
        RuntimeError: If the token fails validation against the NSRR API.
    """
    token = token or os.environ.get("NSRR_TOKEN")
    if not token:
        raise OSError(
            "No NSRR token provided.  Set the NSRR_TOKEN environment "
            "variable or pass token= explicitly.\n"
            "  1. Sign in at https://sleepdata.org\n"
            "  2. Go to your profile → API Token\n"
            "  3. export NSRR_TOKEN=<your-token>"
        )

    resp = requests.get(
        f"{_NSRR_API}/account/profile.json",
        params={"auth_token": token},
        timeout=30,
    )
    if resp.status_code != 200 or not resp.json().get("authenticated"):
        raise RuntimeError(
            "NSRR token validation failed.  Check that your token is "
            "correct and that your account has access to the dataset."
        )

    email = resp.json().get("email", "unknown")
    logger.info("Authenticated with NSRR as %s", email)
    return token


def list_nsrr_files(
    db_slug: str,
    *,
    subfolder: str = "",
    token: str,
) -> list[dict]:
    """Recursively list files in an NSRR dataset.

    Args:
        db_slug: Dataset identifier (e.g. ``"mesa"``).
        subfolder: Subdirectory to list within the dataset.
        token: Validated NSRR API token.

    Returns:
        List of dicts with keys ``"full_path"``, ``"file_size"``,
        ``"is_file"``.
    """
    items: list[dict] = []
    path = subfolder or ""
    url = f"{_NSRR_API}/datasets/{db_slug}/files.json"

    resp = requests.get(
        url,
        params={"auth_token": token, "path": path},
        timeout=30,
    )
    resp.raise_for_status()

    for entry in resp.json():
        if entry.get("is_file"):
            items.append(entry)
        else:
            sub = entry.get("full_path", "")
            if sub:
                items.extend(
                    list_nsrr_files(db_slug, subfolder=sub, token=token)
                )

    return items


def download_nsrr_file(
    db_slug: str,
    remote_path: str,
    local_path: Path,
    *,
    token: str,
    expected_size: int | None = None,
) -> bool:
    """Download a single file from NSRR.

    Args:
        db_slug: Dataset identifier.
        remote_path: Path within the dataset (from ``list_nsrr_files``).
        local_path: Where to save locally.
        token: NSRR API token.
        expected_size: If provided, skip download when local file matches.

    Returns:
        ``True`` if file was downloaded, ``False`` if skipped.
    """
    if expected_size and local_path.exists() and local_path.stat().st_size == expected_size:
        return False

    local_path.parent.mkdir(parents=True, exist_ok=True)

    url = (
        f"https://sleepdata.org/datasets/{db_slug}/files/a/{token}"
        f"/m/compressionkit/{remote_path}"
    )

    resp = requests.get(url, stream=True, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to download {remote_path} (HTTP {resp.status_code}).  "
            f"Make sure you have access to the '{db_slug}' dataset on NSRR."
        )

    total = int(resp.headers.get("content-length", 0))
    with open(local_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=local_path.name, leave=False) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return True


def download_nsrr(
    db_slug: str,
    data_dir: str | os.PathLike,
    *,
    subfolder: str = "",
    pattern: str = "*",
    token: str | None = None,
    num_workers: int | None = None,
) -> int:
    """Download files from an NSRR dataset.

    Args:
        db_slug: Dataset slug (e.g. ``"mesa"`` or ``"mesa-commercial-use"``).
        data_dir: Local directory to store downloaded files.
        subfolder: Only download from this subdirectory.
        pattern: Glob-style filter on file paths (default ``"*"``).
        token: NSRR API token.  If ``None``, reads ``NSRR_TOKEN`` env var.
        num_workers: Unused, kept for API compatibility.

    Returns:
        Number of files downloaded (not counting skipped).

    Example::

        export NSRR_TOKEN="your-token-here"

        from compressionkit.datasets.nsrr import download_nsrr
        download_nsrr("mesa", "./datasets/mesa")
    """
    token = authenticate_nsrr(token)

    logger.info("Listing files in %s/%s ...", db_slug, subfolder)
    files = list_nsrr_files(db_slug, subfolder=subfolder, token=token)

    if pattern != "*":
        from fnmatch import fnmatch
        files = [f for f in files if fnmatch(f["full_path"], pattern)]

    logger.info("Found %d files to sync", len(files))
    data_dir = Path(data_dir)
    downloaded = 0

    for entry in tqdm(files, desc=f"Downloading {db_slug}", unit="file"):
        remote_path = entry["full_path"]
        local_path = data_dir / remote_path
        size = entry.get("file_size")

        if download_nsrr_file(db_slug, remote_path, local_path, token=token, expected_size=size):
            downloaded += 1

    logger.info("Downloaded %d / %d files (%d already present)", downloaded, len(files), len(files) - downloaded)
    return downloaded
