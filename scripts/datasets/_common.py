"""Shared CLI conventions for ``scripts/datasets/download_*.py``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

DEFAULT_DATA_ROOT = Path("datasets")


def make_parser(*, slug: str, description: str) -> argparse.ArgumentParser:
    """Build the standard argument parser used by every ingest script.

    Every script accepts:
        --root         destination root (default: ``datasets/``).
        --limit N      stop after N records (smoke testing).
        --force        re-download even if files exist.
        --skip-convert pull raw archives only; skip h5 conversion.
        --skip-download already extracted? just convert from --raw-dir.
        --raw-dir      where raw downloads live (default: ``<root>/<slug>_raw``).
        --upload-s3 KEY  optional ``bucket/key`` to publish the canonical zip.
        -v / -q        log verbosity.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Destination root for canonical h5 files (default: datasets/).",
    )
    p.add_argument("--raw-dir", type=Path, default=None, help=f"Raw download dir (default: <root>/{slug}_raw).")
    p.add_argument("--limit", type=int, default=None, help="Smoke test: only ingest the first N records.")
    p.add_argument("--force", action="store_true", help="Re-download even if archives or h5 files already exist.")
    p.add_argument("--skip-convert", action="store_true", help="Download raw archives only; skip converting to h5.")
    p.add_argument("--skip-download", action="store_true", help="Use existing files in --raw-dir; skip downloading.")
    p.add_argument(
        "--upload-s3",
        default=None,
        metavar="BUCKET/KEY",
        help="After conversion, zip <root>/<slug>/ and upload to s3://BUCKET/KEY.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    p.add_argument("-q", "--quiet", action="store_true", help="Only print warnings and errors.")
    return p


def setup_logging(args: argparse.Namespace) -> logging.Logger:
    """Standard logger setup based on ``-v``/``-q`` flags."""
    import sys

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    # ``force=True`` is required because tensorflow may already have
    # configured the root logger by the time we get here.
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    return logging.getLogger("ingest")


def resolve_dirs(args: argparse.Namespace, slug: str) -> tuple[Path, Path]:
    """Return ``(canonical_dir, raw_dir)`` and create them.

    canonical_dir = ``<root>/<slug>``
    raw_dir       = ``<args.raw_dir>`` or ``<root>/<slug>_raw``
    """
    canonical = (args.root / slug).resolve()
    raw = (args.raw_dir or args.root / f"{slug}_raw").resolve()
    canonical.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    return canonical, raw


def maybe_upload_s3(canonical_dir: Path, target: str | None, *, slug: str) -> None:
    """If ``target`` is provided as ``bucket/key``, build a zip and push it."""
    if not target:
        return
    if "/" not in target:
        raise ValueError(f"--upload-s3 must be 'bucket/key', got {target!r}")
    bucket, key = target.split("/", 1)
    from compressionkit.datasets._download import s3_upload_zip

    s3_upload_zip(
        canonical_dir,
        bucket=bucket,
        key=key,
        workdir=canonical_dir.parent,
        glob="*.h5",
    )
