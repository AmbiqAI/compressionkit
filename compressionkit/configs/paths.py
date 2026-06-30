"""Filesystem-path defaults for configuration models.

Keeps dataset-root defaults environment-driven so the package works outside the
dev container (e.g. a plain ``pip install``) without editing code or configs.
"""

from __future__ import annotations

import os

#: Environment variable that overrides the default dataset root directory.
DATASETS_DIR_ENV = "COMPRESSIONKIT_DATASETS_DIR"


def default_datasets_dir() -> str:
    """Return the default dataset root directory.

    Resolved from the ``COMPRESSIONKIT_DATASETS_DIR`` environment variable when
    set, otherwise a ``datasets`` directory relative to the current working
    directory (the repository / workspace root).
    """
    return os.environ.get(DATASETS_DIR_ENV, "datasets")
