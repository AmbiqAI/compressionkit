"""``python -m compressionkit`` — dispatch to a registered recipe."""

from __future__ import annotations

import sys

from compressionkit.recipes import dispatch

if __name__ == "__main__":
    sys.exit(dispatch())
