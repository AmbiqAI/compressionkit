"""Export RVQ codebook tables as C headers and NumPy archives.

The codebook is a pure lookup table with fixed memory layout, making it
ideal for direct embedding in firmware as a const array.  Each RVQ level
has its own ``(num_embeddings, embedding_dim)`` table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def extract_codebooks(rvq_weights: list[np.ndarray]) -> list[np.ndarray]:
    """Extract per-level codebook matrices from RVQ weight list.

    The weight list from ``rvq.get_weights()`` contains one embedding matrix
    per RVQ level in order, each with shape ``(num_embeddings, embedding_dim)``.

    Args:
        rvq_weights: List of weight arrays from ``rvq.get_weights()``.

    Returns:
        List of codebook arrays, one per level.
    """
    codebooks = []
    for w in rvq_weights:
        if w.ndim == 2:
            codebooks.append(w)
    return codebooks


def _format_c_array(
    name: str,
    data: np.ndarray,
    dtype_str: str = "float",
) -> str:
    """Format a NumPy array as a C const array string."""
    flat = data.flatten()
    rows = []
    rows.append(f"// shape: {list(data.shape)}")
    rows.append(f"static const {dtype_str} {name}[{len(flat)}] = {{")
    # Write in chunks of 8 values per line
    chunk_size = 8
    for i in range(0, len(flat), chunk_size):
        chunk = flat[i : i + chunk_size]
        vals = ", ".join(f"{v:.8g}f" for v in chunk)
        comma = "," if i + chunk_size < len(flat) else ""
        rows.append(f"    {vals}{comma}")
    rows.append("};")
    return "\n".join(rows)


def export_codebooks_npz(
    rvq_weights: list[np.ndarray],
    output_path: Path,
) -> Path:
    """Save RVQ codebook tables to a NumPy ``.npz`` archive.

    Args:
        rvq_weights: Weight list from ``rvq.get_weights()``.
        output_path: Path for the ``.npz`` file.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codebooks = extract_codebooks(rvq_weights)
    arrays = {f"level_{i}": cb for i, cb in enumerate(codebooks)}
    np.savez(output_path, **arrays)
    logger.info("Exported %d codebook level(s) to %s", len(codebooks), output_path)
    return output_path


def export_codebooks_header(
    rvq_weights: list[np.ndarray],
    output_path: Path,
    *,
    prefix: str = "rvq_codebook",
    guard: str | None = None,
) -> Path:
    """Export RVQ codebook tables as a C header file.

    Args:
        rvq_weights: Weight list from ``rvq.get_weights()``.
        output_path: Path for the ``.h`` file.
        prefix: Prefix for C array names (e.g. ``rvq_codebook_l0``).
        guard: Include guard macro name; auto-generated if ``None``.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codebooks = extract_codebooks(rvq_weights)

    if guard is None:
        guard = output_path.stem.upper().replace("-", "_").replace(".", "_") + "_H"

    lines: list[str] = []
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append(f"#define {prefix.upper()}_NUM_LEVELS {len(codebooks)}")
    if codebooks:
        lines.append(f"#define {prefix.upper()}_NUM_EMBEDDINGS {codebooks[0].shape[0]}")
        lines.append(f"#define {prefix.upper()}_EMBEDDING_DIM {codebooks[0].shape[1]}")
    lines.append("")

    for i, cb in enumerate(codebooks):
        lines.append(_format_c_array(f"{prefix}_l{i}", cb))
        lines.append("")

    lines.append(f"#endif  // {guard}")
    lines.append("")

    output_path.write_text("\n".join(lines))
    logger.info("Exported %d codebook level(s) as C header to %s", len(codebooks), output_path)
    return output_path


__all__ = [
    "export_codebooks_header",
    "export_codebooks_npz",
    "extract_codebooks",
]
