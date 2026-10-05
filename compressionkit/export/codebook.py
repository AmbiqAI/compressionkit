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


def extract_codebooks(
    rvq_weights: list[np.ndarray],
    *,
    num_levels: int | None = None,
    use_ema: bool = False,
    kmeans_init: bool = False,
) -> list[np.ndarray]:
    """Extract per-level codebook matrices from RVQ weight list.

    Plain RVQ stores one matrix per level. EMA RVQ stores a codebook,
    count vector, and embedding-sum matrix per level, followed by an optional
    scalar k-means flag. The layout must be declared; matrix rank alone
    cannot distinguish codebooks from EMA accumulators.

    Args:
        rvq_weights: List of weight arrays from ``rvq.get_weights()``.
        num_levels: Expected number of levels. Required for EMA checkpoints.
        use_ema: Whether the source is an EMA RVQ checkpoint.
        kmeans_init: Whether the EMA checkpoint includes its warm-start flag.

    Returns:
        List of codebook arrays, one per level.

    Raises:
        ValueError: The weights do not match the declared layout.
    """
    if use_ema and num_levels is None:
        raise ValueError("EMA codebook extraction requires num_levels")
    if kmeans_init and not use_ema:
        raise ValueError("kmeans_init requires an EMA checkpoint")
    levels = len(rvq_weights) if num_levels is None else num_levels
    if levels < 1:
        raise ValueError("At least one RVQ level is required")
    stride = 3 if use_ema else 1
    expected = stride * levels + int(kmeans_init)
    if len(rvq_weights) != expected:
        raise ValueError(f"Expected {expected} RVQ weight arrays for {levels} levels, got {len(rvq_weights)}")
    codebooks: list[np.ndarray] = []
    for level in range(levels):
        cb = np.asarray(rvq_weights[stride * level])
        if cb.ndim != 2 or min(cb.shape) < 1 or not np.issubdtype(cb.dtype, np.floating):
            raise ValueError(f"Level {level}: expected a nonempty floating-point codebook matrix")
        if not np.isfinite(cb).all():
            raise ValueError(f"Level {level}: non-finite codebook values")
        if codebooks and cb.shape[1] != codebooks[0].shape[1]:
            raise ValueError("RVQ codebooks must share an embedding dimension")
        if use_ema:
            count, total = (np.asarray(w) for w in rvq_weights[stride * level + 1 : stride * level + 3])
            if count.shape != (cb.shape[0],) or total.shape != cb.shape:
                raise ValueError(f"Level {level}: EMA count/sum shapes do not match the codebook")
        codebooks.append(cb)
    if kmeans_init:
        flag = np.asarray(rvq_weights[-1])
        if flag.shape != () or flag.item() not in (0, 1):
            raise ValueError("Expected a scalar 0/1 kmeans_done flag")
    return codebooks


def load_rvq_weights(path: str | Path) -> list[np.ndarray]:
    """Read a positional RVQ training archive in numeric weight order.

    Reject missing or unexpected keys instead of silently reordering or
    skipping state. Lexical sorting would place ``arr_10`` before ``arr_2``.
    """
    with np.load(path, allow_pickle=False) as archive:
        keys = [f"arr_{i}" for i in range(len(archive.files))]
        if set(archive.files) != set(keys):
            raise ValueError("RVQ training archive must contain contiguous arr_0 ... arr_N keys")
        return [archive[key] for key in keys]


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
        # Nine significant digits round-trip float32. C requires a decimal
        # point or exponent before the suffix (``0f`` is not a valid literal).
        literals = [f"{v:.9g}" for v in chunk]
        vals = ", ".join(f"{v if '.' in v or 'e' in v else v + '.0'}f" for v in literals)
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
        rvq_weights: Codebook matrices only; use ``extract_codebooks`` first
            for an EMA checkpoint.
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
        rvq_weights: Codebook matrices only; use ``extract_codebooks`` first
            for an EMA checkpoint.
        output_path: Path for the ``.h`` file.
        prefix: Prefix for C array names (e.g. ``rvq_codebook_l0``).
        guard: Include guard macro name; auto-generated if ``None``.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codebooks = extract_codebooks(rvq_weights)

    if any(cb.shape != codebooks[0].shape for cb in codebooks):
        raise ValueError("The C header requires equally sized RVQ codebooks")

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
    "load_rvq_weights",
]
