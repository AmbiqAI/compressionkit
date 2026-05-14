"""Verify that cross-channel prior architectures export to INT8 LiteRT.

Builds each xlead prior variant, runs the helia_edge TFLite converter,
and reports model size and op compatibility.

Usage:
    python scripts/verify_xlead_litert.py --vocab-size 256 --context-length 16 --num-leads 12
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def verify_prior_export(
    build_fn,
    *,
    vocab_size: int,
    context_length: int,
    num_leads: int,
    embed_dim: int,
    num_layers: int,
    kernel_size: int,
    name: str,
    rep_data: np.ndarray,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Attempt INT8 TFLite export of a prior model and report results.

    Args:
        build_fn: Builder function for the prior.
        vocab_size: Codebook size.
        context_length: Temporal context length.
        num_leads: Number of leads.
        embed_dim: Embedding dimension.
        num_layers: Number of conv layers.
        kernel_size: Kernel size.
        name: Model name.
        rep_data: Representative dataset for quantization calibration.
        output_dir: Where to write the .tflite file (temp if None).

    Returns:
        Dict with export_success, model_size_bytes, num_params, and any errors.
    """
    from compressionkit.export.tflite import _convert_and_export

    model = build_fn(
        vocab_size=vocab_size,
        context_length=context_length,
        num_leads=num_leads,
        embed_dim=embed_dim,
        num_layers=num_layers,
        kernel_size=kernel_size,
    )
    num_params = model.count_params()

    result: dict[str, Any] = {
        "name": name,
        "num_params": num_params,
        "export_success": False,
        "model_size_bytes": 0,
        "error": None,
    }

    try:
        if output_dir is None:
            tmp = tempfile.mkdtemp()
            out = Path(tmp)
        else:
            out = output_dir
            out.mkdir(parents=True, exist_ok=True)

        tflite_path, _ = _convert_and_export(
            model,
            rep_dataset=rep_data,
            output_dir=out,
            tflite_name=f"{name}.tflite",
            header_name=f"{name}.h",
            c_array_name=name,
            quantization="INT8",
            io_type="int8",
        )
        result["export_success"] = True
        result["model_size_bytes"] = tflite_path.stat().st_size
        logger.info("%s: exported %d bytes (%d params)", name, result["model_size_bytes"], num_params)
    except Exception as e:
        result["error"] = str(e)
        logger.error("%s: export failed — %s", name, e)

    return result


def verify_all(
    *,
    vocab_size: int = 256,
    context_length: int = 16,
    num_leads: int = 12,
    embed_dim: int = 48,
    num_layers: int = 4,
    kernel_size: int = 5,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Verify all xlead prior variants export to INT8 LiteRT."""
    from compressionkit.generative.xlead_prior import (
        build_xlead_concat_prior,
        build_xlead_interleave_prior,
    )

    results = []

    # xlead_concat: input (B, T, L) int32
    rep_concat = np.random.default_rng(0).integers(0, vocab_size, size=(8, context_length, num_leads)).astype(np.int32)
    results.append(
        verify_prior_export(
            build_xlead_concat_prior,
            vocab_size=vocab_size,
            context_length=context_length,
            num_leads=num_leads,
            embed_dim=embed_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            name="xlead_concat",
            rep_data=rep_concat,
            output_dir=output_dir,
        )
    )

    # xlead_interleave: input (B, T*L) int32
    seq_len = context_length * num_leads
    rep_interleave = np.random.default_rng(1).integers(0, vocab_size, size=(8, seq_len)).astype(np.int32)
    results.append(
        verify_prior_export(
            build_xlead_interleave_prior,
            vocab_size=vocab_size,
            context_length=context_length,
            num_leads=num_leads,
            embed_dim=embed_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            name="xlead_interleave",
            rep_data=rep_interleave,
            output_dir=output_dir,
        )
    )

    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--context-length", type=int, default=16)
    ap.add_argument("--num-leads", type=int, default=12)
    ap.add_argument("--embed-dim", type=int, default=48)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--kernel-size", type=int, default=5)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    results = verify_all(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_leads=args.num_leads,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        output_dir=args.output_dir,
    )

    all_success = all(r["export_success"] for r in results)
    summary = {
        "all_exported": all_success,
        "models": results,
    }

    report_path = args.report or Path("results/xlead_litert_verification.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Wrote %s — all_exported=%s", report_path, all_success)

    if not all_success:
        for r in results:
            if not r["export_success"]:
                logger.error("FAILED: %s — %s", r["name"], r["error"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
