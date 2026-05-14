"""Compare cross-lead token structure for 12-lead encoder strategies.

Strategy A: Reuse single-lead encoder applied independently per lead (per_lead=True).
Strategy B: Joint 12-lead codec that sees all channels simultaneously.

Outputs a JSON with per-lead token entropy, cross-lead mutual information
estimates, and basic frame-level statistics.

Usage:
    python scripts/compare_12lead_strategies.py \
        --single-lead-model results/ecg_rvq_256hz_32x_golden \
        --joint-model results/ecg_rvq_256hz_08x_ds8_l2_filt_12lead \
        --data-dir datasets/ptbxl_12lead \
        --frame-size 256 \
        --max-signals 100
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _token_entropy(tokens: np.ndarray) -> float:
    """Shannon entropy in bits of a flat token sequence."""
    flat = tokens.ravel()
    if flat.size == 0:
        return 0.0
    _, counts = np.unique(flat, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _cross_lead_mi(lead_a: np.ndarray, lead_b: np.ndarray, vocab_size: int) -> float:
    """Estimate MI between two lead token streams via joint histogram."""
    if lead_a.size == 0:
        return 0.0
    joint = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for a, b in zip(lead_a.ravel(), lead_b.ravel()):
        joint[a, b] += 1
    joint /= joint.sum() + 1e-12
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pointwise = np.where(joint > 0, joint * np.log2(joint / (pa * pb + 1e-12) + 1e-12), 0.0)
    return float(pointwise.sum())


def compare_strategies(
    tokens_independent: np.ndarray,
    tokens_joint: np.ndarray | None,
    *,
    vocab_size: int = 256,
) -> dict:
    """Compute cross-lead comparison metrics.

    Args:
        tokens_independent: (num_leads, N, tpf, levels) from per-lead extraction.
        tokens_joint: (N, tpf, levels) or None if joint model not available.
        vocab_size: Codebook size for MI estimation.

    Returns:
        Dict with per-lead entropy, mean cross-lead MI, and optionally
        the same for the joint-model token stream.
    """
    num_leads = tokens_independent.shape[0]
    # Flatten to per-lead 1-D streams for level 0
    per_lead_streams = [tokens_independent[l, :, :, 0].ravel() for l in range(num_leads)]

    per_lead_entropy = [_token_entropy(s) for s in per_lead_streams]

    # Pairwise MI among leads
    mi_values = []
    for i in range(num_leads):
        for j in range(i + 1, num_leads):
            mi_values.append(_cross_lead_mi(per_lead_streams[i], per_lead_streams[j], vocab_size))

    result: dict = {
        "strategy": "independent_per_lead",
        "num_leads": num_leads,
        "per_lead_entropy_bits": per_lead_entropy,
        "mean_entropy_bits": float(np.mean(per_lead_entropy)),
        "mean_cross_lead_mi_bits": float(np.mean(mi_values)) if mi_values else 0.0,
        "max_cross_lead_mi_bits": float(np.max(mi_values)) if mi_values else 0.0,
    }

    if tokens_joint is not None:
        joint_entropy = _token_entropy(tokens_joint[:, :, 0])
        result["joint_strategy"] = {
            "token_entropy_bits": joint_entropy,
            "num_frames": int(tokens_joint.shape[0]),
        }

    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--single-lead-model", type=Path, required=True)
    ap.add_argument("--joint-model", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--frame-size", type=int, default=256)
    ap.add_argument("--max-signals", type=int, default=100)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    # Defer heavy imports until we actually run
    from compressionkit.generative.token_extraction import extract_rvq_tokens
    from compressionkit.models import load_autoencoder

    # Load signals
    signal_files = sorted(args.data_dir.glob("*.npy"))[: args.max_signals]
    if not signal_files:
        logger.error("No .npy files found in %s", args.data_dir)
        return 1
    signals = [np.load(f) for f in signal_files]
    logger.info("Loaded %d signals from %s", len(signals), args.data_dir)

    # Strategy A: single-lead encoder applied per lead

    model_a = load_autoencoder(args.single_lead_model)
    tokens_a = extract_rvq_tokens(model_a, signals, frame_size=args.frame_size, per_lead=True, batch_size=32)
    logger.info("Strategy A tokens shape: %s", tokens_a.shape)

    # Strategy B: joint 12-lead model (if available)
    tokens_b = None
    if args.joint_model and args.joint_model.exists():
        model_b = load_autoencoder(args.joint_model)
        tokens_b = extract_rvq_tokens(model_b, signals, frame_size=args.frame_size, per_lead=False, batch_size=32)
        logger.info("Strategy B tokens shape: %s", tokens_b.shape)

    result = compare_strategies(tokens_a, tokens_b, vocab_size=args.vocab_size)

    out = args.output or Path("results/12lead_strategy_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
