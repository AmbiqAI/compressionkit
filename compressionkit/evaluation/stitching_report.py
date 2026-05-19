"""Stitching A/B report — compare frame-stitching strategies for a codec.

Thin orchestration layer on top of :mod:`compressionkit.evaluation.stitching`
and :mod:`compressionkit.evaluation.metrics`. The goal is to make it trivial
to answer the customer-facing question:

    *"Does the choice of frame-stitching change the downstream metric we
    care about, and by how much?"*

The report is a tidy ``pandas.DataFrame`` with one row per
``(method, hop_ratio)`` combination and columns for PRD, seam discontinuity,
and (optionally) physio metrics. Designed to feed straight into a markdown
table or scorecard plot.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from compressionkit.evaluation.codec import Codec
from compressionkit.evaluation.metrics import compute_signal_metrics
from compressionkit.evaluation.stitching import (
    STITCH_METHODS,
    PredictFn,
    seam_discontinuity_ratio,
    stitch,
)

__all__ = [
    "codec_predict_fn",
    "compare_stitching_methods",
]


def codec_predict_fn(codec: Codec) -> PredictFn:
    """Adapt a :class:`Codec` to the ``(N, 1, T, 1) -> (N, 1, T, 1)`` API
    expected by :func:`compressionkit.evaluation.stitching.stitch`.

    Each frame in the batch is encoded then decoded independently. Single
    channel only (matches the stitching module's 1-D scope).
    """

    def _predict(batch_4d: np.ndarray) -> np.ndarray:
        if batch_4d.ndim != 4 or batch_4d.shape[1] != 1 or batch_4d.shape[3] != 1:
            raise ValueError(f"predict_fn expects shape (N, 1, T, 1); got {batch_4d.shape}")
        n, _, t, _ = batch_4d.shape
        out = np.empty_like(batch_4d, dtype=np.float32)
        for i in range(n):
            frame = batch_4d[i, 0, :, 0]
            enc = codec.encode(frame)
            dec = codec.decode(enc)
            out[i, 0, :, 0] = np.asarray(dec, dtype=np.float32).reshape(t)
        return out

    return _predict


def compare_stitching_methods(
    codec: Codec,
    signal: np.ndarray,
    *,
    methods: Iterable[str] | None = None,
    hop_ratios: Iterable[float] = (0.25, 0.5, 0.75),
    tukey_alpha: float = 0.25,
) -> pd.DataFrame:
    """Run a long-signal A/B over stitching methods × hop ratios.

    Args:
        codec: Any :class:`Codec`-compatible adapter (e.g. ``RvqCodec``,
            ``SpihtAcCodec``). Used via :func:`codec_predict_fn`.
        signal: Long 1-D signal to reconstruct. Must be at least
            ``2 * codec.frame_size`` samples long for seam metrics.
        methods: Iterable of method names from
            :data:`compressionkit.evaluation.stitching.STITCH_METHODS`.
            Defaults to all four.
        hop_ratios: Iterable of hop ratios to sweep. ``hard_concat`` ignores
            ``hop_ratio`` and is reported once.
        tukey_alpha: Taper fraction forwarded to ``tukey_overlap_add``.

    Returns:
        ``pandas.DataFrame`` with columns:
            * ``method`` (str)
            * ``hop_ratio`` (float; NaN for ``hard_concat``)
            * ``prd`` (percent)
            * ``seam_ratio`` (1.0 = invisible; >1 = visible seam)
            * ``seam_rms`` / ``non_seam_rms`` (raw)
            * ``num_seams`` (int)
            * ``n_samples`` (int)

    Raises:
        ValueError: If *signal* is too short for stitching.
    """
    sig = np.asarray(signal, dtype=np.float32).reshape(-1)
    frame_size = codec.frame_size
    if sig.size < 2 * frame_size:
        raise ValueError(f"signal length {sig.size} is too short for frame_size {frame_size}; need >= {2 * frame_size}")

    method_names = list(methods) if methods is not None else list(STITCH_METHODS)
    unknown = [m for m in method_names if m not in STITCH_METHODS]
    if unknown:
        raise ValueError(f"Unknown stitching methods: {unknown}")

    predict_fn = codec_predict_fn(codec)
    rows: list[dict[str, float | str | int]] = []

    for method in method_names:
        if method == "hard_concat":
            sweep = [float("nan")]
        else:
            sweep = list(hop_ratios)

        for hop_ratio in sweep:
            kwargs: dict[str, float] = {}
            if not np.isnan(hop_ratio):
                kwargs["hop_ratio"] = float(hop_ratio)
            if method == "tukey_overlap_add":
                kwargs["alpha"] = float(tukey_alpha)

            recon = stitch(method, predict_fn, sig, frame_size, **kwargs)
            prd = float(compute_signal_metrics(sig, recon)["prd_percent"])

            # Seam metric needs a concrete hop; default to frame_size for hard_concat.
            seam_hop_ratio = 1.0 if method == "hard_concat" else float(hop_ratio)
            seam = seam_discontinuity_ratio(recon, frame_size=frame_size, hop_ratio=seam_hop_ratio)

            rows.append(
                {
                    "method": method,
                    "hop_ratio": float(hop_ratio),
                    "prd": prd,
                    "seam_ratio": float(seam["ratio"]),
                    "seam_rms": float(seam["seam_rms"]),
                    "non_seam_rms": float(seam["non_seam_rms"]),
                    "num_seams": int(seam["num_seams"]),
                    "n_samples": int(sig.size),
                }
            )

    return pd.DataFrame(rows)
