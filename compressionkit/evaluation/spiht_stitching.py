"""Long-recording stitching evaluation for SPIHT (DSP-only) codecs.

Mirrors the output shape of :func:`evaluate_stitching` (which targets
trained Keras models) so the same scorecard reader can consume the
result. SPIHT has no learned state, so the per-frame call collapses to
``codec.encode → codec.decode`` instead of a batched Keras predict.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from compressionkit.evaluation.codec import Codec
from compressionkit.evaluation.metrics import compute_signal_metrics
from compressionkit.evaluation.stitching import (
    STITCH_METHODS,
    seam_discontinuity_ratio,
    stitch,
)

logger = logging.getLogger(__name__)

SignalLoader = Callable[[], Iterable[np.ndarray]]


def _codec_predict_fn(codec: Codec):
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


def evaluate_spiht_stitching(
    codec: Codec,
    signals: Iterable[np.ndarray],
    *,
    methods: list[str],
    hop_ratio: float = 0.5,
    sample_rate: int | None = None,
    seam_radius: int = 4,
    max_signals: int | None = None,
    normalize_per_frame: bool = True,
) -> dict[str, Any]:
    """Stitch each method over a set of long signals and aggregate metrics.

    Args:
        codec: Any :class:`Codec` (e.g. :class:`SpihtAcCodec`).
        signals: Iterable of 1-D float32 signals already at the target
            sample rate. Each must be at least ``2 * codec.frame_size`` long.
        methods: Subset of :data:`STITCH_METHODS` to evaluate.
        hop_ratio: Hop ratio for overlap-based methods (ignored by
            ``hard_concat``).
        sample_rate: Optional, recorded in the report for downstream use.
        seam_radius: Sample radius around each seam used for the
            seam-discontinuity ratio metric.
        max_signals: Cap on number of signals processed (after filtering
            for length); ``None`` processes all.
        normalize_per_frame: If True, z-normalize the *signal* once before
            stitching so it matches the per-frame z-norm used by the
            standalone golden runner. The codec still sees frames as the
            stitcher carves them; this just keeps amplitudes in the same
            range as the per-frame eval.

    Returns:
        Dict with the same shape as
        :func:`compressionkit.evaluation.ecg_stitching.evaluate_stitching`::

            {
              "duration_sec": float,
              "hop_ratio": float,
              "frame_size": int,
              "num_recordings_eval": int,
              "methods": {
                method_name: {
                  "num_recordings": int,
                  "mean_prd_percent": float,
                  "mean_cosine_similarity": float,
                  "mean_mse": float,
                  "mean_seam_ratio": float,
                  "mean_seam_rms": float,
                  "mean_non_seam_rms": float,
                }
              }
            }
    """
    unknown = [m for m in methods if m not in STITCH_METHODS]
    if unknown:
        raise ValueError(f"Unknown stitching methods: {unknown}. Known: {sorted(STITCH_METHODS)}")

    frame_size = codec.frame_size
    predict_fn = _codec_predict_fn(codec)

    _keys = ["prd_percent", "cosine_similarity", "mse", "seam_ratio", "seam_rms", "non_seam_rms"]
    per_method: dict[str, dict[str, list[float]]] = {m: {k: [] for k in _keys} for m in methods}
    n_used = 0
    total_samples = 0

    for sig in signals:
        s = np.asarray(sig, dtype=np.float32).reshape(-1)
        if s.size < 2 * frame_size:
            continue
        if normalize_per_frame:
            std = float(np.std(s))
            if std < 1e-3:
                continue
            s = (s - float(np.mean(s))) / (std + 1e-6)
        total_samples += int(s.size)
        for method in methods:
            kwargs: dict[str, Any] = {}
            if method != "hard_concat":
                kwargs["hop_ratio"] = float(hop_ratio)
            recon = stitch(method, predict_fn, s, frame_size, **kwargs)
            m = compute_signal_metrics(s, recon)
            effective_hop = 1.0 if method == "hard_concat" else float(hop_ratio)
            seam = seam_discontinuity_ratio(recon, frame_size=frame_size, hop_ratio=effective_hop, radius=seam_radius)
            per_method[method]["prd_percent"].append(float(m["prd_percent"]))
            per_method[method]["cosine_similarity"].append(float(m["cosine_similarity"]))
            per_method[method]["mse"].append(float(m["mse"]))
            if np.isfinite(seam["ratio"]):
                per_method[method]["seam_ratio"].append(float(seam["ratio"]))
                per_method[method]["seam_rms"].append(float(seam["seam_rms"]))
                per_method[method]["non_seam_rms"].append(float(seam["non_seam_rms"]))
        n_used += 1
        if max_signals is not None and n_used >= max_signals:
            break

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    method_summary: dict[str, dict[str, float | int]] = {}
    for m, acc in per_method.items():
        method_summary[m] = {
            "num_recordings": len(acc["prd_percent"]),
            "mean_prd_percent": _mean(acc["prd_percent"]),
            "mean_cosine_similarity": _mean(acc["cosine_similarity"]),
            "mean_mse": _mean(acc["mse"]),
            "mean_seam_ratio": _mean(acc["seam_ratio"]),
            "mean_seam_rms": _mean(acc["seam_rms"]),
            "mean_non_seam_rms": _mean(acc["non_seam_rms"]),
        }

    duration_sec = float(total_samples / sample_rate / max(n_used, 1)) if sample_rate else float("nan")
    return {
        "duration_sec": duration_sec,
        "hop_ratio": float(hop_ratio),
        "frame_size": int(frame_size),
        "num_recordings_eval": int(n_used),
        "sample_rate": int(sample_rate) if sample_rate else None,
        "methods": method_summary,
    }


__all__ = ["evaluate_spiht_stitching"]
