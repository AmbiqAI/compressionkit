"""Reusable codec robustness scoring block.

Wearable deployment is noisier than the clinical datasets these codecs are
trained/evaluated on (PTB-XL, BIDMC, ...). This module characterizes how each
codec *degrades* as the input is pushed into heavier contamination, so the
scorecard can report wearable-grade robustness rather than only clean-data
fidelity.

The module is deliberately modality- and codec-agnostic: callers build a list
of :class:`RobustnessCondition` (each a named, pre-contaminated input batch with
a known noise level) plus optional :class:`ImprintProbe` objects, and the engine
runs every condition through the codec and scores reconstruction vs the *clean
truth* proxy. Per-modality input construction (empirical-SNR ladder, additive
artifact families, pure-noise probes) lives in the evaluator script.

Reference convention: every condition is scored against the same ``clean`` array
(the filtered clean-truth proxy), so PRD/correlation are comparable across the
empirical-SNR ladder and the additive artifact families.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "RobustnessCondition",
    "ImprintProbe",
    "evaluate_robustness",
    "summarize_robustness_for_scorecard",
]


@dataclass
class RobustnessCondition:
    """A single named, pre-contaminated input batch scored against clean truth.

    Attributes:
        label: Stable identifier, e.g. ``"snr_0db"`` or ``"motion@0.50"``.
        group: Coarse axis: ``"empirical_snr"``, ``"artifact"`` or ``"reference"``.
        inputs: ``(N, L)`` codec input (already normalized), one row per clean frame.
        level: SNR in dB for ``empirical_snr``; severity in ``[0, 1]`` for
            ``artifact``; ``None`` for reference columns (clean / native).
        family: Artifact family name when ``group == "artifact"``, else ``None``.
    """

    label: str
    group: str
    inputs: np.ndarray
    level: float | None = None
    family: str | None = None


@dataclass
class ImprintProbe:
    """A pure-noise / pure-artifact batch (no underlying signal).

    A safe codec should *not* invent physiological structure from noise. The
    engine pushes ``inputs`` through the codec and measures invented periodicity
    via ``autocorr_fn`` plus the residual reconstruction energy.

    Attributes:
        label: Stable identifier, e.g. ``"pure_noise"`` or ``"pure_motion@0.75"``.
        inputs: ``(M, L)`` noise-only batch.
        autocorr_fn: Maps ``(recon, sample_rate)`` to a per-frame band-limited
            autocorrelation peak (RR band for ECG, pulse band for PPG). High =
            hallucinated rhythm.
    """

    label: str
    inputs: np.ndarray
    autocorr_fn: Callable[[np.ndarray, float], np.ndarray]


def _prd(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(ref - est, axis=-1)
    den = np.linalg.norm(ref, axis=-1) + 1e-12
    return 100.0 * num / den


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1)) + 1e-12
    return num / den


def _encode_decode_batch(codec: Any, frames: np.ndarray) -> np.ndarray:
    """Run a codec over a batch, normalizing each reconstruction like the evals."""
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        enc = codec.encode(f)
        rec = codec.decode(enc)
        rec = np.asarray(rec, dtype=np.float32).reshape(-1)[: frames.shape[1]]
        rec = (rec - rec.mean()) / (rec.std() + 1e-9)
        out[i] = rec
    return out


def _aggregate(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
    }


def _crossing_snr(snr_db: list[float], prd: list[float], threshold: float) -> float | None:
    """Input SNR (dB) at which output PRD first crosses ``threshold``.

    Ladder is given cleanest-first. Returns the interpolated SNR where PRD == the
    threshold (lower => the codec stays usable into noisier inputs). ``None`` if
    PRD never reaches the threshold across the tested ladder; a value below the
    noisiest tested SNR if PRD is already above threshold everywhere.
    """
    pts = sorted(zip(snr_db, prd, strict=True), key=lambda t: -t[0])  # cleanest first
    if not pts:
        return None
    if all(p < threshold for _s, p in pts):
        return None  # never degrades past threshold in the tested range
    if all(p >= threshold for _s, p in pts):
        return float(pts[0][0])  # already bad at the cleanest tested point
    for (s_hi, p_hi), (s_lo, p_lo) in zip(pts[:-1], pts[1:], strict=False):
        if (p_hi - threshold) * (p_lo - threshold) <= 0 and p_lo != p_hi:
            frac = (threshold - p_hi) / (p_lo - p_hi)
            return float(s_hi + frac * (s_lo - s_hi))
    return None


def evaluate_robustness(
    codec: Any,
    clean: np.ndarray,
    conditions: list[RobustnessCondition],
    *,
    sample_rate: float,
    imprint: list[ImprintProbe] | None = None,
    heavy_snr_db: tuple[float, ...] = (0.0, -6.0),
    prd_threshold: float = 10.0,
    morphology_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Score a codec's degradation across contaminated conditions.

    Args:
        codec: Object exposing ``encode(frame)`` / ``decode(encoded)``.
        clean: ``(N, L)`` clean-truth proxy; every condition is scored against it.
        conditions: Pre-built contaminated input batches (see builders in the
            evaluator). Each ``inputs`` array must be ``(N, L)``.
        sample_rate: Hz.
        imprint: Optional pure-noise probes for the hallucination check.
        heavy_snr_db: SNR levels highlighted in the headline (interpolated).
        prd_threshold: PRD% used for the ``robust_snr_db`` crossing metric.
        morphology_fn: Optional ``(clean, recon) -> {metric: value}`` callback.
            When provided, derived-physiology morphology metrics (e.g. the PPG
            AC-amplitude / pulse-shape / dicrotic-notch SpO2 proxy) are computed
            per reference and empirical-SNR condition and summarized in the
            ``morphology`` headline so SpO2-readiness can be tracked vs noise.

    Returns:
        A nested dict with ``reference``, ``empirical_snr``, ``artifacts``,
        ``imprint`` and a ``headline`` summary.
    """
    clean = np.asarray(clean, dtype=np.float32)
    record: dict[str, Any] = {
        "sample_rate": float(sample_rate),
        "n_windows": int(clean.shape[0]),
        "prd_threshold": float(prd_threshold),
        "reference": {},
        "empirical_snr": [],
        "artifacts": {},
        "imprint": {},
        "headline": {},
    }

    ladder: list[tuple[float, float]] = []  # (snr_db, prd_mean)
    morph_ladder: dict[str, list[tuple[float, float]]] = {}  # metric -> [(snr_db, value)]
    clean_morph: dict[str, float] | None = None
    for cond in conditions:
        recon = _encode_decode_batch(codec, cond.inputs)
        prd = _prd(clean, recon)
        corr = _corr(clean, recon)
        entry = {
            "label": cond.label,
            "level": cond.level,
            "prd": _aggregate(prd),
            "corr_mean": float(np.nanmean(corr)),
        }
        # Derived-physiology morphology (SpO2 proxy) on signal-bearing conditions.
        if morphology_fn is not None and cond.group in ("reference", "empirical_snr"):
            try:
                morph = morphology_fn(clean, recon)
            except Exception:  # noqa: BLE001 - morphology is best-effort
                morph = {}
            if morph:
                entry["morphology"] = morph
                if cond.label == "clean":
                    clean_morph = morph
                if cond.group == "empirical_snr" and cond.level is not None:
                    for key, val in morph.items():
                        if isinstance(val, (int, float)) and np.isfinite(val):
                            morph_ladder.setdefault(key, []).append((float(cond.level), float(val)))
        if cond.group == "reference":
            record["reference"][cond.label] = entry
        elif cond.group == "empirical_snr":
            record["empirical_snr"].append(entry)
            if cond.level is not None and np.isfinite(entry["prd"]["mean"]):
                ladder.append((float(cond.level), float(entry["prd"]["mean"])))
        elif cond.group == "artifact":
            fam = cond.family or "unknown"
            record["artifacts"].setdefault(fam, []).append(entry)
        else:  # pragma: no cover - defensive
            record.setdefault("other", {})[cond.label] = entry

    # Headline: empirical-SNR degradation curve summary.
    if ladder:
        ladder.sort(key=lambda t: -t[0])  # cleanest first
        snr_axis = [s for s, _ in ladder]
        prd_axis = [p for _, p in ladder]
        headline: dict[str, Any] = {}
        for tgt in heavy_snr_db:
            headline[f"prd_at_{tgt:g}db"] = float(np.interp(tgt, snr_axis[::-1], prd_axis[::-1]))
        # Degradation slope: ΔPRD per −1 dB (least-squares over the ladder).
        if len(ladder) >= 2:
            slope = float(np.polyfit(np.array(snr_axis), np.array(prd_axis), 1)[0])
            headline["prd_slope_per_db"] = -slope  # PRD rise per dB of SNR loss
        headline["robust_snr_db_at_prd_threshold"] = _crossing_snr(snr_axis, prd_axis, prd_threshold)
        record["headline"]["empirical_snr"] = headline

    # Headline: worst additive artifact at the mid severity.
    if record["artifacts"]:
        worst_family = None
        worst_prd = -1.0
        per_family_mid: dict[str, float] = {}
        for fam, rows in record["artifacts"].items():
            mid = min(rows, key=lambda r: abs((r["level"] or 0.0) - 0.5))
            per_family_mid[fam] = mid["prd"]["mean"]
            if np.isfinite(mid["prd"]["mean"]) and mid["prd"]["mean"] > worst_prd:
                worst_prd, worst_family = mid["prd"]["mean"], fam
        record["headline"]["artifacts"] = {
            "prd_by_family_mid_severity": per_family_mid,
            "worst_family": worst_family,
            "worst_prd_mid_severity": worst_prd if worst_family else float("nan"),
        }

    # Imprint probes.
    if imprint:
        for probe in imprint:
            recon = _encode_decode_batch(codec, probe.inputs)
            in_ac = probe.autocorr_fn(probe.inputs, sample_rate)
            out_ac = probe.autocorr_fn(recon, sample_rate)
            residual_rms = float(np.sqrt(np.mean(recon**2)))
            record["imprint"][probe.label] = {
                "input_autocorr_mean": float(np.nanmean(in_ac)),
                "output_autocorr_mean": float(np.nanmean(out_ac)),
                "residual_rms": residual_rms,
            }
        # Headline imprint = worst (max) invented periodicity across probes.
        worst = max(
            record["imprint"].values(),
            key=lambda d: d["output_autocorr_mean"]
            if np.isfinite(d["output_autocorr_mean"])
            else -1.0,
        )
        record["headline"]["imprint_output_autocorr_max"] = worst["output_autocorr_mean"]

    # Headline: derived-physiology morphology (SpO2 readiness) vs noise.
    if clean_morph or morph_ladder:
        morph_headline: dict[str, Any] = {}
        if clean_morph:
            morph_headline["clean"] = clean_morph
        for tgt in heavy_snr_db:
            at_level: dict[str, float] = {}
            for key, pairs in morph_ladder.items():
                if len(pairs) < 1:
                    continue
                pairs_sorted = sorted(pairs, key=lambda t: t[0])  # ascending SNR
                snr_axis = [s for s, _ in pairs_sorted]
                val_axis = [v for _, v in pairs_sorted]
                at_level[key] = float(np.interp(tgt, snr_axis, val_axis))
            if at_level:
                morph_headline[f"at_{tgt:g}db"] = at_level
        if morph_headline:
            record["headline"]["morphology"] = morph_headline

    return record


def summarize_robustness_for_scorecard(record: dict[str, Any]) -> dict[str, Any]:
    """Compact, customer-facing robustness block for ``quality_scorecard.json``.

    Keeps the full degradation ladder small: the headline numbers, the per-SNR
    PRD curve, the per-family mid-severity PRD, and the imprint summary.
    """
    if not isinstance(record, dict):
        return {}
    ladder = [
        {
            "snr_db": e.get("level"),
            "prd_mean": e["prd"]["mean"],
            "prd_median": e["prd"]["median"],
            **({"morphology": e["morphology"]} if isinstance(e.get("morphology"), dict) else {}),
        }
        for e in record.get("empirical_snr", [])
        if isinstance(e, dict) and "prd" in e
    ]
    out: dict[str, Any] = {
        "n_windows": record.get("n_windows"),
        "prd_threshold": record.get("prd_threshold"),
        "headline": record.get("headline", {}),
        "empirical_snr_curve": ladder,
        "reference": {
            k: v.get("prd", {}).get("mean")
            for k, v in record.get("reference", {}).items()
            if isinstance(v, dict)
        },
    }
    # Morphology (SpO2 proxy) reference values, when present.
    ref_morph = {
        k: v["morphology"]
        for k, v in record.get("reference", {}).items()
        if isinstance(v, dict) and isinstance(v.get("morphology"), dict)
    }
    if ref_morph:
        out["reference_morphology"] = ref_morph
    artifacts = record.get("artifacts", {})
    if artifacts:
        out["artifacts"] = {
            fam: [
                {"severity": r.get("level"), "prd_mean": r["prd"]["mean"]}
                for r in rows
                if isinstance(r, dict) and "prd" in r
            ]
            for fam, rows in artifacts.items()
        }
    if record.get("imprint"):
        out["imprint"] = record["imprint"]
    return out
