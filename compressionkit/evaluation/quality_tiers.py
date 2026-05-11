"""ECG compression quality tiers for consumer and clinical applications.

Defines three quality tiers with pass/fail thresholds for ECG compression:

- **Diagnostic**: Clinical diagnosis — morphology, ST-segment, and arrhythmia
  classification must be preserved. Targets cardiologist-grade fidelity.
- **Monitoring**: Continuous clinical monitoring — ICU, Holter, real-time
  arrhythmia alerting. Heart rate and rhythm accuracy are critical;
  fine morphological detail may be relaxed.
- **Wellness**: Consumer wearables — heart-rate trends, basic rhythm awareness,
  stress/recovery tracking. Tolerates visible waveform distortion as long as
  beat detection remains reliable.

Threshold sources:
  - PRD bounds follow Zigel et al. (2000) quality scale and common literature
    consensus (PRD < 9 % ≈ "very good", < 15 % ≈ "good").
  - HR accuracy aligns with IEC 80601-2-61 (pulse oximetry HR: ±3 BPM)
    tightened for diagnostic use.
  - Peak-timing and HRV bounds are empirically derived from our PTB-XL
    256 Hz compression sweep (4×–32× CR).
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------


class QualityTier(enum.StrEnum):
    """Named quality tier for ECG compression."""

    DIAGNOSTIC = "diagnostic"
    MONITORING = "monitoring"
    WELLNESS = "wellness"


# ---------------------------------------------------------------------------
# Threshold spec
# ---------------------------------------------------------------------------


class TierThresholds(BaseModel):
    """Pass/fail thresholds for a single quality tier.

    All fields use *upper-bound-is-bad* convention: a metric value **at or
    below** the threshold passes (except ``cosine_min`` and ``peak_within_10ms_pct``
    which are lower bounds).
    """

    # -- waveform fidelity --
    prd_max: float = Field(description="Maximum PRD (%) — waveform distortion budget.")
    cosine_min: float = Field(description="Minimum cosine similarity (1.0 = perfect).")

    # -- heart-rate accuracy --
    hr_mae_max_bpm: float = Field(description="Maximum HR MAE (BPM).")

    # -- peak / R-wave timing --
    peak_timing_mae_max_ms: float = Field(description="Maximum R-peak timing MAE (ms).")
    peak_within_10ms_pct: float = Field(
        description="Minimum % of matched peaks within ±10 ms (0–100)."
    )

    # -- HRV preservation --
    sdnn_mae_max_ms: float = Field(description="Maximum SDNN MAE (ms).")
    rmssd_mae_max_ms: float = Field(description="Maximum RMSSD MAE (ms).")


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

TIER_THRESHOLDS: dict[QualityTier, TierThresholds] = {
    QualityTier.DIAGNOSTIC: TierThresholds(
        prd_max=9.0,
        cosine_min=0.995,
        hr_mae_max_bpm=1.0,
        peak_timing_mae_max_ms=8.0,
        peak_within_10ms_pct=95.0,
        sdnn_mae_max_ms=5.0,
        rmssd_mae_max_ms=5.0,
    ),
    QualityTier.MONITORING: TierThresholds(
        prd_max=15.0,
        cosine_min=0.985,
        hr_mae_max_bpm=3.0,
        peak_timing_mae_max_ms=15.0,
        peak_within_10ms_pct=90.0,
        sdnn_mae_max_ms=10.0,
        rmssd_mae_max_ms=15.0,
    ),
    QualityTier.WELLNESS: TierThresholds(
        prd_max=25.0,
        cosine_min=0.970,
        hr_mae_max_bpm=5.0,
        peak_timing_mae_max_ms=30.0,
        peak_within_10ms_pct=80.0,
        sdnn_mae_max_ms=20.0,
        rmssd_mae_max_ms=25.0,
    ),
}

# Ordered strictest → most relaxed for grading.
_TIER_ORDER: list[QualityTier] = [
    QualityTier.DIAGNOSTIC,
    QualityTier.MONITORING,
    QualityTier.WELLNESS,
]


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


class MetricValues(BaseModel):
    """Observed metric values for a model or experiment."""

    prd: float = Field(description="PRD (%)")
    cosine: float = Field(description="Cosine similarity")
    hr_mae_bpm: float = Field(description="HR MAE (BPM)")
    peak_timing_mae_ms: float = Field(description="R-peak timing MAE (ms)")
    peak_within_10ms_pct: float = Field(description="% peaks within ±10 ms")
    sdnn_mae_ms: float | None = Field(default=None, description="SDNN MAE (ms)")
    rmssd_mae_ms: float | None = Field(default=None, description="RMSSD MAE (ms)")


class TierResult(BaseModel):
    """Result of grading a model against a single tier."""

    tier: QualityTier
    passed: bool
    failures: dict[str, str] = Field(
        default_factory=dict,
        description="Metric name → human-readable reason for each failing criterion.",
    )


class GradeResult(BaseModel):
    """Overall quality grade for a model."""

    achieved_tier: QualityTier | None = Field(
        description="Highest (strictest) tier passed, or None if none passed."
    )
    tier_results: dict[QualityTier, TierResult]


# ---------------------------------------------------------------------------
# Grading logic
# ---------------------------------------------------------------------------


def check_tier(metrics: MetricValues, tier: QualityTier) -> TierResult:
    """Check whether *metrics* pass all thresholds for *tier*."""
    t = TIER_THRESHOLDS[tier]
    failures: dict[str, str] = {}

    if metrics.prd > t.prd_max:
        failures["prd"] = f"{metrics.prd:.2f}% > {t.prd_max:.1f}%"
    if metrics.cosine < t.cosine_min:
        failures["cosine"] = f"{metrics.cosine:.4f} < {t.cosine_min:.3f}"
    if metrics.hr_mae_bpm > t.hr_mae_max_bpm:
        failures["hr_mae_bpm"] = f"{metrics.hr_mae_bpm:.2f} > {t.hr_mae_max_bpm:.1f} BPM"
    if metrics.peak_timing_mae_ms > t.peak_timing_mae_max_ms:
        failures["peak_timing_mae_ms"] = (
            f"{metrics.peak_timing_mae_ms:.1f}ms > {t.peak_timing_mae_max_ms:.0f}ms"
        )
    if metrics.peak_within_10ms_pct < t.peak_within_10ms_pct:
        failures["peak_within_10ms_pct"] = (
            f"{metrics.peak_within_10ms_pct:.0f}% < {t.peak_within_10ms_pct:.0f}%"
        )
    if metrics.sdnn_mae_ms is not None and metrics.sdnn_mae_ms > t.sdnn_mae_max_ms:
        failures["sdnn_mae_ms"] = f"{metrics.sdnn_mae_ms:.1f}ms > {t.sdnn_mae_max_ms:.0f}ms"
    if metrics.rmssd_mae_ms is not None and metrics.rmssd_mae_ms > t.rmssd_mae_max_ms:
        failures["rmssd_mae_ms"] = f"{metrics.rmssd_mae_ms:.1f}ms > {t.rmssd_mae_max_ms:.0f}ms"

    return TierResult(tier=tier, passed=len(failures) == 0, failures=failures)


def grade(metrics: MetricValues) -> GradeResult:
    """Grade *metrics* against all tiers, returning the highest achieved."""
    tier_results: dict[QualityTier, TierResult] = {}
    achieved: QualityTier | None = None

    for tier in _TIER_ORDER:
        result = check_tier(metrics, tier)
        tier_results[tier] = result
        if result.passed and achieved is None:
            achieved = tier  # First (strictest) passing tier wins.

    return GradeResult(achieved_tier=achieved, tier_results=tier_results)


def format_grade_report(name: str, metrics: MetricValues, result: GradeResult) -> str:
    """Return a human-readable grade report for one model."""
    lines: list[str] = []
    tag = result.achieved_tier.value.upper() if result.achieved_tier else "BELOW WELLNESS"
    lines.append(f"  {name}: {tag}")

    for tier in _TIER_ORDER:
        tr = result.tier_results[tier]
        status = "PASS" if tr.passed else "FAIL"
        lines.append(f"    {tier.value:12s}  {status}")
        for metric_name, reason in tr.failures.items():
            lines.append(f"      ✗ {metric_name}: {reason}")

    return "\n".join(lines)
