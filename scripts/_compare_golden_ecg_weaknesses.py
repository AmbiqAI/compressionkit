"""Aggregate quality_scorecard.json across the 6 golden ECG runs and surface
weaknesses (not just lowest MSE)."""

from __future__ import annotations

import json
from pathlib import Path

CRS = ["02", "04", "08", "16", "32", "64"]
ROOT = Path("results")


def load_card(cr: str) -> dict:
    return json.loads((ROOT / f"ecg_rvq_256hz_{cr}x_golden" / "quality_scorecard.json").read_text())


def fmt(v, prec=2):
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else str(v)


def section(title: str) -> None:
    print(f"\n### {title}")


def header(cols: list[str]) -> None:
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")


def row(vals: list) -> None:
    print("| " + " | ".join(str(v) for v in vals) + " |")


cards = {cr: load_card(cr) for cr in CRS}

# 1. Bitrate + headline
section("Bitrate + headline reconstruction (mean ± std, p90)")
header(["CR", "bpt", "CR_codec", "CR_learned", "PRD %", "PRD p90", "RMSE", "cosine"])
for cr in CRS:
    c = cards[cr]
    bp = c["bitrate"]
    td = c["time_domain"]
    row(
        [
            f"{cr}×",
            fmt(bp.get("val_bits_per_token"), 2),
            fmt(bp.get("codec_compression_ratio"), 1),
            fmt(bp.get("cr_codec_learned"), 2),
            f"{td['prd_percent']['mean']:.2f} ± {td['prd_percent']['std']:.2f}",
            fmt(td["prd_percent"]["p90"], 2),
            f"{td['rmse']['mean']:.4f}",
            f"{td['cosine_similarity']['mean']:.4f}",
        ]
    )

# 2. Spectral per-band
section("Per-band PSD relative error (mean) — where energy is lost")
header(["CR", "0.5–5 Hz", "5–15 Hz (R-peak)", "15–40 Hz (QRS detail)", "40–80 Hz (HF/noise)", "weighted-freq PRD %"])
for cr in CRS:
    sp = cards[cr]["spectral"]
    pb = sp["per_band_rel_error"]
    row(
        [
            f"{cr}×",
            f"{pb['band_0.5_5_rel_error']['mean']:.3f}",
            f"{pb['band_5_15_rel_error']['mean']:.3f}",
            f"{pb['band_15_40_rel_error']['mean']:.3f}",
            f"{pb['band_40_80_rel_error']['mean']:.3f}",
            f"{sp['weighted_freq_prd_percent']['mean']:.2f}",
        ]
    )

# 3. Physiology vs raw
section("HR + peak-timing vs RAW original (denoiser-penalising reference)")
header(["CR", "HR MAE", "HR p90", "HR std AE", "peak match %", "peak Δt MAE ms", "Δt p90 ms", "Δt within 10ms %"])
for cr in CRS:
    p = cards[cr]["physiology"]["vs_raw_original"]
    row(
        [
            f"{cr}×",
            f"{p['hr_mae_bpm']:.2f}",
            f"{p['hr_p90_ae_bpm']:.2f}",
            f"{p['hr_std_ae_bpm']:.2f}",
            f"{p['peak_count_exact_match_pct']:.1f}",
            f"{p['peak_timing_mae_ms']:.2f}",
            f"{p['peak_timing_p90_ms']:.2f}",
            f"{p['peak_timing_within_10ms_pct']:.1f}",
        ]
    )

# 4. Physiology vs filtered
section("HR + peak-timing vs FILTERED original (denoising-fair reference)")
header(["CR", "HR MAE", "HR p90", "peak Δt MAE ms", "Δt p90 ms", "Δt within 10ms %", "SDNN MAE ms", "RMSSD MAE ms"])
for cr in CRS:
    p = cards[cr]["physiology"]["vs_filtered_original"]
    row(
        [
            f"{cr}×",
            f"{p['hr_mae_bpm']:.2f}",
            f"{p['hr_p90_ae_bpm']:.2f}",
            f"{p['peak_timing_mae_ms']:.2f}",
            f"{p['peak_timing_p90_ms']:.2f}",
            f"{p['peak_timing_within_10ms_pct']:.1f}",
            f"{p['sdnn_mae_ms']:.2f}",
            f"{p['rmssd_mae_ms']:.2f}",
        ]
    )

# 5. Noise tertile breakdown
section("HR MAE by noise-tertile (vs filtered) — robustness to noise")
header(["CR", "clean (n)", "median (n)", "noisy (n)"])
for cr in CRS:
    tert = cards[cr]["physiology"].get("by_noise_tertile", {}).get("buckets", {})
    cells = []
    for name in ("clean", "median", "noisy"):
        b = tert.get(name, {})
        # The tertile aggregator stratifies vs_raw HR MAE; show what's there.
        hr = b.get("hr_mae_bpm", {})
        if hr.get("n"):
            cells.append(f"{hr['mean']:.2f} ± {hr['std']:.2f} (n={hr['n']})")
        else:
            cells.append("—")
    row([f"{cr}×", *cells])

# 6. Weakness summary: for each CR, where does it rank worst?
section("Weakness fingerprint per CR (highest values vs cleaner CRs)")
print("Looking for: where the worst-50% beat-timing tail lives, where SDNN/RMSSD blow up, which band dies first.")


# Compute per-CR worst-tail spreads
def get(cr, *path):
    x = cards[cr]
    for p in path:
        x = x[p]
    return x


print()
print("| metric | worst-CR onset | description |")
print("|---|---|---|")

# QRS-band PSD: when does 15-40 Hz error cross 10%?
for cr in CRS:
    e = get(cr, "spectral", "per_band_rel_error", "band_15_40_rel_error", "mean")
    if e > 0.10:
        print(f"| 15–40 Hz PSD error >10% | first at {cr}× ({e:.3f}) | QRS detail starts dropping out |")
        break

# 40-80 Hz: when does HF >50% (noise mostly removed)
for cr in CRS:
    e = get(cr, "spectral", "per_band_rel_error", "band_40_80_rel_error", "mean")
    if e > 0.50:
        print(
            f"| 40–80 Hz PSD error >50% | first at {cr}× ({e:.3f}) | noise band aggressively shaped (mostly desirable) |"
        )
        break

# Peak timing p90 > 10ms
for cr in CRS:
    p = get(cr, "physiology", "vs_filtered_original", "peak_timing_p90_ms")
    if p > 10.0:
        print(
            f"| peak-timing p90 >10 ms (vs filtered) | first at {cr}× ({p:.1f}) | tail of beats whose R-peaks shift by ≥1 sample |"
        )
        break

# RMSSD MAE >5ms (HRV vagal tone affected)
for cr in CRS:
    r = get(cr, "physiology", "vs_filtered_original", "rmssd_mae_ms")
    if r > 5.0:
        print(f"| RMSSD MAE >5 ms (vs filtered) | first at {cr}× ({r:.2f}) | beat-to-beat HRV starting to wobble |")
        break

# HR p90 > 5 bpm = bad outliers exist
for cr in CRS:
    h = get(cr, "physiology", "vs_filtered_original", "hr_p90_ae_bpm")
    if h > 5.0:
        print(
            f"| HR error p90 >5 bpm (vs filtered) | first at {cr}× ({h:.2f}) | 10% of windows have a clinically-noticeable HR error |"
        )
        break
