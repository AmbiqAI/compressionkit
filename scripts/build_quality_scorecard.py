"""CLI: build a quality scorecard for a trained codec run.

Example:
    python scripts/build_quality_scorecard.py \\
        results/ecg_rvq_256hz_32x_golden \\
        --modality ecg --sample-rate 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compressionkit.evaluation.scorecard import write_quality_scorecard


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="Path to a trained run directory.")
    ap.add_argument("--modality", choices=("ecg", "ppg"), required=True)
    ap.add_argument("--sample-rate", type=int, required=True)
    ap.add_argument(
        "--noise-estimator",
        choices=("bp", "hf", "qrs"),
        default="bp",
        help="Which noise-floor estimator drives PRDN-noise (default: bp).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to <run_dir>/quality_scorecard.json.",
    )
    ap.add_argument(
        "--min-signal-std",
        type=float,
        default=1.0e-4,
        help="Reject near-flat/corrupted sample windows below this std before aggregation.",
    )
    args = ap.parse_args()

    out = write_quality_scorecard(
        args.run_dir,
        modality=args.modality,
        sample_rate=args.sample_rate,
        noise_estimator=args.noise_estimator,
        min_signal_std=args.min_signal_std,
        output_path=args.output,
    )
    card = json.loads(out.read_text())
    td = card["time_domain"]
    sp = card["spectral"]
    bp = card.get("bitrate", {})
    phys = card.get("physiology", {})

    print(f"Wrote: {out}")
    print(f"  num_samples           : {card['num_samples']}")
    if card.get("num_samples_rejected"):
        print(f"  num_samples_rejected  : {card['num_samples_rejected']}")
    print("  bitrate               :")
    for k in ("val_bits_per_token", "cr_codec_learned", "codec_compression_ratio"):
        if k in bp and bp[k] is not None:
            print(f"    {k:24s} = {bp[k]}")

    # Primary metrics: PRD + MSE/RMSE
    print("  PRIMARY (mean ± std)  :")
    print(
        f"    PRD %                = {td['prd_percent']['mean']:.2f} ± {td['prd_percent']['std']:.2f}  (p90 {td['prd_percent']['p90']:.2f})"
    )
    print(f"    RMSE                 = {td['rmse']['mean']:.4f} ± {td['rmse']['std']:.4f}")
    print(f"    cosine_similarity    = {td['cosine_similarity']['mean']:.4f} ± {td['cosine_similarity']['std']:.4f}")

    # Domain-specific physiology (HR/HRV) is the primary clinical claim
    if phys:
        vsr = phys.get("vs_raw_original", phys)  # backwards-compat
        vsf = phys.get("vs_filtered_original", {})
        if vsr:
            print("  PHYSIOLOGY vs RAW orig:")
            for k, label in (
                ("hr_mae_bpm", "hr_mae_bpm"),
                ("hr_std_ae_bpm", "hr_std_ae_bpm"),
                ("hr_p90_ae_bpm", "hr_p90_ae_bpm"),
                ("hr_bias_bpm", "hr_bias_bpm"),
                ("peak_count_exact_match_pct", "peak_count_exact_match_pct"),
                ("peak_precision_pct", "peak_precision_pct"),
                ("peak_recall_pct", "peak_recall_pct"),
                ("peak_f1_pct", "peak_f1_pct"),
                ("peak_timing_mae_ms", "peak_timing_mae_ms"),
                ("peak_timing_p90_ms", "peak_timing_p90_ms"),
                ("peak_timing_within_10ms_pct", "peak_timing_within_10ms_pct"),
                ("ibi_mae_ms", "ibi_mae_ms"),
                ("sdnn_mae_ms", "sdnn_mae_ms"),
                ("rmssd_mae_ms", "rmssd_mae_ms"),
            ):
                if k in vsr and vsr[k] is not None:
                    print(f"    {label:32s} = {vsr[k]:.4f}")
            peak = vsr.get("peak_alignment") if isinstance(vsr, dict) else None
            if peak:
                print("  PPG PEAK ALIGNMENT:")
                for k in (
                    "peak_precision_pct",
                    "peak_recall_pct",
                    "peak_f1_pct",
                    "peak_timing_mae_ms",
                    "peak_timing_p90_ms",
                    "ibi_mae_ms",
                    "total_missed_peaks",
                    "total_extra_peaks",
                ):
                    if k in peak and peak[k] is not None:
                        print(f"    {k:32s} = {peak[k]:.4f}")
        if vsf:
            print("  PHYSIOLOGY vs FILTERED orig (denoising-fair reference):")
            for k in (
                "hr_mae_bpm",
                "hr_p90_ae_bpm",
                "peak_timing_mae_ms",
                "peak_timing_within_10ms_pct",
                "sdnn_mae_ms",
                "rmssd_mae_ms",
            ):
                if k in vsf and vsf[k] is not None:
                    print(f"    {k:32s} = {vsf[k]:.4f}")
        tert = phys.get("by_noise_tertile") if isinstance(phys, dict) else None
        if tert:
            print("  PHYSIOLOGY by noise tertile (HR MAE):")
            for name in ("clean", "median", "noisy"):
                b = tert["buckets"].get(name, {})
                hr = b.get("hr_mae_bpm", {})
                if hr.get("n"):
                    print(
                        f"    {name:8s} (n={hr['n']:3d}) hr_mae = {hr['mean']:.3f} ± {hr['std']:.3f}  p90 {hr['p90']:.3f}"
                    )

    # Frequency-domain reconstruction quality
    print("  SPECTRAL (mean ± std) :")
    print(
        f"    band_total_rel_error = {sp['band_total_rel_error']['mean']:.4f} ± {sp['band_total_rel_error']['std']:.4f}"
    )
    print(
        f"    weighted_freq_prd %  = {sp['weighted_freq_prd_percent']['mean']:.2f} ± {sp['weighted_freq_prd_percent']['std']:.2f}"
    )
    print(f"    coherence            = {sp['coherence']['mean']:.4f} ± {sp['coherence']['std']:.4f}")

    # Supplementary: PRDN-noise (interpret with care; see scripts/sanity_clean_ecg.py)
    if td["prdn_noise_percent"].get("n"):
        print("  SUPPLEMENTARY         :")
        print(
            f"    PRDN-noise %         = {td['prdn_noise_percent']['mean']:.2f} ± {td['prdn_noise_percent']['std']:.2f}"
        )
        print(f"      (noise estimator: {card['noise_estimator']}; compare against PRD on clean synthetic for context)")


if __name__ == "__main__":
    main()
