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


def _fmt_agg(block: dict, *, precision: int = 2) -> str:
    """Format an aggregate block defensively."""
    if not isinstance(block, dict) or not block.get("n"):
        return "n/a"
    mean = block.get("mean")
    std = block.get("std")
    p90 = block.get("p90")
    if mean is None or std is None:
        return "n/a"
    if p90 is None:
        return f"{mean:.{precision}f} ± {std:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}  (p90 {p90:.{precision}f})"


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
    ap.add_argument(
        "--clean-reference",
        type=Path,
        default=None,
        help="Optional .npz/.npy bundle of aligned clean-reference windows for truth-aware evaluation.",
    )
    ap.add_argument(
        "--clean-reference-key",
        type=str,
        default=None,
        help="Optional array key when --clean-reference points to a .npz.",
    )
    ap.add_argument(
        "--clean-reference-label",
        type=str,
        default="clean_truth",
        help="Label stored in the output for the clean-reference bundle.",
    )
    ap.add_argument(
        "--adversarial-metrics",
        type=Path,
        default=None,
        help="Optional JSON artifact with hallucination/adversarial metrics to merge into the scorecard.",
    )
    ap.add_argument(
        "--imprinting-metrics",
        type=Path,
        default=None,
        help="Optional JSON artifact with localized imprinting metrics to merge into the scorecard.",
    )
    args = ap.parse_args()

    out = write_quality_scorecard(
        args.run_dir,
        modality=args.modality,
        sample_rate=args.sample_rate,
        noise_estimator=args.noise_estimator,
        min_signal_std=args.min_signal_std,
        clean_reference_path=args.clean_reference,
        clean_reference_key=args.clean_reference_key,
        clean_reference_label=args.clean_reference_label,
        adversarial_metrics_path=args.adversarial_metrics,
        imprinting_metrics_path=args.imprinting_metrics,
        output_path=args.output,
    )
    card = json.loads(out.read_text())
    td = card["time_domain"]
    sp = card["spectral"]
    bp = card.get("bitrate", {})
    phys = card.get("physiology", {})

    print(f"Wrote: {out}")
    hl = card.get("headline")
    if isinstance(hl, dict):
        print("  HEADLINE              :")

        def _hl(value: object, suffix: str = "") -> str:
            return f"{value}{suffix}" if value is not None else "n/a"

        print(f"    compression_ratio        = {_hl(hl.get('compression_ratio'), 'x')}")
        print(f"    faithful_prd_vs_input %  = {_hl(hl.get('faithful_prd_vs_input_pct'))}")
        print(f"    truth_prd_vs_clean %     = {_hl(hl.get('truth_prd_vs_clean_pct'))}")
        print(f"    truth_prd_native_noise % = {_hl(hl.get('truth_prd_at_native_noise_pct'))}")
        print(f"    prd_slope (PRD/dB)       = {_hl(hl.get('prd_degradation_slope_per_db'))}")
        print(f"    prd_at_0db / -6db %      = {_hl(hl.get('prd_at_0db_pct'))} / {_hl(hl.get('prd_at_-6db_pct'))}")
        print(f"    imprint_output_autocorr  = {_hl(hl.get('imprint_output_autocorr'))}")
    print(f"  num_samples           : {card['num_samples']}")
    if card.get("num_samples_rejected"):
        print(f"  num_samples_rejected  : {card['num_samples_rejected']}")
    print("  bitrate               :")
    for k in ("val_bits_per_token", "cr_codec_learned", "codec_compression_ratio"):
        if k in bp and bp[k] is not None:
            print(f"    {k:24s} = {bp[k]}")

    # Primary metrics: PRD + MSE/RMSE
    print("  PRIMARY (mean ± std)  :")
    print(f"    PRD %                = {_fmt_agg(td['prd_percent'], precision=2)}")
    print(f"    RMSE                 = {_fmt_agg(td['rmse'], precision=4)}")
    print(f"    cosine_similarity    = {_fmt_agg(td['cosine_similarity'], precision=4)}")

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
    print(f"    band_total_rel_error = {_fmt_agg(sp['band_total_rel_error'], precision=4)}")
    print(f"    weighted_freq_prd %  = {_fmt_agg(sp['weighted_freq_prd_percent'], precision=2)}")
    print(f"    coherence            = {_fmt_agg(sp['coherence'], precision=4)}")

    # Supplementary: PRDN-noise (interpret with care; see scripts/sanity_clean_ecg.py)
    if td["prdn_noise_percent"].get("n"):
        print("  SUPPLEMENTARY         :")
        print(
            f"    PRDN-noise %         = {td['prdn_noise_percent']['mean']:.2f} ± {td['prdn_noise_percent']['std']:.2f}"
        )
        print(f"      (noise estimator: {card['noise_estimator']}; compare against PRD on clean synthetic for context)")

    clean_ref = card.get("clean_reference")
    if isinstance(clean_ref, dict):
        base_td = clean_ref.get("input_baseline", {}).get("time_domain", {})
        out_td = clean_ref.get("reconstruction", {}).get("time_domain", {})
        denoise_td = clean_ref.get("denoising", {}).get("time_domain", {})
        print(f"  CLEAN REFERENCE ({clean_ref.get('label', 'clean_truth')}) :")
        if base_td.get("prd_percent", {}).get("n"):
            print(f"    input PRD vs clean    = {_fmt_agg(base_td['prd_percent'], precision=2)}")
        if out_td.get("prd_percent", {}).get("n"):
            print(f"    recon PRD vs clean    = {_fmt_agg(out_td['prd_percent'], precision=2)}")
        if denoise_td.get("prd_percent_improvement") is not None:
            print(f"    denoise delta PRD     = {denoise_td['prd_percent_improvement']:.2f}")
        if denoise_td.get("cosine_similarity_improvement") is not None:
            print(f"    denoise delta cosine  = {denoise_td['cosine_similarity_improvement']:.4f}")

    hallucination = card.get("hallucination")
    if isinstance(hallucination, dict):
        zero = hallucination.get("zero_input", {})
        print("  HALLUCINATION         :")
        for key in (
            "output_l2_when_input_zero",
            "hallucinated_peaks",
            "output_band_power",
            "output_energy",
        ):
            value = zero.get(key)
            if value is not None:
                print(f"    {key:32s} = {value:.6f}")

    imprinting = card.get("imprinting")
    if isinstance(imprinting, dict):
        metrics = imprinting.get("metrics", {})
        print("  IMPRINTING            :")
        gap_rate = metrics.get("gap_peak_rate")
        if gap_rate is not None:
            print(f"    {'gap_peak_rate':32s} = {gap_rate:.6f}")
        for key in (
            "local_energy_ratio",
            "local_cosine_to_target",
            "local_prd_percent",
            "masked_vs_clean_local_prd",
            "outside_prd_percent",
        ):
            block = metrics.get(key)
            if isinstance(block, dict) and block.get("mean") is not None:
                print(f"    {key:32s} = {_fmt_agg(block, precision=4)}")


if __name__ == "__main__":
    main()
