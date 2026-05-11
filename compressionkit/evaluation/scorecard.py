"""Customer-facing quality scorecard.

Aggregates time-domain, spectral, physiological, and stability metrics for
a single trained codec run into one JSON document. Designed to be runnable
post-hoc against existing ``results/...`` directories without retraining.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from compressionkit.evaluation.metrics import (
    compute_signal_metrics,
    summarize_physiokit_alignment,
    summarize_ecg_alignment,
    summarize_ppg_peak_alignment,
)
from compressionkit.evaluation.noise import (
    estimate_ecg_noise_floor,
    estimate_ppg_noise_floor,
)
from compressionkit.evaluation.spectral_metrics import (
    ECG_DEFAULT_BANDS,
    ECG_DEFAULT_COHERENCE_BAND,
    ECG_DEFAULT_FREQ_WEIGHTS,
    PPG_DEFAULT_BANDS,
    PPG_DEFAULT_COHERENCE_BAND,
    PPG_DEFAULT_FREQ_WEIGHTS,
    psd_band_error,
    spectral_coherence,
    weighted_freq_prd,
)


def _aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def _filter_signal_safe(
    signal: np.ndarray,
    *,
    fs: int,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase bandpass via physiokit; falls back to original on error."""
    try:
        import physiokit as pk

        return np.asarray(
            pk.signal.filter_signal(
                np.asarray(signal, dtype=np.float64),
                lowcut=float(lowcut),
                highcut=float(highcut),
                sample_rate=int(fs),
                order=int(order),
                forward_backward=True,
            ),
            dtype=np.float32,
        )
    except Exception:
        return np.asarray(signal, dtype=np.float32)


def _load_samples(run_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load (original, reconstructed) pairs from per-sample CSV artifacts."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for csv_path in sorted(run_dir.glob("sample_*.csv")):
        df = pd.read_csv(csv_path)
        if "original" not in df.columns or "reconstructed" not in df.columns:
            continue
        pairs.append((
            df["original"].to_numpy(dtype=np.float32),
            df["reconstructed"].to_numpy(dtype=np.float32),
        ))
    return pairs


def _read_summary(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _read_stitching(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "stitching_report.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _read_best_entropy(run_dir: Path) -> dict[str, Any]:
    """Pick the best (lowest val_bits_per_token) entropy_report.json."""
    root = run_dir / "entropy_prior"
    if not root.exists():
        return {}
    best: dict[str, Any] | None = None
    best_bpt = float("inf")
    for tag_dir in sorted(root.iterdir()):
        if not tag_dir.is_dir():
            continue
        rpt = tag_dir / "entropy_report.json"
        if not rpt.exists():
            continue
        try:
            d = json.loads(rpt.read_text())
        except Exception:
            continue
        bpt = float(d.get("metrics", {}).get("val_bits_per_token", float("inf")))
        if bpt < best_bpt:
            best_bpt = bpt
            best = {"tag": tag_dir.name, **d}
    return best or {}


def build_quality_scorecard(
    run_dir: Path,
    *,
    modality: str,
    sample_rate: int,
    bands: list[tuple[float, float]] | None = None,
    freq_weights: list[tuple[float, float, float]] | None = None,
    coherence_band: tuple[float, float] | None = None,
    noise_estimator: str = "bp",
    min_signal_std: float = 1.0e-4,
) -> dict[str, Any]:
    """Compute the full scorecard for a single run directory.

    Args:
        run_dir: Path to a trained codec run with ``sample_*.csv`` artifacts
            and a ``summary.json`` (and optionally ``stitching_report.json``
            and an ``entropy_prior/`` subtree).
        modality: ``"ecg"`` or ``"ppg"``.
        sample_rate: Hz.
        bands: Optional override for PSD band split.
        freq_weights: Optional override for weighted-frequency PRD weights.
        coherence_band: Optional override for the coherence integration band.
        noise_estimator: Which estimator drives PRDN-noise. One of
            ``"bp"`` (bandpass-residual), ``"hf"`` (high-frequency power),
            or ``"qrs"`` (R-peak-locked, ECG only). Default ``"bp"``.
        min_signal_std: Reject near-flat/corrupted sample windows below this
            original-signal standard deviation before computing aggregate
            scorecard metrics. Default ``1e-4``.

    Returns:
        Scorecard dict with sections ``bitrate``, ``time_domain``,
        ``spectral``, ``physiology``, ``stability``, ``context``, plus
        ``run_dir`` and ``modality``.
    """
    run_dir = Path(run_dir)
    modality = modality.lower()
    if modality not in ("ecg", "ppg"):
        raise ValueError(f"Unknown modality {modality!r}; expected 'ecg' or 'ppg'.")

    if bands is None:
        bands = list(ECG_DEFAULT_BANDS) if modality == "ecg" else list(PPG_DEFAULT_BANDS)
    if freq_weights is None:
        freq_weights = (
            list(ECG_DEFAULT_FREQ_WEIGHTS) if modality == "ecg"
            else list(PPG_DEFAULT_FREQ_WEIGHTS)
        )
    if coherence_band is None:
        coherence_band = (
            ECG_DEFAULT_COHERENCE_BAND if modality == "ecg"
            else PPG_DEFAULT_COHERENCE_BAND
        )

    loaded_pairs = _load_samples(run_dir)
    pairs = [
        (orig, recon)
        for orig, recon in loaded_pairs
        if np.isfinite(orig).all()
        and np.isfinite(recon).all()
        and float(np.std(orig)) >= min_signal_std
    ]
    summary = _read_summary(run_dir)
    stitching = _read_stitching(run_dir)
    entropy = _read_best_entropy(run_dir)

    # --- Time-domain + per-sample noise + PRDN-noise ---------------------
    prd_vals: list[float] = []
    prdn_vals: list[float] = []
    rmse_vals: list[float] = []
    cos_vals: list[float] = []
    noise_power_vals: list[float] = []
    noise_rms_vals: list[float] = []
    qrs_snr_vals: list[float] = []
    band_err_per_band: dict[str, list[float]] = {}
    band_total_errs: list[float] = []
    wfprd_vals: list[float] = []
    coh_vals: list[float] = []

    for orig, recon in pairs:
        if modality == "ecg":
            nf = estimate_ecg_noise_floor(orig, fs=sample_rate)
        else:
            nf = estimate_ppg_noise_floor(orig, fs=sample_rate)

        if noise_estimator == "hf":
            np_est = nf.get("hf_noise_power", 0.0)
        elif noise_estimator == "qrs":
            # qrs SNR is in dB → convert to a power estimate against signal RMS
            snr_db = nf.get("qrs_snr_db")
            tmpl_rms = nf.get("qrs_template_rms")
            if snr_db is not None and not np.isnan(snr_db) and tmpl_rms:
                np_est = float((tmpl_rms ** 2) * (10 ** (-snr_db / 10.0)))
            else:
                np_est = nf.get("bp_noise_power", 0.0)
        else:
            np_est = nf.get("bp_noise_power", 0.0)

        m = compute_signal_metrics(orig, recon, noise_power=np_est)
        prd_vals.append(m["prd_percent"])
        prdn_vals.append(m.get("prdn_noise_percent", float("nan")))
        rmse_vals.append(m["rmse"])
        cos_vals.append(m["cosine_similarity"])
        noise_power_vals.append(float(np_est))
        if "bp_noise_rms" in nf:
            noise_rms_vals.append(float(nf["bp_noise_rms"]))
        if modality == "ecg":
            snr = nf.get("qrs_snr_db", float("nan"))
            if isinstance(snr, float) and not np.isnan(snr):
                qrs_snr_vals.append(snr)

        # Spectral
        be = psd_band_error(orig, recon, fs=sample_rate, bands=bands)
        for k, v in be.items():
            if k.endswith("_rel_error"):
                band_err_per_band.setdefault(k, []).append(float(v))
        band_total_errs.append(float(be["band_total_rel_error"]))
        wf = weighted_freq_prd(orig, recon, fs=sample_rate, weights=freq_weights)
        wfprd_vals.append(float(wf["weighted_freq_prd_percent"]))
        coh = spectral_coherence(
            orig, recon, fs=sample_rate, band=coherence_band,
        )
        # spectral_coherence returns a dict with one entry; pull the value
        coh_vals.append(float(next(iter(coh.values()))))

    # --- Physiology (ECG: HR/HRV) ----------------------------------------
    physiology: dict[str, Any] = {}
    if modality == "ecg" and pairs:
        originals = np.stack([p[0] for p in pairs])
        recons = np.stack([p[1] for p in pairs])

        # vs raw original (current/canonical reference; subject to detector
        # noise sensitivity on noisy ground truth)
        ecg_summary_raw, per_sample_raw = summarize_ecg_alignment(
            originals, recons, sample_rate=sample_rate,
        )
        # vs filtered original (closer to "true" peaks; if the codec is
        # denoising correctly, vs_filtered errors should be <= vs_raw)
        filtered_originals = np.stack([
            _filter_signal_safe(o, fs=sample_rate, lowcut=0.5, highcut=40.0)
            for o in originals
        ])
        ecg_summary_filt, _ = summarize_ecg_alignment(
            filtered_originals, recons, sample_rate=sample_rate,
        )

        physiology = {
            "vs_raw_original": ecg_summary_raw or {},
            "vs_filtered_original": ecg_summary_filt or {},
        }

        # Noise-tertile stratification of vs_raw HR errors. Only run when
        # we have per-sample noise rms and at least 6 valid pairs.
        if (
            per_sample_raw is not None
            and len(noise_rms_vals) == len(per_sample_raw)
            and sum(1 for x in per_sample_raw if x is not None) >= 6
        ):
            valid_idx = [
                i for i, x in enumerate(per_sample_raw) if x is not None
            ]
            valid_noise = np.array([noise_rms_vals[i] for i in valid_idx])
            t1, t2 = np.percentile(valid_noise, [33.33, 66.67])

            def _bucket(v: float) -> str:
                if v <= t1:
                    return "clean"
                if v <= t2:
                    return "median"
                return "noisy"

            buckets: dict[str, dict[str, list[float]]] = {
                k: {"hr_abs_err": [], "sdnn_abs_err": [], "rmssd_abs_err": []}
                for k in ("clean", "median", "noisy")
            }
            for i in valid_idx:
                ps = per_sample_raw[i]
                if ps is None:
                    continue
                bucket = _bucket(noise_rms_vals[i])
                d = ps.get("delta", {})
                if "hr_bpm" in d:
                    buckets[bucket]["hr_abs_err"].append(abs(float(d["hr_bpm"])))
                if "sdnn_ms" in d:
                    buckets[bucket]["sdnn_abs_err"].append(abs(float(d["sdnn_ms"])))
                if "rmssd_ms" in d:
                    buckets[bucket]["rmssd_abs_err"].append(
                        abs(float(d["rmssd_ms"]))
                    )

            physiology["by_noise_tertile"] = {
                "thresholds_bp_noise_rms": {
                    "clean_max": float(t1),
                    "median_max": float(t2),
                },
                "buckets": {
                    name: {
                        "n": len(b["hr_abs_err"]),
                        "hr_mae_bpm": _aggregate(b["hr_abs_err"]),
                        "sdnn_mae_ms": _aggregate(b["sdnn_abs_err"]),
                        "rmssd_mae_ms": _aggregate(b["rmssd_abs_err"]),
                    }
                    for name, b in buckets.items()
                },
            }
    elif modality == "ppg":
        if pairs:
            originals = np.stack([p[0] for p in pairs])
            recons = np.stack([p[1] for p in pairs])
            ppg_summary, per_sample = summarize_physiokit_alignment(
                originals,
                recons,
                sample_rate=sample_rate,
                low_hz=0.5,
                high_hz=8.0,
                order=3,
                min_peaks=5,
            )
            physiology = ppg_summary or {}
            peak_summary, peak_per_sample = summarize_ppg_peak_alignment(
                originals,
                recons,
                sample_rate=sample_rate,
                low_hz=0.5,
                high_hz=8.0,
                order=3,
                min_peaks=5,
                timing_tolerance_ms=125.0,
            )
            if peak_summary:
                physiology["peak_alignment"] = peak_summary

            if (
                per_sample is not None
                and len(noise_rms_vals) == len(per_sample)
                and sum(1 for x in per_sample if x is not None) >= 6
            ):
                valid_idx = [i for i, x in enumerate(per_sample) if x is not None]
                valid_noise = np.array([noise_rms_vals[i] for i in valid_idx])
                t1, t2 = np.percentile(valid_noise, [33.33, 66.67])

                def _bucket(v: float) -> str:
                    if v <= t1:
                        return "clean"
                    if v <= t2:
                        return "median"
                    return "noisy"

                buckets: dict[str, dict[str, list[float]]] = {
                    k: {
                        "hr_abs_err": [],
                        "sdnn_abs_err": [],
                        "rmssd_abs_err": [],
                        "peak_timing_err": [],
                        "peak_f1": [],
                    }
                    for k in ("clean", "median", "noisy")
                }
                for i in valid_idx:
                    ps = per_sample[i]
                    if ps is None:
                        continue
                    bucket = _bucket(noise_rms_vals[i])
                    delta = ps.get("delta", {})
                    if "hr_bpm" in delta:
                        buckets[bucket]["hr_abs_err"].append(abs(float(delta["hr_bpm"])))
                    if "sdnn_ms" in delta:
                        buckets[bucket]["sdnn_abs_err"].append(abs(float(delta["sdnn_ms"])))
                    if "rmssd_ms" in delta:
                        buckets[bucket]["rmssd_abs_err"].append(abs(float(delta["rmssd_ms"])))
                    if peak_per_sample is not None and i < len(peak_per_sample):
                        pps = peak_per_sample[i]
                        if pps is not None:
                            if pps.get("peak_timing_mae_ms") is not None:
                                buckets[bucket]["peak_timing_err"].append(
                                    float(pps["peak_timing_mae_ms"])
                                )
                            buckets[bucket]["peak_f1"].append(100.0 * float(pps["f1"]))

                physiology["by_noise_tertile"] = {
                    "thresholds_bp_noise_rms": {
                        "clean_max": float(t1),
                        "median_max": float(t2),
                    },
                    "buckets": {
                        name: {
                            "n": len(bucket_vals["hr_abs_err"]),
                            "hr_mae_bpm": _aggregate(bucket_vals["hr_abs_err"]),
                            "sdnn_mae_ms": _aggregate(bucket_vals["sdnn_abs_err"]),
                            "rmssd_mae_ms": _aggregate(bucket_vals["rmssd_abs_err"]),
                            "peak_timing_mae_ms": _aggregate(bucket_vals["peak_timing_err"]),
                            "peak_f1_pct": _aggregate(bucket_vals["peak_f1"]),
                        }
                        for name, bucket_vals in buckets.items()
                    },
                }

    # --- Bitrate ---------------------------------------------------------
    bitrate: dict[str, Any] = {}
    if entropy:
        em = entropy.get("metrics", {})
        bitrate.update({
            "best_prior_tag": entropy.get("tag"),
            "val_bits_per_token": em.get("val_bits_per_token"),
            "val_bits_per_frame": em.get("val_bits_per_frame"),
            "cr_codec_uniform": em.get("cr_codec_uniform"),
            "cr_codec_learned": em.get("cr_codec_learned"),
        })
    cmp = {}
    for source in (
        summary.get("compression"),
        summary.get("compression_stats"),
        summary.get("h5_eval_metrics", {}).get("compression"),
    ):
        if isinstance(source, dict):
            cmp.update(source)
    if cmp:
        bitrate.setdefault(
            "codec_compression_ratio",
            cmp.get("compression_ratio", cmp.get("uniform_compression_ratio")),
        )
        for key in (
            "effective_downsample_factor",
            "downsample_factor",
            "effective_sample_rate_hz",
            "uniform_codec_bitrate_bps",
            "uniform_bits_per_sample",
            "uniform_bits_per_frame",
            "raw_bitrate_bps",
            "raw_bits_per_frame",
        ):
            if key in cmp:
                bitrate.setdefault(key, cmp.get(key))
        bitrate.setdefault("encoder_total_params", cmp.get("encoder_total_params"))

    # --- Stability (stitching) -------------------------------------------
    stability: dict[str, Any] = {}
    if stitching:
        # stitching_report.json: each method has summary stats already
        for method_name, stats in stitching.items():
            if not isinstance(stats, dict):
                continue
            stability[method_name] = {
                k: stats.get(k) for k in (
                    "prd_percent_mean", "cosine_similarity_mean",
                    "mse_mean", "seam_ratio_mean", "seam_rms_mean",
                ) if k in stats
            }

    return {
        "run_dir": str(run_dir),
        "modality": modality,
        "sample_rate": sample_rate,
        "num_samples_loaded": len(loaded_pairs),
        "num_samples": len(pairs),
        "num_samples_rejected": len(loaded_pairs) - len(pairs),
        "min_signal_std": min_signal_std,
        "noise_estimator": noise_estimator,
        "bitrate": bitrate,
        "time_domain": {
            "prd_percent": _aggregate(prd_vals),
            "prdn_noise_percent": _aggregate(
                [v for v in prdn_vals if not np.isnan(v)]
            ),
            "rmse": _aggregate(rmse_vals),
            "cosine_similarity": _aggregate(cos_vals),
        },
        "spectral": {
            "band_total_rel_error": _aggregate(band_total_errs),
            "per_band_rel_error": {
                k: _aggregate(v) for k, v in band_err_per_band.items()
            },
            "weighted_freq_prd_percent": _aggregate(wfprd_vals),
            "coherence": _aggregate(coh_vals),
        },
        "physiology": physiology,
        "stability": stability,
        "context": {
            "noise_power": _aggregate(noise_power_vals),
            "noise_rms": _aggregate(noise_rms_vals),
            "qrs_snr_db": _aggregate(qrs_snr_vals),
        },
    }


def write_quality_scorecard(
    run_dir: Path,
    *,
    modality: str,
    sample_rate: int,
    output_path: Path | None = None,
    **kwargs: Any,
) -> Path:
    """Build and persist a scorecard. Returns the output path."""
    card = build_quality_scorecard(
        run_dir, modality=modality, sample_rate=sample_rate, **kwargs,
    )
    out = output_path or (Path(run_dir) / "quality_scorecard.json")
    out.write_text(json.dumps(card, indent=2))
    return out


__all__ = ["build_quality_scorecard", "write_quality_scorecard"]
