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

from compressionkit.evaluation.ecg_morphology import evaluate_ecg_morphology
from compressionkit.evaluation.metrics import (
    compute_signal_metrics,
    summarize_ecg_alignment,
    summarize_physiokit_alignment,
    summarize_ppg_peak_alignment,
)
from compressionkit.evaluation.noise import (
    estimate_ecg_noise_floor,
    estimate_ppg_noise_floor,
)
from compressionkit.evaluation.ppg_morphology import evaluate_ppg_morphology
from compressionkit.evaluation.robustness import summarize_robustness_for_scorecard
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


def _match_run_record(blob: Any, run_dir: Path) -> Any:
    """Extract the record for ``run_dir`` from a loose JSON artifact."""
    run_name = run_dir.name
    run_dir_str = str(run_dir)
    if isinstance(blob, dict):
        if run_name in blob:
            return blob[run_name]
        candidate_name = blob.get("run_name")
        candidate_dir = blob.get("run_dir")
        if candidate_name == run_name or candidate_dir == run_dir_str:
            return blob
        return None
    if isinstance(blob, list):
        if blob and all(isinstance(item, dict) and "test_name" in item for item in blob):
            return blob
        for item in blob:
            if not isinstance(item, dict):
                continue
            candidate_name = item.get("run_name")
            candidate_dir = item.get("run_dir")
            if candidate_name == run_name or candidate_dir == run_dir_str:
                return item
    return None


def _load_optional_json(path: Path | None) -> Any:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _summarize_adversarial_record(record: Any, *, source_path: Path) -> dict[str, Any] | None:
    if not isinstance(record, list):
        return None
    tests: dict[str, Any] = {}
    for row in record:
        if not isinstance(row, dict) or "test_name" not in row:
            continue
        tests[str(row["test_name"])] = row
    if not tests:
        return None
    zero = tests.get("zero_input", {})
    return {
        "source": str(source_path),
        "zero_input": {
            "output_l2_when_input_zero": zero.get("output_l2_when_input_zero"),
            "hallucinated_peaks": zero.get("hallucinated_peaks"),
            "output_band_power": zero.get("output_band_power"),
            "output_energy": zero.get("output_energy"),
        },
        "tests": tests,
    }


def _summarize_imprinting_record(record: Any, *, source_path: Path) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    valid_samples = record.get("valid_samples")
    normalized_metrics: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict) and "mean" in value and "n" not in value and valid_samples is not None:
            normalized_metrics[key] = {"n": int(valid_samples), **value}
        else:
            normalized_metrics[key] = value
    return {
        "source": str(source_path),
        "sample_rate": record.get("sample_rate"),
        "valid_samples": valid_samples,
        "occlusion_window_ms": record.get("occlusion_window_ms"),
        "definition": record.get("definition", {}),
        "metrics": normalized_metrics,
    }


def _aggregate(values: list[float]) -> dict[str, float]:
    raw = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(raw)
    arr = raw[finite_mask]
    n_dropped = int(raw.size - arr.size)
    if arr.size == 0:
        return {"n": 0, "n_dropped": n_dropped}
    return {
        "n": int(arr.size),
        "n_dropped": n_dropped,
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
        pairs.append(
            (
                df["original"].to_numpy(dtype=np.float32),
                df["reconstructed"].to_numpy(dtype=np.float32),
            )
        )
    return pairs


def _load_clean_reference_arrays(
    reference_path: Path,
    *,
    num_samples: int,
    key: str | None = None,
) -> np.ndarray:
    """Load aligned clean-reference windows from ``.npz`` or ``.npy``.

    Supported formats:

    * ``.npy``: a single array of shape ``(N, T)``.
    * ``.npz``: one of ``clean_truth``, ``clean``, ``reference``, or
      ``references`` by default; ``key`` overrides auto-detection.
    """
    reference_path = Path(reference_path)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    if reference_path.suffix == ".npy":
        arr = np.load(reference_path)
    elif reference_path.suffix == ".npz":
        blob = np.load(reference_path)
        chosen_key = key
        if chosen_key is None:
            for candidate in ("clean_truth", "clean", "reference", "references"):
                if candidate in blob:
                    chosen_key = candidate
                    break
        if chosen_key is None or chosen_key not in blob:
            raise KeyError(
                f"Could not find clean-reference array in {reference_path}; available keys={sorted(blob.files)}"
            )
        arr = blob[chosen_key]
    else:
        raise ValueError(f"Unsupported clean-reference format {reference_path.suffix!r}; expected .npy or .npz")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Clean-reference array must be 2-D, got shape {arr.shape}")
    if arr.shape[0] != num_samples:
        raise ValueError(f"Clean-reference sample count mismatch: expected {num_samples}, got {arr.shape[0]}")
    return arr


def _compute_reference_view(
    references: np.ndarray,
    candidates: np.ndarray,
    *,
    modality: str,
    sample_rate: int,
    bands: list[tuple[float, float]],
    freq_weights: list[tuple[float, float, float]],
    coherence_band: tuple[float, float],
) -> dict[str, Any]:
    """Compute scorecard sections for ``candidates`` measured against ``references``."""
    prd_vals: list[float] = []
    rmse_vals: list[float] = []
    cos_vals: list[float] = []
    band_err_per_band: dict[str, list[float]] = {}
    band_total_errs: list[float] = []
    wfprd_vals: list[float] = []
    coh_vals: list[float] = []

    for ref, cand in zip(references, candidates):
        m = compute_signal_metrics(ref, cand)
        prd_vals.append(m["prd_percent"])
        rmse_vals.append(m["rmse"])
        cos_vals.append(m["cosine_similarity"])

        be = psd_band_error(ref, cand, fs=sample_rate, bands=bands)
        for metric_name, value in be.items():
            if metric_name.endswith("_rel_error"):
                band_err_per_band.setdefault(metric_name, []).append(float(value))
        band_total_errs.append(float(be["band_total_rel_error"]))
        wf = weighted_freq_prd(ref, cand, fs=sample_rate, weights=freq_weights)
        wfprd_vals.append(float(wf["weighted_freq_prd_percent"]))
        coh = spectral_coherence(ref, cand, fs=sample_rate, band=coherence_band)
        coh_vals.append(float(next(iter(coh.values()))))

    physiology: dict[str, Any] = {}
    morphology: dict[str, Any] = {}
    if modality == "ecg":
        ecg_summary_raw, _ = summarize_ecg_alignment(
            references,
            candidates,
            sample_rate=sample_rate,
        )
        filtered_references = np.stack(
            [_filter_signal_safe(r, fs=sample_rate, lowcut=0.5, highcut=40.0) for r in references]
        )
        ecg_summary_filt, _ = summarize_ecg_alignment(
            filtered_references,
            candidates,
            sample_rate=sample_rate,
        )
        physiology = {
            "vs_reference": ecg_summary_raw or {},
            "vs_filtered_reference": ecg_summary_filt or {},
        }
        try:
            morphology = evaluate_ecg_morphology(
                references,
                candidates,
                sample_rate=sample_rate,
            )
        except Exception:
            morphology = {}
    elif modality == "ppg":
        ppg_summary, _ = summarize_physiokit_alignment(
            references,
            candidates,
            sample_rate=sample_rate,
            low_hz=0.5,
            high_hz=8.0,
            order=3,
            min_peaks=5,
        )
        physiology = ppg_summary or {}
        peak_summary, _ = summarize_ppg_peak_alignment(
            references,
            candidates,
            sample_rate=sample_rate,
            low_hz=0.5,
            high_hz=8.0,
            order=3,
            min_peaks=5,
            timing_tolerance_ms=125.0,
        )
        if peak_summary:
            physiology["peak_alignment"] = peak_summary
        try:
            morphology = evaluate_ppg_morphology(
                references,
                candidates,
                sample_rate=sample_rate,
                low_hz=0.5,
                high_hz=8.0,
                order=3,
            )
        except Exception:
            morphology = {}

    return {
        "time_domain": {
            "prd_percent": _aggregate(prd_vals),
            "rmse": _aggregate(rmse_vals),
            "cosine_similarity": _aggregate(cos_vals),
        },
        "spectral": {
            "band_total_rel_error": _aggregate(band_total_errs),
            "per_band_rel_error": {k: _aggregate(v) for k, v in band_err_per_band.items()},
            "weighted_freq_prd_percent": _aggregate(wfprd_vals),
            "coherence": _aggregate(coh_vals),
        },
        "physiology": physiology,
        "morphology": morphology,
    }


def _metric_delta(
    baseline: dict[str, Any],
    improved: dict[str, Any],
    path: tuple[str, ...],
    *,
    higher_is_better: bool = False,
) -> float | None:
    """Return aggregate-mean improvement along ``path`` when present."""
    left: Any = baseline
    right: Any = improved
    for key in path:
        if not isinstance(left, dict) or key not in left or not isinstance(right, dict) or key not in right:
            return None
        left = left[key]
        right = right[key]
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    if "mean" not in left or "mean" not in right:
        return None
    base_mean = left["mean"]
    out_mean = right["mean"]
    if base_mean is None or out_mean is None:
        return None
    return float(out_mean - base_mean) if higher_is_better else float(base_mean - out_mean)


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


def _safe_round(value: Any, ndigits: int = 3) -> float | None:
    """Round to ``ndigits`` while tolerating ``None``/non-numeric inputs."""
    try:
        if value is None:
            return None
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _build_scorecard_headline(out: dict[str, Any]) -> dict[str, Any]:
    """Consolidate the decision-relevant metrics into one top-level block.

    Surfaces *both* faithfulness (PRD vs the recorded, still-noisy input) and
    truth fidelity (PRD vs clean ground truth, taken from the robustness
    fixture) so denoising lanes are not misjudged by faithfulness alone,
    alongside the robust-SNR floor, the empirical-SNR anchors, and the
    pure-noise imprint (hallucination) probe. All values are pulled from blocks
    already computed elsewhere in the scorecard; this is a read-only summary.
    """
    td = out.get("time_domain", {}) or {}
    bitrate = out.get("bitrate", {}) or {}
    robustness = out.get("robustness", {}) or {}
    rb_headline = robustness.get("headline", {}) if isinstance(robustness, dict) else {}
    emp = rb_headline.get("empirical_snr", {}) if isinstance(rb_headline, dict) else {}
    reference = robustness.get("reference", {}) if isinstance(robustness, dict) else {}

    return {
        "num_samples": out.get("num_samples"),
        "compression_ratio": _safe_round(bitrate.get("codec_compression_ratio"), 3),
        "faithful_prd_vs_input_pct": _safe_round(td.get("prd_percent", {}).get("mean")),
        "truth_prd_vs_clean_pct": _safe_round(reference.get("clean")),
        "truth_prd_at_native_noise_pct": _safe_round(reference.get("native")),
        "prd_degradation_slope_per_db": _safe_round(emp.get("prd_slope_per_db")),
        "prd_at_0db_pct": _safe_round(emp.get("prd_at_0db")),
        "prd_at_-6db_pct": _safe_round(emp.get("prd_at_-6db")),
        "imprint_output_autocorr": _safe_round(rb_headline.get("imprint_output_autocorr_max"), 4),
    }


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
    clean_reference_path: Path | None = None,
    clean_reference_key: str | None = None,
    clean_reference_label: str = "clean_truth",
    adversarial_metrics_path: Path | None = None,
    imprinting_metrics_path: Path | None = None,
    robustness_metrics_path: Path | None = None,
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
        clean_reference_path: Optional path to a ``.npz`` / ``.npy`` bundle
            of aligned clean-reference windows, one per loaded sample. When
            provided, the output gains a ``clean_reference`` block with
            reconstruction-vs-truth metrics, input-vs-truth baselines, and
            denoising deltas.
        clean_reference_key: Optional array key when ``clean_reference_path``
            is a ``.npz``.
        clean_reference_label: Human-readable label stored in the output.
        adversarial_metrics_path: Optional JSON artifact with adversarial or
            hallucination metrics. Supports a dict keyed by run name or a list
            of per-run dicts.
        imprinting_metrics_path: Optional JSON artifact with localized
            imprinting metrics. Supports a list of per-run dicts or a single
            per-run dict.
        robustness_metrics_path: Optional path to the ``robustness_metrics.json``
            written by ``scripts/eval_golden_robustness``. When ``None``, the
            ``robustness_metrics.json`` beside the run is auto-detected and, if
            present, summarized into a ``robustness`` block (empirical-SNR
            degradation curve, additive artifact families, imprint probe).

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
        freq_weights = list(ECG_DEFAULT_FREQ_WEIGHTS) if modality == "ecg" else list(PPG_DEFAULT_FREQ_WEIGHTS)
    if coherence_band is None:
        coherence_band = ECG_DEFAULT_COHERENCE_BAND if modality == "ecg" else PPG_DEFAULT_COHERENCE_BAND

    loaded_pairs = _load_samples(run_dir)
    valid_indices = [
        i
        for i, (orig, recon) in enumerate(loaded_pairs)
        if np.isfinite(orig).all() and np.isfinite(recon).all() and float(np.std(orig)) >= min_signal_std
    ]
    pairs = [loaded_pairs[i] for i in valid_indices]
    summary = _read_summary(run_dir)
    stitching = _read_stitching(run_dir)
    entropy = _read_best_entropy(run_dir)
    adversarial_blob = _load_optional_json(adversarial_metrics_path)
    imprinting_blob = _load_optional_json(imprinting_metrics_path)
    # Robustness: auto-detect the per-run artifact written by
    # ``scripts/eval_golden_robustness`` unless an explicit path is given.
    if robustness_metrics_path is None:
        default_robustness = run_dir / "robustness_metrics.json"
        robustness_metrics_path = default_robustness if default_robustness.exists() else None
    robustness_blob = _load_optional_json(robustness_metrics_path)

    clean_reference_block: dict[str, Any] = {}
    if clean_reference_path is not None and loaded_pairs:
        clean_all = _load_clean_reference_arrays(
            Path(clean_reference_path),
            num_samples=len(loaded_pairs),
            key=clean_reference_key,
        )
        clean_filtered = clean_all[valid_indices] if valid_indices else clean_all[:0]
        if clean_filtered.size > 0 and len(clean_filtered) != len(pairs):
            raise ValueError(
                f"Filtered clean-reference count mismatch: {len(clean_filtered)} vs {len(pairs)} kept samples"
            )
        if len(pairs) > 0 and clean_filtered.shape[1] != pairs[0][0].shape[0]:
            raise ValueError(
                f"Clean-reference frame length mismatch: {clean_filtered.shape[1]} vs {pairs[0][0].shape[0]}"
            )

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
                np_est = float((tmpl_rms**2) * (10 ** (-snr_db / 10.0)))
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
            orig,
            recon,
            fs=sample_rate,
            band=coherence_band,
        )
        # spectral_coherence returns a dict with one entry; pull the value
        coh_vals.append(float(next(iter(coh.values()))))

    # --- Physiology (ECG: HR/HRV) ----------------------------------------
    physiology: dict[str, Any] = {}
    morphology: dict[str, Any] = {}
    if modality == "ecg" and pairs:
        originals = np.stack([p[0] for p in pairs])
        recons = np.stack([p[1] for p in pairs])

        # vs raw original (current/canonical reference; subject to detector
        # noise sensitivity on noisy ground truth)
        ecg_summary_raw, per_sample_raw = summarize_ecg_alignment(
            originals,
            recons,
            sample_rate=sample_rate,
        )
        # vs filtered original (closer to "true" peaks; if the codec is
        # denoising correctly, vs_filtered errors should be <= vs_raw)
        filtered_originals = np.stack(
            [_filter_signal_safe(o, fs=sample_rate, lowcut=0.5, highcut=40.0) for o in originals]
        )
        ecg_summary_filt, _ = summarize_ecg_alignment(
            filtered_originals,
            recons,
            sample_rate=sample_rate,
        )

        physiology = {
            "vs_raw_original": ecg_summary_raw or {},
            "vs_filtered_original": ecg_summary_filt or {},
        }

        # Beat morphology (R-amplitude, QRS width, ST deviation, T-wave,
        # baseline drift, beat-shape correlation).
        try:
            morphology = evaluate_ecg_morphology(
                originals,
                recons,
                sample_rate=sample_rate,
            )
        except Exception:
            morphology = {}

        # Noise-tertile stratification of vs_raw HR errors. Only run when
        # we have per-sample noise rms and at least 6 valid pairs.
        if (
            per_sample_raw is not None
            and len(noise_rms_vals) == len(per_sample_raw)
            and sum(1 for x in per_sample_raw if x is not None) >= 6
        ):
            valid_idx = [i for i, x in enumerate(per_sample_raw) if x is not None]
            valid_noise = np.array([noise_rms_vals[i] for i in valid_idx])
            t1, t2 = np.percentile(valid_noise, [33.33, 66.67])

            def _bucket(v: float) -> str:
                if v <= t1:
                    return "clean"
                if v <= t2:
                    return "median"
                return "noisy"

            buckets: dict[str, dict[str, list[float]]] = {
                k: {"hr_abs_err": [], "sdnn_abs_err": [], "rmssd_abs_err": []} for k in ("clean", "median", "noisy")
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
                    buckets[bucket]["rmssd_abs_err"].append(abs(float(d["rmssd_ms"])))

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

            # Pulse morphology (AC amplitude, upstroke slope, pulse width,
            # rise/fall times, pulse-shape correlation, dicrotic notch).
            try:
                morphology = evaluate_ppg_morphology(
                    originals,
                    recons,
                    sample_rate=sample_rate,
                    low_hz=0.5,
                    high_hz=8.0,
                    order=3,
                )
            except Exception:
                morphology = {}

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
                                buckets[bucket]["peak_timing_err"].append(float(pps["peak_timing_mae_ms"]))
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
        bitrate.update(
            {
                "best_prior_tag": entropy.get("tag"),
                "val_bits_per_token": em.get("val_bits_per_token"),
                "val_bits_per_frame": em.get("val_bits_per_frame"),
                "cr_codec_uniform": em.get("cr_codec_uniform"),
                "cr_codec_learned": em.get("cr_codec_learned"),
            }
        )
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
        # Accept both layouts:
        #   * legacy / direct: ``{method_name: {prd_percent_mean: ...}, ...}``
        #   * evaluate_stitching shape:
        #       ``{"methods": {method_name: {mean_prd_percent: ...}}, ...}``
        methods_blob = stitching.get("methods") if isinstance(stitching.get("methods"), dict) else stitching
        # Pairs of (scorecard key, candidate source keys in priority order).
        _STITCH_KEY_MAP: list[tuple[str, tuple[str, ...]]] = [
            ("prd_percent_mean", ("prd_percent_mean", "mean_prd_percent")),
            ("cosine_similarity_mean", ("cosine_similarity_mean", "mean_cosine_similarity")),
            ("mse_mean", ("mse_mean", "mean_mse")),
            ("seam_ratio_mean", ("seam_ratio_mean", "mean_seam_ratio")),
            ("seam_rms_mean", ("seam_rms_mean", "mean_seam_rms")),
            ("non_seam_rms_mean", ("non_seam_rms_mean", "mean_non_seam_rms")),
            ("hr_mae_bpm", ("hr_mae_bpm",)),
            ("rmssd_mae_ms", ("rmssd_mae_ms",)),
            ("sdnn_mae_ms", ("sdnn_mae_ms",)),
            ("num_recordings", ("num_recordings",)),
        ]
        for method_name, stats in methods_blob.items():
            if not isinstance(stats, dict):
                continue
            block: dict[str, Any] = {}
            for out_key, src_keys in _STITCH_KEY_MAP:
                for src_key in src_keys:
                    if src_key in stats:
                        block[out_key] = stats[src_key]
                        break
            if block:
                stability[method_name] = block

    # --- Noise-tertile stratification (time-domain + spectral) -----------
    by_noise_tertile: dict[str, Any] = {}
    n_valid_noise = len(noise_rms_vals)
    if n_valid_noise >= 6 and n_valid_noise == len(prd_vals):
        noise_arr = np.array(noise_rms_vals)
        t1, t2 = np.percentile(noise_arr, [33.33, 66.67])

        def _noise_bucket(v: float) -> str:
            if v <= t1:
                return "clean"
            if v <= t2:
                return "median"
            return "noisy"

        td_buckets: dict[str, dict[str, list[float]]] = {
            k: {"prd": [], "prdn": [], "rmse": [], "cosine": []} for k in ("clean", "median", "noisy")
        }
        sp_buckets: dict[str, dict[str, list[float]]] = {
            k: {"band_total": [], "wfprd": [], "coherence": []} for k in ("clean", "median", "noisy")
        }
        for i in range(n_valid_noise):
            bucket = _noise_bucket(noise_rms_vals[i])
            td_buckets[bucket]["prd"].append(prd_vals[i])
            if not np.isnan(prdn_vals[i]):
                td_buckets[bucket]["prdn"].append(prdn_vals[i])
            td_buckets[bucket]["rmse"].append(rmse_vals[i])
            td_buckets[bucket]["cosine"].append(cos_vals[i])
            sp_buckets[bucket]["band_total"].append(band_total_errs[i])
            sp_buckets[bucket]["wfprd"].append(wfprd_vals[i])
            sp_buckets[bucket]["coherence"].append(coh_vals[i])

        by_noise_tertile = {
            "thresholds_bp_noise_rms": {
                "clean_max": float(t1),
                "median_max": float(t2),
            },
            "buckets": {
                name: {
                    "n": len(td_buckets[name]["prd"]),
                    "time_domain": {
                        "prd_percent": _aggregate(td_buckets[name]["prd"]),
                        "prdn_noise_percent": _aggregate(td_buckets[name]["prdn"]),
                        "rmse": _aggregate(td_buckets[name]["rmse"]),
                        "cosine_similarity": _aggregate(td_buckets[name]["cosine"]),
                    },
                    "spectral": {
                        "band_total_rel_error": _aggregate(
                            sp_buckets[name]["band_total"],
                        ),
                        "weighted_freq_prd_percent": _aggregate(
                            sp_buckets[name]["wfprd"],
                        ),
                        "coherence": _aggregate(sp_buckets[name]["coherence"]),
                    },
                }
                for name in ("clean", "median", "noisy")
            },
        }

    if clean_reference_path is not None and pairs:
        originals = np.stack([p[0] for p in pairs])
        recons = np.stack([p[1] for p in pairs])
        clean_references = np.asarray(clean_filtered, dtype=np.float32)
        baseline_view = _compute_reference_view(
            clean_references,
            originals,
            modality=modality,
            sample_rate=sample_rate,
            bands=bands,
            freq_weights=freq_weights,
            coherence_band=coherence_band,
        )
        output_view = _compute_reference_view(
            clean_references,
            recons,
            modality=modality,
            sample_rate=sample_rate,
            bands=bands,
            freq_weights=freq_weights,
            coherence_band=coherence_band,
        )
        clean_reference_block = {
            "label": clean_reference_label,
            "path": str(Path(clean_reference_path)),
            "num_samples": int(clean_references.shape[0]),
            "input_baseline": baseline_view,
            "reconstruction": output_view,
            "denoising": {
                "time_domain": {
                    "prd_percent_improvement": _metric_delta(
                        baseline_view,
                        output_view,
                        ("time_domain", "prd_percent"),
                    ),
                    "rmse_improvement": _metric_delta(
                        baseline_view,
                        output_view,
                        ("time_domain", "rmse"),
                    ),
                    "cosine_similarity_improvement": _metric_delta(
                        baseline_view,
                        output_view,
                        ("time_domain", "cosine_similarity"),
                        higher_is_better=True,
                    ),
                },
                "spectral": {
                    "band_total_rel_error_improvement": _metric_delta(
                        baseline_view,
                        output_view,
                        ("spectral", "band_total_rel_error"),
                    ),
                    "weighted_freq_prd_percent_improvement": _metric_delta(
                        baseline_view,
                        output_view,
                        ("spectral", "weighted_freq_prd_percent"),
                    ),
                    "coherence_improvement": _metric_delta(
                        baseline_view,
                        output_view,
                        ("spectral", "coherence"),
                        higher_is_better=True,
                    ),
                },
            },
        }

    out = {
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
            "prdn_noise_percent": _aggregate([v for v in prdn_vals if not np.isnan(v)]),
            "rmse": _aggregate(rmse_vals),
            "cosine_similarity": _aggregate(cos_vals),
        },
        "spectral": {
            "band_total_rel_error": _aggregate(band_total_errs),
            "per_band_rel_error": {k: _aggregate(v) for k, v in band_err_per_band.items()},
            "weighted_freq_prd_percent": _aggregate(wfprd_vals),
            "coherence": _aggregate(coh_vals),
        },
        "physiology": physiology,
        "morphology": morphology,
        "stability": stability,
        "by_noise_tertile": by_noise_tertile,
        "context": {
            "noise_power": _aggregate(noise_power_vals),
            "noise_rms": _aggregate(noise_rms_vals),
            "qrs_snr_db": _aggregate(qrs_snr_vals),
        },
    }
    if clean_reference_block:
        out["clean_reference"] = clean_reference_block
    if adversarial_blob is not None:
        adversarial_record = _match_run_record(adversarial_blob, run_dir)
        hallucination = _summarize_adversarial_record(
            adversarial_record,
            source_path=Path(adversarial_metrics_path),
        )
        if hallucination:
            out["hallucination"] = hallucination
    if imprinting_blob is not None:
        imprinting_record = _match_run_record(imprinting_blob, run_dir)
        imprinting = _summarize_imprinting_record(
            imprinting_record,
            source_path=Path(imprinting_metrics_path),
        )
        if imprinting:
            out["imprinting"] = imprinting
    if robustness_blob is not None:
        robustness = summarize_robustness_for_scorecard(robustness_blob)
        if robustness:
            out["robustness"] = robustness
    # Lead with a consolidated, decision-relevant summary (read-only view over
    # the blocks computed above): faithfulness + truth fidelity + robust-SNR
    # floor + imprint safety, so denoisers are not judged on faithfulness alone.
    out = {"headline": _build_scorecard_headline(out), **out}
    return out


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
        run_dir,
        modality=modality,
        sample_rate=sample_rate,
        **kwargs,
    )
    out = output_path or (Path(run_dir) / "quality_scorecard.json")
    out.write_text(json.dumps(card, indent=2))
    return out


__all__ = ["build_quality_scorecard", "write_quality_scorecard"]
