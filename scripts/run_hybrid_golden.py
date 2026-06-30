"""End-to-end Hybrid (learned denoiser + SPIHT) golden run.

The Hybrid lane is the third v1 codec family alongside DSP (SPIHT) and AI
(RVQ). It pre-denoises each frame with a trained wavelet-gain network and then
hands the cleaned signal to the unchanged SPIHT encoder. The operating point is
fully described by a :class:`~compressionkit.experiments.registry.GoldenExperiment`
whose ``hybrid`` field (:class:`HybridSpec`) declares the denoiser, codec
backend, wavelet, and levels.

``HybridSpec.strategy`` is an open dispatch key so new hybrid approaches
(DSP-first, noise predictors, unrolled shrinkage, denoiser+RVQ, ...) can be
added here without changing the registry model.

Outputs under ``results/{run_name}/`` mirror the SPIHT/RVQ goldens:

* ``summary.json``           — codec parameters + aggregate metrics
* ``sample_NNN.csv``         — per-frame (original, reconstructed) CSVs
* ``quality_scorecard.json`` — via
  :func:`compressionkit.evaluation.scorecard.build_quality_scorecard`
* ``deploy/``                — SPIHT deploy package + denoiser artifact + manifest

Usage::

    source .venv/bin/activate
    python scripts/run_hybrid_golden.py --experiment-id ecg-hybrid-8x
    python scripts/run_hybrid_golden.py --experiment-id ppg-hybrid-8x
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from compressionkit.evaluation.codec import LearnedShrinkSpihtCodec
from compressionkit.evaluation.scorecard import build_quality_scorecard
from compressionkit.experiments.registry import GoldenExperiment, get_golden
from compressionkit.export.spiht_deploy import export_spiht_deploy
from compressionkit.pipeline.learned_stages import load_wavelet_gain_preprocessor
from compressionkit.runtime.spiht import SpihtCodec

logger = logging.getLogger(__name__)


# Per-modality operating points (match the SPIHT/RVQ goldens).
_FRAME_SIZE = {"ecg": 512, "ppg": 320}
_ECG_SOURCE_RATE = 500  # PTB-XL native sample rate

# PPG v1 unified-cache evaluation domain (mirrors run_spiht_golden_ppg).
_PPG_V1_DATASET_ID = "ppg-unified-strict-sanitize-v1"
_PPG_V1_CACHE_ROOT = "datasets/ppg_cache_strict_sanitize"
_PPG_V1_SOURCE_SLUGS = ("bidmc", "butppg", "ppg_dalia", "wesad")

_SUPPORTED_STRATEGIES = {"wavelet_gain_spiht"}


def _per_frame_metrics(orig: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    err = orig - recon
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    sig_pow = float(np.sum(orig**2))
    err_pow = float(np.sum(err**2))
    if sig_pow > 0 and err_pow > 0:
        prd = 100.0 * float(np.sqrt(err_pow / sig_pow))
        snr_db = 10.0 * float(np.log10(sig_pow / err_pow))
    elif err_pow == 0:
        prd = 0.0
        snr_db = float("inf")
    else:
        prd = 0.0
        snr_db = 0.0
    return {"prd": prd, "mse": mse, "mae": mae, "snr_db": snr_db}


def _aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr)),
    }


def _load_ecg_frames(exp: GoldenExperiment, args: argparse.Namespace, frame_size: int) -> list[np.ndarray]:
    from compressionkit.datasets.ecg import _resample, load_ecg_file_splits, load_ecg_signal

    datasets_dir = Path(args.datasets_dir)
    _, val_files, _ = load_ecg_file_splits(
        datasets_dir, args.dataset_glob, train_ratio=0.8, val_ratio=0.2, seed=42
    )
    if args.max_val_files:
        val_files = val_files[: args.max_val_files]
    logger.info("Using %d ECG validation files", len(val_files))

    frames: list[np.ndarray] = []
    skipped_files = 0
    for fpath in val_files:
        try:
            signal_native = load_ecg_signal(fpath, lead_index=args.lead_index)
            signal = _resample(signal_native, _ECG_SOURCE_RATE, exp.sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", fpath.name, exc)
            skipped_files += 1
            continue
        for start in range(0, len(signal) - frame_size + 1, frame_size):
            frames.append(signal[start : start + frame_size].astype(np.float32))
    logger.info("Loaded %d ECG frames (skipped files=%d)", len(frames), skipped_files)
    return frames


def _load_ppg_frames(exp: GoldenExperiment, args: argparse.Namespace, frame_size: int) -> list[np.ndarray]:
    from compressionkit.datasets.ppg_cache import SourceWeight, load_cached_raw_windows

    if exp.dataset_id != _PPG_V1_DATASET_ID:
        raise SystemExit(
            f"PPG hybrid golden expects dataset_id {_PPG_V1_DATASET_ID!r}, got {exp.dataset_id!r}."
        )
    sources = [SourceWeight(slug=slug, weight=1.0) for slug in args.cache_sources]
    windows = load_cached_raw_windows(
        sources,
        cache_root=Path(args.cache_root),
        frame_size=frame_size,
        split="val",
        max_windows=args.num_windows,
        seed=args.seed,
    )
    logger.info(
        "Using %d unified-cache PPG validation windows from %s (%s)",
        len(windows), args.cache_root, ", ".join(args.cache_sources),
    )
    return [np.asarray(w, dtype=np.float32) for w in windows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True, help="e.g. ecg-hybrid-8x or ppg-hybrid-8x")
    parser.add_argument("--results-root", default="results")
    # ECG data source
    parser.add_argument("--datasets-dir", default="/home/vscode/datasets")
    parser.add_argument("--dataset-glob", default="ptbxl/*.h5")
    parser.add_argument("--max-val-files", type=int, default=200)
    parser.add_argument("--lead-index", type=int, default=1)
    # PPG data source
    parser.add_argument("--cache-root", default=_PPG_V1_CACHE_ROOT)
    parser.add_argument("--cache-sources", nargs="+", default=list(_PPG_V1_SOURCE_SLUGS))
    parser.add_argument("--num-windows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    # Output controls
    parser.add_argument("--max-samples-csv", type=int, default=1000)
    parser.add_argument("--disable-ac", action="store_true")
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    exp = get_golden(args.experiment_id)
    if exp.method != "hybrid" or exp.hybrid is None:
        raise SystemExit(
            f"experiment {args.experiment_id!r} is method={exp.method!r}; this script is for hybrid goldens."
        )
    spec = exp.hybrid
    if spec.strategy not in _SUPPORTED_STRATEGIES:
        raise SystemExit(
            f"hybrid strategy {spec.strategy!r} is not implemented by this runner "
            f"(supported: {sorted(_SUPPORTED_STRATEGIES)})."
        )
    if spec.backend != "spiht":
        raise SystemExit(f"hybrid backend {spec.backend!r} is not implemented by this runner (spiht only).")

    frame_size = _FRAME_SIZE[exp.modality]
    run_dir = Path(args.results_root) / exp.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    denoiser_path = Path(spec.denoiser_path)
    pre = load_wavelet_gain_preprocessor(
        denoiser_path, frame_size=frame_size, wavelet=spec.wavelet, levels=spec.levels
    )
    codec = LearnedShrinkSpihtCodec(
        name=exp.experiment_id.replace("-", "_"),
        modality=exp.modality,
        sample_rate=exp.sample_rate,
        frame_size=frame_size,
        target_cr=float(exp.compression_ratio),
        wavelet=spec.wavelet,
        levels=spec.levels,
        use_ac=not args.disable_ac,
        coeff_denoiser=pre.coeff_denoiser,
    )
    logger.info(
        "Hybrid codec: strategy=%s denoiser=%s fs=%d frame=%d cr=%gx wavelet=%s L=%d use_ac=%s max_bits=%d",
        spec.strategy, denoiser_path, codec.sample_rate, codec.frame_size, codec.target_cr,
        codec.wavelet, codec.levels, codec.use_ac, codec.max_bits,
    )

    if exp.modality == "ecg":
        raw_frames = _load_ecg_frames(exp, args, frame_size)
    else:
        raw_frames = _load_ppg_frames(exp, args, frame_size)

    # Clear stale per-frame CSVs so the rebuilt scorecard reflects only this run.
    for csv_path in run_dir.glob("sample_*.csv"):
        csv_path.unlink()

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    bits_used: list[int] = []
    skipped_flat_frames = 0
    for frame in raw_frames:
        frame = np.asarray(frame, dtype=np.float32)
        std = float(np.std(frame))
        if std < 1e-3:
            skipped_flat_frames += 1
            continue
        frame_norm = (frame - float(np.mean(frame))) / (std + 1e-6)
        enc = codec.encode(frame_norm)
        recon = codec.decode(enc).astype(np.float32)
        pairs.append((frame_norm, recon))
        bits_used.append(int(enc.nbits))

    if not pairs:
        raise SystemExit("No frames evaluated.")

    n_frames = len(pairs)
    logger.info("Evaluated %d frames (skipped flat frames=%d)", n_frames, skipped_flat_frames)

    prd_list, mse_list, mae_list, snr_list = [], [], [], []
    for orig, recon in pairs:
        m = _per_frame_metrics(orig, recon)
        prd_list.append(m["prd"])
        mse_list.append(m["mse"])
        mae_list.append(m["mae"])
        snr_list.append(m["snr_db"])

    bits_per_sample = int(codec.bits_per_sample)
    mean_bits = float(np.mean(bits_used))
    frame_duration_s = codec.frame_size / codec.sample_rate
    bitrate_bps = mean_bits / frame_duration_s
    raw_bps = codec.sample_rate * bits_per_sample
    actual_cr = raw_bps / bitrate_bps if bitrate_bps > 0 else float("inf")
    raw_bits_per_frame = codec.frame_size * bits_per_sample

    prd_stats = _aggregate(prd_list)
    snr_stats = _aggregate(snr_list)
    mse_stats = _aggregate(mse_list)
    mae_stats = _aggregate(mae_list)

    summary = {
        "experiment_id": exp.experiment_id,
        "run_name": exp.run_name,
        "modality": exp.modality,
        "method": exp.method,
        "hybrid": {
            "strategy": spec.strategy,
            "backend": spec.backend,
            "denoiser_path": str(denoiser_path),
            "wavelet": spec.wavelet,
            "levels": spec.levels,
            "prefilter_low_hz": spec.prefilter_low_hz,
            "prefilter_high_hz": spec.prefilter_high_hz,
        },
        "codec": {
            "modality": codec.modality,
            "sample_rate": codec.sample_rate,
            "frame_size": codec.frame_size,
            "target_cr": codec.target_cr,
            "wavelet": codec.wavelet,
            "levels": codec.levels,
            "use_ac": codec.use_ac,
            "bits_per_sample": bits_per_sample,
            "max_bits": codec.max_bits,
        },
        "data": {
            "dataset_id": exp.dataset_id,
            "datasets_dir": str(args.datasets_dir),
            "n_frames": n_frames,
        },
        "metrics": {
            "n_frames": n_frames,
            "skipped_flat_frames": skipped_flat_frames,
            "prd_percent": {k: round(v, 4) for k, v in prd_stats.items()},
            "snr_db": {k: round(v, 3) for k, v in snr_stats.items()},
            "mse": {k: round(v, 6) for k, v in mse_stats.items()},
            "mae": {k: round(v, 6) for k, v in mae_stats.items()},
            "bits_per_frame": {
                "mean": round(mean_bits, 2),
                "max_budget": codec.max_bits,
                "min_used": int(np.min(bits_used)),
                "max_used": int(np.max(bits_used)),
            },
            "bitrate_bps": round(bitrate_bps, 2),
            "raw_bps": raw_bps,
            "actual_cr": round(actual_cr, 3),
        },
        # ``compression`` block is what build_quality_scorecard() reads to
        # populate its top-level ``bitrate`` section.
        "compression": {
            "compression_ratio": round(actual_cr, 4),
            "uniform_codec_bitrate_bps": round(bitrate_bps, 2),
            "uniform_bits_per_frame": round(mean_bits, 2),
            "raw_bitrate_bps": raw_bps,
            "raw_bits_per_frame": raw_bits_per_frame,
            "effective_sample_rate_hz": codec.sample_rate,
            "downsample_factor": 1,
            "effective_downsample_factor": 1,
            "encoder_total_params": 0,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary.json")

    n_csv = min(args.max_samples_csv, n_frames)
    for idx in range(n_csv):
        orig, recon = pairs[idx]
        pd.DataFrame(
            {"time_index": np.arange(len(orig)), "original": orig, "reconstructed": recon}
        ).to_csv(run_dir / f"sample_{idx:03d}.csv", index=False)
    logger.info("Wrote %d per-frame sample CSVs", n_csv)

    scorecard_summary: dict[str, float | int] = {
        "mean_prd_percent": summary["metrics"]["prd_percent"]["mean"],
        "median_prd_percent": summary["metrics"]["prd_percent"]["median"],
        "p95_prd_percent": summary["metrics"]["prd_percent"]["p95"],
        "mean_snr_db": summary["metrics"]["snr_db"]["mean"],
        "median_snr_db": summary["metrics"]["snr_db"]["median"],
        "actual_cr": summary["metrics"]["actual_cr"],
        "bitrate_bps": summary["metrics"]["bitrate_bps"],
        "n_frames": summary["metrics"]["n_frames"],
    }

    try:
        scorecard = build_quality_scorecard(run_dir, modality=exp.modality, sample_rate=exp.sample_rate)
        (run_dir / "quality_scorecard.json").write_text(json.dumps(scorecard, indent=2))
        logger.info("Wrote quality_scorecard.json")
    except Exception:  # noqa: BLE001
        logger.exception("Quality scorecard build failed; continuing without it.")

    if not args.skip_deploy:
        _export_deploy(exp, codec, denoiser_path, run_dir, scorecard_summary, use_ac=not args.disable_ac)

    print()
    print(f"=== Hybrid golden complete: {exp.experiment_id} ===")
    print(f"Run dir          : {run_dir}")
    print(f"Frames evaluated : {n_frames}")
    print(f"Target CR        : {codec.target_cr:.2f}x   Actual CR: {actual_cr:.2f}x")
    print(f"Bitrate          : {bitrate_bps:.1f} bps  (raw {raw_bps} bps)")
    print(f"PRD%   mean/median/p95 : {prd_stats['mean']:.2f} / {prd_stats['median']:.2f} / {prd_stats['p95']:.2f}")
    print(f"SNR dB mean/median     : {snr_stats['mean']:.2f} / {snr_stats['median']:.2f}")


def _export_deploy(
    exp: GoldenExperiment,
    codec: LearnedShrinkSpihtCodec,
    denoiser_path: Path,
    run_dir: Path,
    scorecard_summary: dict[str, float | int],
    *,
    use_ac: bool,
) -> None:
    """Write a Hybrid deploy package: SPIHT backend + denoiser artifact + manifest."""
    deploy_dir = run_dir / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    spiht_runtime = SpihtCodec(
        modality=exp.modality,
        sample_rate=exp.sample_rate,
        frame_size=codec.frame_size,
        target_cr=float(exp.compression_ratio),
        wavelet=codec.wavelet,
        levels=codec.levels,
        use_ac=use_ac,
        name=exp.experiment_id.replace("-", "_") + "_spiht",
    )
    model_card_info = {
        "experiment_id": exp.experiment_id,
        "modality": exp.modality,
        "method": exp.method,
        "sample_rate": exp.sample_rate,
        "compression_ratio": exp.compression_ratio,
        "wavelet": codec.wavelet,
        "levels": codec.levels,
        "use_ac": use_ac,
        "frame_size": codec.frame_size,
        "dataset_id": exp.dataset_id,
        "hybrid_strategy": exp.hybrid.strategy if exp.hybrid else None,
    }
    try:
        arts = export_spiht_deploy(
            spiht_runtime,
            output_dir=deploy_dir,
            model_card_info=model_card_info,
            scorecard_summary=scorecard_summary,
        )
        logger.info("SPIHT deploy artifacts written to %s", deploy_dir)
    except Exception:  # noqa: BLE001
        logger.exception("SPIHT deploy export failed; continuing with denoiser + manifest only.")
        arts = None

    # Stage the trained denoiser alongside the SPIHT package.
    denoiser_dst = deploy_dir / "denoiser_gain_model.keras"
    try:
        shutil.copyfile(denoiser_path, denoiser_dst)
        config_src = denoiser_path.parent / "train_config.json"
        if config_src.exists():
            shutil.copyfile(config_src, deploy_dir / "denoiser_train_config.json")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to stage denoiser artifact into deploy package.")

    manifest = {
        "pipeline": "hybrid",
        "experiment_id": exp.experiment_id,
        "run_name": exp.run_name,
        "stages": [
            {
                "stage": "denoise",
                "type": exp.hybrid.strategy if exp.hybrid else None,
                "artifact": denoiser_dst.name,
                "wavelet": codec.wavelet,
                "levels": codec.levels,
                "frame_size": codec.frame_size,
            },
            {
                "stage": "codec",
                "type": "spiht",
                "wavelet": codec.wavelet,
                "levels": codec.levels,
                "use_ac": use_ac,
                "target_cr": exp.compression_ratio,
                "artifacts": arts.as_dict() if arts is not None else None,
            },
        ],
    }
    (deploy_dir / "hybrid_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote hybrid deploy manifest to %s", deploy_dir / "hybrid_manifest.json")


if __name__ == "__main__":
    main()
