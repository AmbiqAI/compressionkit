"""End-to-end SPIHT PPG golden run.

Produces a golden directory under ``results/{run_name}/`` that mirrors the
layout used by RVQ goldens:

* ``summary.json``         — codec parameters + aggregate metrics
* ``sample_NNN.csv``       — per-frame (original, reconstructed) CSVs
* ``quality_scorecard.json`` — built via
  :func:`compressionkit.evaluation.scorecard.build_quality_scorecard`
* ``deploy/``              — weightless deploy package via
  :func:`compressionkit.export.spiht_deploy.export_spiht_deploy`

The canonical v1 encode/decode pipeline runs on the unified strict-sanitized
PPG validation cache so SPIHT metrics are measured on the same held-out domain
as the faithful RVQ goldens.

Usage::

    source .venv/bin/activate
    python scripts/run_spiht_golden_ppg.py --experiment-id ppg-spiht-4x

Add ``--max-val-files`` to bound runtime; default 40 mirrors
``scripts/spiht_baseline_ppg.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from compressionkit.configs.paths import default_datasets_dir
from compressionkit.datasets.ppg import load_ppg_file_splits, load_ppg_signal
from compressionkit.datasets.ppg_cache import SourceWeight, load_cached_raw_windows
from compressionkit.evaluation.codec import SpihtAcCodec
from compressionkit.evaluation.scorecard import build_quality_scorecard
from compressionkit.evaluation.spiht_stitching import evaluate_spiht_stitching
from compressionkit.experiments.registry import get_golden
from compressionkit.export.spiht_deploy import export_spiht_deploy
from compressionkit.runtime.spiht import SpihtCodec

logger = logging.getLogger(__name__)


# Per-modality SPIHT defaults — mirrors compressionkit.experiments.runner.
_PPG_FRAME_SIZE = 320
_PPG_WAVELET = "coif5"
_PPG_LEVELS = 6
_TUNED_WAVELET = "bior4.4"
_TUNED_LEVELS = 6
_PPG_V1_DATASET_ID = "ppg-unified-strict-sanitize-v1"
_PPG_V1_CACHE_ROOT = "datasets/ppg_cache_strict_sanitize"
_PPG_V1_SOURCE_SLUGS = ("bidmc", "butppg", "ppg_dalia", "wesad")


def _purge_old_sample_csvs(run_dir: Path) -> None:
    """Delete stale sample CSVs so the rebuilt scorecard uses only the current run."""
    for csv_path in run_dir.glob("sample_*.csv"):
        csv_path.unlink()


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


def _load_eval_frames_from_cache(args: argparse.Namespace, *, frame_size: int) -> np.ndarray:
    sources = [SourceWeight(slug=slug, weight=1.0) for slug in args.cache_sources]
    return load_cached_raw_windows(
        sources,
        cache_root=Path(args.cache_root),
        frame_size=frame_size,
        split="val",
        max_windows=args.num_windows,
        seed=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="ppg-spiht-4x")
    parser.add_argument("--datasets-dir", default=default_datasets_dir())
    parser.add_argument(
        "--dataset-glob",
        default="mesa-commercial-use/polysomnography/edfs/*.edf",
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--max-val-files", type=int, default=40)
    parser.add_argument("--cache-root", default=_PPG_V1_CACHE_ROOT)
    parser.add_argument(
        "--cache-sources",
        nargs="+",
        default=list(_PPG_V1_SOURCE_SLUGS),
        help="Unified-cache source slugs for canonical v1 evaluation.",
    )
    parser.add_argument(
        "--num-windows", type=int, default=1000,
        help="Number of cached validation windows to score for canonical v1 evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples-per-file", type=int, default=64 * 60 * 5,
        help="Per-file PPG sample count (default: 5 min @ 64 Hz).",
    )
    parser.add_argument(
        "--max-samples-csv", type=int, default=1000,
        help="Number of sample_NNN.csv per-frame files to write for the scorecard.",
    )
    parser.add_argument("--target-label", default="Pleth")
    parser.add_argument(
        "--wavelet",
        default=_PPG_WAVELET,
        help=(
            "Wavelet to use for the SPIHT codec. "
            f"Default: {_PPG_WAVELET}. Tuned PPG setting: {_TUNED_WAVELET}."
        ),
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=_PPG_LEVELS,
        help=(
            "DWT levels for the SPIHT codec. "
            f"Default: {_PPG_LEVELS}. Tuned PPG setting: {_TUNED_LEVELS}."
        ),
    )
    parser.add_argument(
        "--disable-ac",
        action="store_true",
        help="Disable arithmetic coding. By default this script uses AC.",
    )
    parser.add_argument(
        "--tuned",
        action="store_true",
        help=(
            "Shortcut for the tuned PPG operating point: "
            f"wavelet={_TUNED_WAVELET}, levels={_TUNED_LEVELS}."
        ),
    )
    parser.add_argument(
        "--max-stitching-signals", type=int, default=10,
        help="Number of long signals to retain for the stitching evaluation.",
    )
    parser.add_argument("--stitching-hop-ratio", type=float, default=0.5)
    parser.add_argument(
        "--skip-stitching", action="store_true",
        help="Skip the long-signal stitching evaluation (stability scorecard block).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.tuned:
        args.wavelet = _TUNED_WAVELET
        args.levels = _TUNED_LEVELS

    exp = get_golden(args.experiment_id)
    if exp.method != "spiht" or exp.modality != "ppg":
        raise SystemExit(
            f"experiment {args.experiment_id!r} is method={exp.method!r} modality={exp.modality!r}; "
            "this script is for PPG SPIHT goldens only."
        )

    run_dir = Path(args.results_root) / exp.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    codec = SpihtCodec(
        modality=exp.modality,
        sample_rate=exp.sample_rate,
        frame_size=_PPG_FRAME_SIZE,
        target_cr=float(exp.compression_ratio),
        wavelet=args.wavelet,
        levels=args.levels,
        use_ac=not args.disable_ac,
        name=exp.experiment_id.replace("-", "_"),
    )
    logger.info(
        "Codec: fs=%d frame=%d target_cr=%gx wavelet=%s L=%d use_ac=%s max_bits=%d",
        codec.sample_rate,
        codec.frame_size,
        codec.target_cr,
        codec.wavelet,
        codec.levels,
        codec.use_ac,
        codec.max_bits,
    )

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    bits_used: list[int] = []
    stitching_signals: list[np.ndarray] = []
    skipped_files = 0
    skipped_flat_frames = 0

    datasets_dir = Path(args.datasets_dir)
    val_files: list[Path] = []
    if exp.dataset_id == _PPG_V1_DATASET_ID:
        frames = _load_eval_frames_from_cache(args, frame_size=codec.frame_size)
        logger.info(
            "Using %d unified-cache validation windows from %s (%s)",
            len(frames),
            args.cache_root,
            ", ".join(args.cache_sources),
        )
        for frame in frames:
            frame = np.asarray(frame, dtype=np.float32)
            std = float(np.std(frame))
            if std < 1e-3:
                skipped_flat_frames += 1
                continue
            frame_norm = (frame - float(np.mean(frame))) / (std + 1e-6)
            enc = codec.compress(frame_norm)
            recon = codec.decompress(enc).astype(np.float32)
            pairs.append((frame_norm, recon))
            bits_used.append(int(enc.nbits))
    else:
        _, val_files, _ = load_ppg_file_splits(
            datasets_dir, args.dataset_glob,
            train_ratio=0.8, val_ratio=0.2, seed=42,
        )
        if args.max_val_files:
            val_files = val_files[: args.max_val_files]
        logger.info("Using %d legacy validation files", len(val_files))

        for fpath in val_files:
            try:
                signal = load_ppg_signal(
                    fpath,
                    target_rate=codec.sample_rate,
                    num_samples=args.samples_per_file,
                    target_label=args.target_label,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort over heterogeneous files
                logger.warning("Skipping %s: %s", fpath.name, exc)
                skipped_files += 1
                continue

            # Retain long signals (>= 2 frames) for stitching evaluation.
            if len(signal) >= 2 * codec.frame_size and len(stitching_signals) < args.max_stitching_signals:
                stitching_signals.append(np.asarray(signal, dtype=np.float32))

            for start in range(0, len(signal) - codec.frame_size + 1, codec.frame_size):
                frame = signal[start : start + codec.frame_size].astype(np.float32)
                std = float(np.std(frame))
                if std < 1e-3:
                    skipped_flat_frames += 1
                    continue
                frame_norm = (frame - float(np.mean(frame))) / (std + 1e-6)
                enc = codec.compress(frame_norm)
                recon = codec.decompress(enc).astype(np.float32)
                pairs.append((frame_norm, recon))
                bits_used.append(int(enc.nbits))

    if not pairs:
        raise SystemExit("No frames evaluated — check dataset path and glob.")

    n_frames = len(pairs)
    logger.info(
        "Evaluated %d frames (skipped files=%d, flat frames=%d)",
        n_frames, skipped_files, skipped_flat_frames,
    )

    prd_list, mse_list, mae_list, snr_list = [], [], [], []
    for orig, recon in pairs:
        m = _per_frame_metrics(orig, recon)
        prd_list.append(m["prd"])
        mse_list.append(m["mse"])
        mae_list.append(m["mae"])
        snr_list.append(m["snr_db"])

    mean_bits = float(np.mean(bits_used))
    frame_duration_s = codec.frame_size / codec.sample_rate
    bitrate_bps = mean_bits / frame_duration_s
    raw_bps = codec.sample_rate * codec.config.bits_per_sample
    actual_cr = raw_bps / bitrate_bps if bitrate_bps > 0 else float("inf")
    raw_bits_per_frame = codec.frame_size * codec.config.bits_per_sample

    prd_stats = _aggregate(prd_list)
    snr_stats = _aggregate(snr_list)
    mse_stats = _aggregate(mse_list)
    mae_stats = _aggregate(mae_list)

    summary = {
        "experiment_id": exp.experiment_id,
        "run_name": exp.run_name,
        "modality": exp.modality,
        "method": exp.method,
        "codec": {
            "modality": codec.modality,
            "sample_rate": codec.sample_rate,
            "frame_size": codec.frame_size,
            "target_cr": codec.target_cr,
            "wavelet": codec.wavelet,
            "levels": codec.levels,
            "use_ac": codec.use_ac,
            "bits_per_sample": codec.config.bits_per_sample,
            "max_bits": codec.max_bits,
        },
        "data": {
            "dataset_id": exp.dataset_id,
            "evaluation_source": "unified_cache" if exp.dataset_id == _PPG_V1_DATASET_ID else "mesa_val_split",
            "cache_root": args.cache_root if exp.dataset_id == _PPG_V1_DATASET_ID else None,
            "cache_sources": list(args.cache_sources) if exp.dataset_id == _PPG_V1_DATASET_ID else None,
            "num_windows_requested": args.num_windows if exp.dataset_id == _PPG_V1_DATASET_ID else None,
            "seed": args.seed if exp.dataset_id == _PPG_V1_DATASET_ID else 42,
            "datasets_dir": str(datasets_dir) if exp.dataset_id != _PPG_V1_DATASET_ID else None,
            "dataset_glob": args.dataset_glob if exp.dataset_id != _PPG_V1_DATASET_ID else None,
            "n_val_files": len(val_files),
            "skipped_files": skipped_files,
            "samples_per_file": args.samples_per_file,
            "target_label": args.target_label,
            "split_seed": 42,
            "train_ratio": 0.8,
            "val_ratio": 0.2,
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

    # ------------------------------------------------------------------
    # Stitching evaluation (long-signal seam behavior) → stitching_report.json
    # build_quality_scorecard() reads this to populate the ``stability``
    # section.
    # ------------------------------------------------------------------
    if stitching_signals and not args.skip_stitching:
        stitch_codec = SpihtAcCodec(
            name=f"{exp.experiment_id}_ac",
            modality=exp.modality,
            sample_rate=codec.sample_rate,
            frame_size=codec.frame_size,
            target_cr=float(codec.target_cr),
            wavelet=codec.wavelet,
            levels=codec.levels,
            use_ac=codec.use_ac,
            bits_per_sample=codec.config.bits_per_sample,
        )
        try:
            stitching_report = evaluate_spiht_stitching(
                stitch_codec,
                stitching_signals,
                methods=["hard_concat", "overlap_add", "linear_crossfade", "tukey_overlap_add"],
                hop_ratio=args.stitching_hop_ratio,
                sample_rate=codec.sample_rate,
            )
            (run_dir / "stitching_report.json").write_text(json.dumps(stitching_report, indent=2))
            logger.info(
                "Wrote stitching_report.json (n_signals=%d, methods=%d)",
                stitching_report["num_recordings_eval"], len(stitching_report["methods"]),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Stitching evaluation failed; continuing without it.")

    _purge_old_sample_csvs(run_dir)
    n_csv = min(args.max_samples_csv, n_frames)
    for idx in range(n_csv):
        orig, recon = pairs[idx]
        df = pd.DataFrame({
            "time_index": np.arange(len(orig)),
            "original": orig,
            "reconstructed": recon,
        })
        df.to_csv(run_dir / f"sample_{idx:03d}.csv", index=False)
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
        scorecard = build_quality_scorecard(
            run_dir,
            modality=exp.modality,
            sample_rate=exp.sample_rate,
        )
        scorecard_path = run_dir / "quality_scorecard.json"
        scorecard_path.write_text(json.dumps(scorecard, indent=2))
        logger.info("Wrote quality_scorecard.json")
    except Exception:  # noqa: BLE001 — scorecard is optional; continue with summary-only
        logger.exception("Quality scorecard build failed; continuing without it.")

    deploy_dir = run_dir / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    model_card_info = {
        "experiment_id": exp.experiment_id,
        "modality": exp.modality,
        "method": exp.method,
        "sample_rate": exp.sample_rate,
        "compression_ratio": exp.compression_ratio,
        "wavelet": codec.wavelet,
        "levels": codec.levels,
        "use_ac": codec.use_ac,
        "frame_size": codec.frame_size,
        "dataset_id": exp.dataset_id,
    }
    arts = export_spiht_deploy(
        codec,
        output_dir=deploy_dir,
        model_card_info=model_card_info,
        scorecard_summary=scorecard_summary,
    )
    logger.info("Deploy artifacts written to %s", deploy_dir)

    print()
    print(f"=== SPIHT golden complete: {exp.experiment_id} ===")
    print(f"Run dir          : {run_dir}")
    print(f"HF repo (target) : {exp.hf_repo_id}")
    print(f"Frames evaluated : {n_frames}")
    print(f"Target CR        : {codec.target_cr:.2f}x")
    print(f"Actual CR        : {actual_cr:.2f}x")
    print(f"Bitrate          : {bitrate_bps:.1f} bps  (raw {raw_bps} bps)")
    print(f"PRD%   mean/median/p95 : {prd_stats['mean']:.2f} / {prd_stats['median']:.2f} / {prd_stats['p95']:.2f}")
    print(f"SNR dB mean/median     : {snr_stats['mean']:.2f} / {snr_stats['median']:.2f}")
    print(f"MSE    mean            : {mse_stats['mean']:.6f}")
    print(f"Bits/frame (mean/used range / budget): "
          f"{mean_bits:.0f} / [{int(np.min(bits_used))}-{int(np.max(bits_used))}] / {codec.max_bits}")
    print(f"Deploy artifacts : {arts.as_dict()}")
    print()
    print("Review the run dir before publishing. To publish later:")
    print(f"  uv run compressionkit golden run {exp.experiment_id} --skip-train --publish --dry-run")


if __name__ == "__main__":
    main()
