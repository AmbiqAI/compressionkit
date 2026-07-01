"""Localized imprinting evaluation for ECG and PPG codec runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import physiokit as pk

from compressionkit.configs.ecg_rvq import EcgRvqConfig
from compressionkit.configs.ppg_rvq import PpgRvqConfig
from compressionkit.datasets.ecg import collect_random_samples as collect_random_ecg_samples
from compressionkit.datasets.ppg import collect_random_samples as collect_random_ppg_samples
from compressionkit.evaluation.metrics import compute_signal_metrics
from compressionkit.evaluation.rvq_codec import RvqCodec
from compressionkit.preprocessing.ecg import build_augmenter as build_ecg_augmenter
from compressionkit.preprocessing.ecg import build_preprocessor as build_ecg_preprocessor
from compressionkit.preprocessing.ppg import build_augmenter as build_ppg_augmenter
from compressionkit.preprocessing.ppg import build_preprocessor as build_ppg_preprocessor
from compressionkit.trainers.ecg_rvq import build_datasets as build_ecg_datasets
from compressionkit.trainers.ppg_rvq import build_datasets as build_ppg_datasets


def _aggregate(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _reconstruct(codec: RvqCodec, frame: np.ndarray) -> np.ndarray:
    recon = codec.decode(codec.encode(np.asarray(frame, dtype=np.float32)))
    return np.asarray(recon, dtype=np.float32)


def load_ecg_validation_targets(run_dir: Path, sample_count: int, seed: int) -> np.ndarray:
    cfg = EcgRvqConfig.model_validate_json((run_dir / "config.json").read_text())
    preprocessor = build_ecg_preprocessor(frame_size=cfg.data.frame_size, epsilon=cfg.data.epsilon)
    augmenter = build_ecg_augmenter(aug_cfg=cfg.data.augmentation, sample_rate=cfg.data.effective_sample_rate)
    _train_ds, val_ds, _validation_steps, _info = build_ecg_datasets(cfg, preprocessor, augmenter)
    _inputs, targets = collect_random_ecg_samples(val_ds, sample_count=sample_count, rng=np.random.default_rng(seed))
    return np.asarray(targets[:, 0, :, 0], dtype=np.float32)


def load_ppg_validation_targets(run_dir: Path, sample_count: int, seed: int) -> np.ndarray:
    cfg = PpgRvqConfig.model_validate_json((run_dir / "config.json").read_text())
    preprocessor = build_ppg_preprocessor(frame_size=cfg.data.frame_size, epsilon=cfg.data.epsilon)
    augmenter = build_ppg_augmenter(tuple(cfg.data.gaussian_noise))
    _train_ds, val_ds, _validation_steps, _info = build_ppg_datasets(cfg, preprocessor, augmenter)
    _inputs, targets = collect_random_ppg_samples(val_ds, sample_count=sample_count, rng=np.random.default_rng(seed))
    return np.asarray(targets[:, 0, :, 0], dtype=np.float32)


def _select_ecg_peak(signal: np.ndarray, sample_rate: int, margin: int) -> int | None:
    try:
        peaks = np.asarray(pk.ecg.find_peaks(signal, sample_rate=sample_rate), dtype=np.int64).reshape(-1)
    except Exception:
        return None
    if peaks.size == 0:
        return None
    valid = peaks[(peaks >= margin) & (peaks < signal.size - margin)]
    if valid.size == 0:
        return None
    center = signal.size // 2
    return int(valid[np.argmin(np.abs(valid - center))])


def _clean_ppg(signal: np.ndarray, sample_rate: int) -> np.ndarray | None:
    try:
        cleaned = pk.ppg.clean(
            np.asarray(signal, dtype=np.float32),
            lowcut=0.5,
            highcut=8.0,
            sample_rate=sample_rate,
            order=3,
        )
    except Exception:
        return None
    return np.asarray(cleaned, dtype=np.float32)


def _select_ppg_peak(signal: np.ndarray, sample_rate: int, margin: int) -> int | None:
    cleaned = _clean_ppg(signal, sample_rate)
    if cleaned is None:
        return None
    try:
        peaks = np.asarray(pk.ppg.find_peaks(cleaned, sample_rate=sample_rate), dtype=np.int64).reshape(-1)
    except Exception:
        return None
    if peaks.size == 0:
        return None
    valid = peaks[(peaks >= margin) & (peaks < signal.size - margin)]
    if valid.size == 0:
        return None
    center = signal.size // 2
    return int(valid[np.argmin(np.abs(valid - center))])


def _peak_in_window_ecg(signal: np.ndarray, sample_rate: int, start: int, stop: int) -> bool:
    try:
        peaks = np.asarray(pk.ecg.find_peaks(signal, sample_rate=sample_rate), dtype=np.int64).reshape(-1)
    except Exception:
        return False
    return bool(np.any((peaks >= start) & (peaks < stop)))


def _peak_in_window_ppg(signal: np.ndarray, sample_rate: int, start: int, stop: int) -> bool:
    cleaned = _clean_ppg(signal, sample_rate)
    if cleaned is None:
        return False
    try:
        peaks = np.asarray(pk.ppg.find_peaks(cleaned, sample_rate=sample_rate), dtype=np.int64).reshape(-1)
    except Exception:
        return False
    return bool(np.any((peaks >= start) & (peaks < stop)))


def _evaluate_targets(
    run_dir: Path,
    *,
    modality: str,
    targets: np.ndarray,
    pre_ms: float,
    post_ms: float,
    peak_tolerance_ms: float,
) -> dict[str, Any]:
    codec = RvqCodec.from_run_dir(run_dir, modality=modality)
    sample_rate = int(codec.sample_rate)
    pre = max(1, round(pre_ms * sample_rate / 1000.0))
    post = max(1, round(post_ms * sample_rate / 1000.0))
    peak_tol = max(1, round(peak_tolerance_ms * sample_rate / 1000.0))

    if modality == "ecg":
        selector = _select_ecg_peak
        peak_in_window = _peak_in_window_ecg
        imprinting_text = (
            "Structured, target-like output that appears inside the occluded local QRS window "
            "despite the input samples in that window being zeroed."
        )
    else:
        selector = _select_ppg_peak
        peak_in_window = _peak_in_window_ppg
        imprinting_text = (
            "Structured, target-like output that appears inside the occluded local pulse window "
            "despite the input samples in that window being zeroed."
        )

    local_prd: list[float] = []
    local_cos: list[float] = []
    local_energy_ratio: list[float] = []
    outside_prd: list[float] = []
    masked_vs_clean_prd: list[float] = []
    gap_peak_hits = 0
    valid_samples = 0

    for target in targets:
        peak = selector(target, sample_rate, margin=max(pre, post) + peak_tol)
        if peak is None:
            continue

        start = max(0, peak - pre)
        stop = min(target.size, peak + post)
        mask = np.zeros(target.size, dtype=bool)
        mask[start:stop] = True
        if mask.sum() == 0 or (~mask).sum() == 0:
            continue

        masked_input = target.copy()
        masked_input[mask] = 0.0

        clean_recon = _reconstruct(codec, target)
        masked_recon = _reconstruct(codec, masked_input)

        local_metrics = compute_signal_metrics(target[mask], masked_recon[mask])
        outside_metrics = compute_signal_metrics(target[~mask], masked_recon[~mask])
        consistency_metrics = compute_signal_metrics(clean_recon[mask], masked_recon[mask])

        target_energy = float(np.sum(np.square(target[mask])))
        recon_energy = float(np.sum(np.square(masked_recon[mask])))
        local_prd.append(local_metrics["prd_percent"])
        local_cos.append(local_metrics["cosine_similarity"])
        local_energy_ratio.append(recon_energy / (target_energy + 1e-8))
        outside_prd.append(outside_metrics["prd_percent"])
        masked_vs_clean_prd.append(consistency_metrics["prd_percent"])
        gap_peak_hits += int(peak_in_window(masked_recon, sample_rate, start, stop))
        valid_samples += 1

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "sample_rate": sample_rate,
        "valid_samples": valid_samples,
        "occlusion_window_ms": {"pre": pre_ms, "post": post_ms},
        "definition": {
            "imprinting": imprinting_text,
            "local_energy_ratio": "Energy in reconstructed occluded window divided by target-window energy.",
            "local_cosine_to_target": "Cosine similarity between reconstructed occluded window and clean target window.",
            "masked_vs_clean_local_prd": (
                "PRD between the occluded-input reconstruction and the clean-input reconstruction in the same local window; "
                "lower means the model changed little despite the missing local evidence."
            ),
            "gap_peak_rate": "Fraction of samples where a detected physiological peak still appears inside the occluded gap.",
        },
        "metrics": {
            "local_prd_percent": _aggregate(local_prd),
            "local_cosine_to_target": _aggregate(local_cos),
            "local_energy_ratio": _aggregate(local_energy_ratio),
            "outside_prd_percent": _aggregate(outside_prd),
            "masked_vs_clean_local_prd": _aggregate(masked_vs_clean_prd),
            "gap_peak_rate": float(gap_peak_hits / valid_samples) if valid_samples else float("nan"),
        },
    }


def evaluate_ecg_local_imprinting_run(
    run_dir: Path,
    *,
    sample_count: int = 64,
    seed: int = 0,
    pre_ms: float = 20.0,
    post_ms: float = 20.0,
    peak_tolerance_ms: float = 10.0,
) -> dict[str, Any]:
    targets = load_ecg_validation_targets(Path(run_dir), sample_count=sample_count, seed=seed)
    return _evaluate_targets(
        Path(run_dir),
        modality="ecg",
        targets=targets,
        pre_ms=pre_ms,
        post_ms=post_ms,
        peak_tolerance_ms=peak_tolerance_ms,
    )


def evaluate_ppg_local_imprinting_run(
    run_dir: Path,
    *,
    sample_count: int = 64,
    seed: int = 0,
    pre_ms: float = 125.0,
    post_ms: float = 125.0,
    peak_tolerance_ms: float = 62.5,
) -> dict[str, Any]:
    targets = load_ppg_validation_targets(Path(run_dir), sample_count=sample_count, seed=seed)
    return _evaluate_targets(
        Path(run_dir),
        modality="ppg",
        targets=targets,
        pre_ms=pre_ms,
        post_ms=post_ms,
        peak_tolerance_ms=peak_tolerance_ms,
    )


def write_local_imprinting_report(
    run_dir: Path,
    *,
    modality: str,
    output_path: Path | None = None,
    sample_count: int = 64,
    seed: int = 0,
    pre_ms: float | None = None,
    post_ms: float | None = None,
    peak_tolerance_ms: float | None = None,
) -> Path:
    run_dir = Path(run_dir)
    modality = modality.lower()
    if modality == "ecg":
        report = evaluate_ecg_local_imprinting_run(
            run_dir,
            sample_count=sample_count,
            seed=seed,
            pre_ms=20.0 if pre_ms is None else pre_ms,
            post_ms=20.0 if post_ms is None else post_ms,
            peak_tolerance_ms=10.0 if peak_tolerance_ms is None else peak_tolerance_ms,
        )
    elif modality == "ppg":
        report = evaluate_ppg_local_imprinting_run(
            run_dir,
            sample_count=sample_count,
            seed=seed,
            pre_ms=125.0 if pre_ms is None else pre_ms,
            post_ms=125.0 if post_ms is None else post_ms,
            peak_tolerance_ms=62.5 if peak_tolerance_ms is None else peak_tolerance_ms,
        )
    else:
        raise ValueError(f"Unknown modality {modality!r}")

    out = output_path or (run_dir / "local_imprinting.json")
    out.write_text(json.dumps(report, indent=2))
    return out


__all__ = [
    "evaluate_ecg_local_imprinting_run",
    "evaluate_ppg_local_imprinting_run",
    "load_ecg_validation_targets",
    "load_ppg_validation_targets",
    "write_local_imprinting_report",
]
