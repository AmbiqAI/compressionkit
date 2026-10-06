"""Select quality-gated, real PPG clips for browser demonstrations.

The helper reads continuous PLETH windows from the BIDMC PPG and Respiration
Dataset and resamples them to the target demo rate.  BIDMC is available under
the Open Data Commons Attribution License v1.0; retain the generated manifest
when distributing these clips.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from compressionkit.datasets.ppg_h5 import _resample
from compressionkit.evaluation.metrics import compute_ppg_physiokit_metrics
from compressionkit.preprocessing.curation import ppg_pulse_snr_db


@dataclass(frozen=True)
class PpgDemoQuality:
    """Signal-quality measurements recorded for one demo clip."""

    clipping_fraction: float
    dynamic_range: float
    heart_rate_bpm: float
    heart_rate_qos: float
    num_peaks: int
    pulse_snr_db: float
    rr_cv: float
    standard_deviation: float


@dataclass(frozen=True)
class PpgDemoClip:
    """One accepted real PPG demo clip and its source provenance."""

    signal: np.ndarray
    quality: PpgDemoQuality
    source_record: str
    source_sample_rate: int
    start_seconds: float


def assess_ppg_demo_clip(signal: np.ndarray, *, sample_rate: int) -> PpgDemoQuality:
    """Validate one single-channel PPG clip for demo suitability.

    This gate favors readable, continuous waveforms with a stable pulse rhythm.
    It is only a demonstration-quality filter and is not a clinical SQI.

    Args:
        signal: One-dimensional PPG signal.
        sample_rate: Signal sample rate in Hz.

    Returns:
        Quality measurements for an accepted clip.

    Raises:
        ValueError: If the clip does not meet the demo-quality thresholds.
    """
    samples = np.asarray(signal, dtype=np.float32).reshape(-1)
    if samples.size < 10 * sample_rate:
        raise ValueError("PPG demo clips must be at least 10 seconds long")
    if not np.isfinite(samples).all():
        raise ValueError("PPG clip contains non-finite samples")

    standard_deviation = float(np.std(samples))
    dynamic_range = float(np.ptp(samples))
    if standard_deviation < 1e-4 or dynamic_range < 1e-3:
        raise ValueError("PPG clip is effectively flat")
    rail_tolerance = max(1e-7, dynamic_range * 1e-6)
    clipping_fraction = float(
        np.mean((samples <= samples.min() + rail_tolerance) | (samples >= samples.max() - rail_tolerance))
    )
    if clipping_fraction > 0.01:
        raise ValueError(f"PPG clip appears clipped ({clipping_fraction:.2%} at its rails)")

    metrics = compute_ppg_physiokit_metrics(
        samples,
        sample_rate=sample_rate,
        low_hz=0.5,
        high_hz=min(8.0, sample_rate / 2.5),
        order=3,
        min_peaks=5,
    )
    if metrics is None:
        raise ValueError("PPG clip has too few detectable pulse peaks")
    heart_rate_bpm = float(metrics["hr_bpm"])
    heart_rate_qos = float(metrics["hr_qos"])
    peak_locations = np.asarray(metrics["peak_locations"], dtype=np.int64)
    rr_intervals = np.diff(peak_locations) / float(sample_rate)
    if rr_intervals.size < 3:
        raise ValueError("PPG clip has too few pulse intervals")
    rr_cv = float(np.std(rr_intervals) / np.mean(rr_intervals))
    pulse_snr_db = float(ppg_pulse_snr_db(samples, sample_rate))
    if not 45.0 <= heart_rate_bpm <= 120.0:
        raise ValueError(f"PPG heart rate {heart_rate_bpm:.1f} BPM is outside the demo range")
    if heart_rate_qos < 0.8:
        raise ValueError(f"PPG heart-rate quality {heart_rate_qos:.2f} is too low for a demo")
    if rr_cv > 0.2:
        raise ValueError(f"PPG pulse-interval variability {rr_cv:.3f} is too high for a clean demo clip")
    if not np.isfinite(pulse_snr_db) or pulse_snr_db < 8.0:
        raise ValueError(f"PPG pulse-template SNR {pulse_snr_db:.1f} dB is too low for a demo")

    return PpgDemoQuality(
        clipping_fraction=clipping_fraction,
        dynamic_range=dynamic_range,
        heart_rate_bpm=heart_rate_bpm,
        heart_rate_qos=heart_rate_qos,
        num_peaks=int(metrics["num_peaks"]),
        pulse_snr_db=pulse_snr_db,
        rr_cv=rr_cv,
        standard_deviation=standard_deviation,
    )


def _record_metadata(record_path: Path) -> tuple[int, str]:
    """Read source metadata from one canonical BIDMC H5 file."""
    with h5py.File(record_path, "r") as handle:
        source_sample_rate = int(handle.attrs["fs"])
        source = str(handle.attrs.get("source", ""))
        record_id = str(handle.attrs.get("patient_id", record_path.stem))
    if source != "bidmc":
        raise ValueError(f"Expected BIDMC record, got source={source!r} in {record_path}")
    return source_sample_rate, record_id


def generate_ppg_demo_clips(
    dataset_dir: str | Path,
    *,
    num_clips: int = 10,
    duration_seconds: float = 30.0,
    sample_rate: int = 64,
    seed: int = 42,
    attempts_per_record: int = 20,
) -> list[PpgDemoClip]:
    """Select quality-gated, real BIDMC PPG clips for a demo.

    Each accepted clip comes from a distinct BIDMC record where possible.
    Windows are continuous in the source recording before resampling.

    Args:
        dataset_dir: Canonical BIDMC H5 directory.
        num_clips: Number of clips to select, from 1 through 10.
        duration_seconds: Length of each clip in seconds; must be at least 10.
        sample_rate: Output sample rate in Hz.
        seed: Seed for deterministic record and window selection.
        attempts_per_record: Candidate windows to try from each record.

    Returns:
        Accepted real PPG clips, each sampled at ``sample_rate``.
    """
    if not 1 <= num_clips <= 10:
        raise ValueError("num_clips must be between 1 and 10")
    if duration_seconds < 10.0:
        raise ValueError("duration_seconds must be at least 10")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if attempts_per_record <= 0:
        raise ValueError("attempts_per_record must be positive")

    records = sorted(Path(dataset_dir).glob("*.h5"))
    if not records:
        raise FileNotFoundError(f"No canonical BIDMC H5 records found in {dataset_dir}")

    rng = np.random.default_rng(seed)
    selected: list[PpgDemoClip] = []
    for record_path in rng.permutation(records):
        source_sample_rate, record_id = _record_metadata(record_path)
        with h5py.File(record_path, "r") as handle:
            source_samples = handle["data"][0].astype(np.float32, copy=False)
        source_length = round(duration_seconds * source_sample_rate)
        if source_samples.size < source_length:
            continue

        for _attempt in range(attempts_per_record):
            max_start = source_samples.size - source_length
            start = int(rng.integers(0, max_start + 1))
            output_window = _resample(source_samples[start : start + source_length], source_sample_rate, sample_rate)
            try:
                quality = assess_ppg_demo_clip(output_window, sample_rate=sample_rate)
            except ValueError:
                continue
            selected.append(
                PpgDemoClip(
                    signal=output_window,
                    quality=quality,
                    source_record=record_id,
                    source_sample_rate=source_sample_rate,
                    start_seconds=start / float(source_sample_rate),
                )
            )
            break
        if len(selected) == num_clips:
            return selected

    raise ValueError(
        f"Selected only {len(selected)} of {num_clips} quality-gated PPG clips from {dataset_dir}; "
        "relax the quality gate or use a larger source dataset."
    )


__all__ = ["PpgDemoClip", "PpgDemoQuality", "assess_ppg_demo_clip", "generate_ppg_demo_clips"]
