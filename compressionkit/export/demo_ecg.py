"""Select quality-gated, real ECG clips for browser demonstrations.

The helper draws continuous MLII windows from the MIT-BIH Arrhythmia Database,
then resamples them to the requested demo rate. MIT-BIH is available under the
Open Data Commons Attribution License v1.0; keep the generated manifest and
its source attribution when publishing the resulting clips.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from compressionkit.datasets.ecg import _resample, load_ecg_signal
from compressionkit.evaluation.metrics import compute_ecg_hr_hrv


@dataclass(frozen=True)
class EcgDemoQuality:
    """Signal-quality measurements recorded for one demo clip."""

    clipping_fraction: float
    dynamic_range: float
    heart_rate_bpm: float
    num_r_peaks: int
    rr_cv: float
    standard_deviation: float


@dataclass(frozen=True)
class EcgDemoClip:
    """One accepted real ECG demo clip and its source provenance."""

    signal: np.ndarray
    quality: EcgDemoQuality
    source_record: str
    source_sample_rate: int
    start_seconds: float


@dataclass(frozen=True)
class EcgDemoExport:
    """Paths produced when ECG demo clips are written to disk."""

    csv_paths: list[Path]
    manifest_path: Path


def assess_ecg_demo_clip(signal: np.ndarray, *, sample_rate: int) -> EcgDemoQuality:
    """Validate one single-channel ECG clip for demo suitability.

    The gate rejects non-finite, nearly flat, rail-clipped, implausibly paced,
    or highly irregular traces. It favors clean, easy-to-inspect signals for a
    waveform-compression demonstration; it is not a clinical quality metric.

    Args:
        signal: One-dimensional ECG signal.
        sample_rate: Signal sample rate in Hz.

    Returns:
        Quality measurements for an accepted clip.

    Raises:
        ValueError: If the clip does not meet the demo-quality thresholds.
    """
    samples = np.asarray(signal, dtype=np.float32).reshape(-1)
    if samples.size < 10 * sample_rate:
        raise ValueError("ECG demo clips must be at least 10 seconds long")
    if not np.isfinite(samples).all():
        raise ValueError("ECG clip contains non-finite samples")

    standard_deviation = float(np.std(samples))
    dynamic_range = float(np.ptp(samples))
    if standard_deviation < 1e-4 or dynamic_range < 1e-3:
        raise ValueError("ECG clip is effectively flat")
    rail_tolerance = max(1e-7, dynamic_range * 1e-6)
    clipping_fraction = float(
        np.mean((samples <= samples.min() + rail_tolerance) | (samples >= samples.max() - rail_tolerance))
    )
    if clipping_fraction > 0.01:
        raise ValueError(f"ECG clip appears clipped ({clipping_fraction:.2%} at its rails)")

    hrv = compute_ecg_hr_hrv(samples, sample_rate=sample_rate)
    if hrv is None:
        raise ValueError("ECG clip has too few detectable R peaks")

    heart_rate_bpm = float(hrv["hr_bpm"])
    peak_locations = np.asarray(hrv["peak_locations"], dtype=np.int64)
    rr_intervals = np.diff(peak_locations) / float(sample_rate)
    if rr_intervals.size < 3:
        raise ValueError("ECG clip has too few R-R intervals")
    rr_cv = float(np.std(rr_intervals) / np.mean(rr_intervals))
    if not 45.0 <= heart_rate_bpm <= 110.0:
        raise ValueError(f"ECG heart rate {heart_rate_bpm:.1f} BPM is outside the demo range")
    if rr_cv > 0.15:
        raise ValueError(f"ECG R-R variability {rr_cv:.3f} is too high for a clean demo clip")

    return EcgDemoQuality(
        clipping_fraction=clipping_fraction,
        dynamic_range=dynamic_range,
        heart_rate_bpm=heart_rate_bpm,
        num_r_peaks=int(hrv["num_peaks"]),
        rr_cv=rr_cv,
        standard_deviation=standard_deviation,
    )


def _record_metadata(record_path: Path) -> tuple[int, str]:
    """Read sample rate and record identifier from one canonical MIT-BIH H5 file."""
    with h5py.File(record_path, "r") as handle:
        source_sample_rate = int(handle.attrs["fs"])
        source = str(handle.attrs.get("source", ""))
        record_id = str(handle.attrs.get("patient_id", record_path.stem))
    if source != "mitdb":
        raise ValueError(f"Expected MIT-BIH record, got source={source!r} in {record_path}")
    return source_sample_rate, record_id


def generate_ecg_demo_clips(
    dataset_dir: str | Path,
    *,
    num_clips: int = 10,
    duration_seconds: float = 30.0,
    sample_rate: int = 256,
    seed: int = 42,
    attempts_per_record: int = 20,
) -> list[EcgDemoClip]:
    """Select quality-gated, real MIT-BIH ECG clips for a demo.

    Each accepted clip comes from a distinct record where possible. Windows are
    continuous in the source recording before resampling to ``sample_rate``.

    Args:
        dataset_dir: Canonical MIT-BIH H5 directory, typically
            ``$COMPRESSIONKIT_DATASETS_DIR/mitdb``.
        num_clips: Number of clips to select, from 1 through 10.
        duration_seconds: Length of each clip in seconds; must be at least 10.
        sample_rate: Output sample rate in Hz.
        seed: Seed for deterministic record and window selection.
        attempts_per_record: Candidate windows to try from each record.

    Returns:
        Accepted real ECG demo clips, each sampled at ``sample_rate``.

    Raises:
        FileNotFoundError: If no canonical MIT-BIH H5 records are available.
        ValueError: If parameters are invalid or too few records pass the
            quality gate.
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
        raise FileNotFoundError(f"No canonical MIT-BIH H5 records found in {dataset_dir}")

    rng = np.random.default_rng(seed)
    selected: list[EcgDemoClip] = []
    for record_path in rng.permutation(records):
        source_sample_rate, record_id = _record_metadata(record_path)
        source_samples = load_ecg_signal(record_path, lead_index=0)
        source_length = round(duration_seconds * source_sample_rate)
        if source_samples.size < source_length:
            continue

        for _attempt in range(attempts_per_record):
            max_start = source_samples.size - source_length
            start = int(rng.integers(0, max_start + 1))
            source_window = source_samples[start : start + source_length]
            output_window = _resample(source_window, source_sample_rate, sample_rate)
            try:
                quality = assess_ecg_demo_clip(output_window, sample_rate=sample_rate)
            except ValueError:
                continue
            selected.append(
                EcgDemoClip(
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
        f"Selected only {len(selected)} of {num_clips} quality-gated ECG clips from {dataset_dir}; "
        "relax the quality gate or use a larger source dataset."
    )


def export_ecg_demo_csvs(
    output_dir: str | Path,
    *,
    dataset_dir: str | Path,
    num_clips: int = 10,
    duration_seconds: float = 30.0,
    sample_rate: int = 256,
    seed: int = 42,
) -> EcgDemoExport:
    """Write real, quality-gated MIT-BIH ECG clips as CSV files and a manifest.

    Each CSV has ``sample_index``, ``time_s``, and ``ecg`` columns. The JSON
    manifest records source attribution, record IDs, source offsets, and the
    quality measurements used to accept each clip.

    Args:
        output_dir: Destination directory for demo artifacts.
        dataset_dir: Canonical MIT-BIH H5 directory.
        num_clips: Number of clips to write, from 1 through 10.
        duration_seconds: Length of each clip in seconds.
        sample_rate: Output sample rate in Hz.
        seed: Seed for deterministic record and window selection.

    Returns:
        The written CSV paths and quality-manifest path.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    clips = generate_ecg_demo_clips(
        dataset_dir,
        num_clips=num_clips,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        seed=seed,
    )

    csv_paths: list[Path] = []
    manifest_clips: list[dict[str, object]] = []
    for index, clip in enumerate(clips, start=1):
        csv_path = destination / f"ecg_demo_{index:02d}.csv"
        sample_indices = np.arange(clip.signal.size, dtype=np.int64)
        times = sample_indices / float(sample_rate)
        rows = np.column_stack((sample_indices, times, clip.signal))
        np.savetxt(
            csv_path,
            rows,
            delimiter=",",
            header="sample_index,time_s,ecg",
            comments="",
            fmt=["%d", "%.9f", "%.8g"],
        )
        csv_paths.append(csv_path)
        manifest_clips.append(
            {
                "file": csv_path.name,
                "num_samples": int(clip.signal.size),
                "duration_seconds": clip.signal.size / float(sample_rate),
                "source_record": clip.source_record,
                "source_sample_rate": clip.source_sample_rate,
                "start_seconds": clip.start_seconds,
                "quality": asdict(clip.quality),
            }
        )

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "modality": "ecg",
                "source": {
                    "dataset": "MIT-BIH Arrhythmia Database v1.0.0",
                    "license": "ODC-By-1.0",
                    "citation": "Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. "
                    "IEEE Eng Med Biol. 2001;20(3):45-50.",
                },
                "sample_rate": sample_rate,
                "num_clips": len(clips),
                "duration_seconds": duration_seconds,
                "seed": seed,
                "clips": manifest_clips,
            },
            indent=2,
        )
        + "\n"
    )
    return EcgDemoExport(csv_paths=csv_paths, manifest_path=manifest_path)


__all__ = [
    "EcgDemoClip",
    "EcgDemoExport",
    "EcgDemoQuality",
    "assess_ecg_demo_clip",
    "export_ecg_demo_csvs",
    "generate_ecg_demo_clips",
]
