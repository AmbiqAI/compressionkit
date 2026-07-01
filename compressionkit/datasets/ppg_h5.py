"""H5-backed PPG dataset loader for the canonical compressionkit layout.

Iterates per-patient ``.h5`` files produced by the ``scripts/datasets/download_*``
ingestion scripts. Each file follows the convention:

* ``data`` dataset of shape ``(channels, samples)`` (``channels==1`` for PPG).
* attrs ``fs`` (Hz), ``patient_id``, ``source``, ``acquisition``.

The loader is intentionally minimal: enumerate windows, robust-normalize, and
yield a ``tf.data`` pipeline keyed by ``data`` / ``source``. Sanitization is
applied per-window via :mod:`compressionkit.preprocessing.sanitize`.

PPG-only — :class:`PpgH5Source` ignores ECG / ACC / label groups in the same
file. Multi-modal usage can read those fields directly with :func:`iter_windows`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from scipy.signal import resample_poly

from compressionkit.configs.paths import default_datasets_dir
from compressionkit.preprocessing.sanitize import (
    SanitizeConfig,
    is_clean_window,
    normalize_window,
)

# Canonical roots — adjust via :class:`PpgH5Source` if your layout differs.
DEFAULT_DATASET_ROOT = Path(default_datasets_dir())


@dataclass(frozen=True)
class PpgH5Source:
    """Description of a single PPG h5 dataset on disk.

    Attributes:
        slug: Directory name under ``root`` (e.g. ``"bidmc"``, ``"ppg_dalia"``).
        root: Parent folder containing per-dataset subdirectories.
        glob: File glob (default ``*.h5``).
        butppg_quality_only: When True and ``slug == "butppg"``, drop sessions
            whose ``quality`` attr is 0 (poor signal).
    """

    slug: str
    root: Path = DEFAULT_DATASET_ROOT
    glob: str = "*.h5"
    butppg_quality_only: bool = True

    def files(self) -> list[Path]:
        return sorted((self.root / self.slug).glob(self.glob))


@dataclass
class WindowSpec:
    """Window extraction config common to all sources.

    Attributes:
        target_fs: Target sample rate in Hz. Sources are resampled with
            polyphase filtering when their native ``fs`` differs.
        window_seconds: Window length in seconds.
        hop_seconds: Hop between successive windows (defaults to window length
            for non-overlapping windows).
        sanitize: Sanitize config; ``None`` disables window rejection.
        normalize: Apply robust per-window z-score (median + MAD).
    """

    target_fs: int = 64
    window_seconds: float = 4.0
    hop_seconds: float | None = None
    sanitize: SanitizeConfig | None = field(default_factory=SanitizeConfig)
    normalize: bool = True

    @property
    def window_samples(self) -> int:
        return round(self.window_seconds * self.target_fs)

    @property
    def hop_samples(self) -> int:
        hop = self.hop_seconds if self.hop_seconds is not None else self.window_seconds
        return max(1, round(hop * self.target_fs))


def _resample(signal: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Polyphase resample a 1-D signal. Returns float32."""
    if fs_in == fs_out:
        return signal.astype(np.float32, copy=False)
    # Reduce the rational factor for filter efficiency.
    g = np.gcd(int(fs_in), int(fs_out))
    up = fs_out // g
    down = fs_in // g
    return resample_poly(signal, up, down).astype(np.float32, copy=False)


def _patient_id(h5: h5py.File, fallback: str) -> str:
    pid = h5.attrs.get("patient_id", fallback)
    if isinstance(pid, bytes):
        pid = pid.decode()
    return str(pid)


def _should_skip(source: PpgH5Source, h5: h5py.File) -> bool:
    if source.slug == "butppg" and source.butppg_quality_only:
        q = h5.attrs.get("quality", 1)
        if int(q) == 0:
            return True
    return False


def patient_split(
    patient_id: str,
    *,
    source: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 1337,
) -> str:
    """Deterministic patient-disjoint split.

    Hashes ``(source, patient_id, seed)`` into ``[0, 1)`` and returns
    ``"train" | "val" | "test"`` based on the cumulative fractions. Same
    patient always lands in the same bucket regardless of caller.

    Args:
        patient_id: Subject identifier (from h5 attrs).
        source: Dataset slug, included so two datasets with overlapping
            patient ids (rare) do not collide.
        train_frac: Train fraction in ``[0, 1]``.
        val_frac: Val fraction in ``[0, 1 - train_frac]``.
        seed: Hash seed.
    """
    digest = hashlib.sha1(f"{source}|{patient_id}|{seed}".encode()).digest()
    # First 8 bytes → uint64 → [0, 1).
    val = int.from_bytes(digest[:8], "big") / 2**64
    if val < train_frac:
        return "train"
    if val < train_frac + val_frac:
        return "val"
    return "test"


@dataclass(frozen=True)
class SplitConfig:
    """Patient-disjoint split parameters used by :func:`iter_windows`."""

    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 1337
    split: str | None = None  # "train" / "val" / "test" or None for all


def iter_windows(
    sources: Iterable[PpgH5Source],
    spec: WindowSpec,
    split_cfg: SplitConfig | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield per-window dicts ``{"data", "source", "patient_id"}``.

    Windows that fail :func:`is_clean_window` are dropped silently. When
    ``split_cfg.split`` is set, only patients in that bucket are yielded.
    Use :func:`summarize_sources` first if you need reject-rate stats.
    """
    win = spec.window_samples
    hop = spec.hop_samples
    for src in sources:
        for path in src.files():
            with h5py.File(path, "r") as h:
                if _should_skip(src, h):
                    continue
                fs_in = int(h.attrs.get("fs", spec.target_fs))
                signal = h["data"][0].astype(np.float32, copy=False)  # PPG channel
                signal = _resample(signal, fs_in, spec.target_fs)
                pid = _patient_id(h, fallback=path.stem)
            if split_cfg is not None and split_cfg.split is not None:
                bucket = patient_split(
                    pid,
                    source=src.slug,
                    train_frac=split_cfg.train_frac,
                    val_frac=split_cfg.val_frac,
                    seed=split_cfg.seed,
                )
                if bucket != split_cfg.split:
                    continue
            n = signal.shape[0]
            for start in range(0, max(0, n - win + 1), hop):
                w = signal[start : start + win][np.newaxis, :]  # (1, win)
                if spec.sanitize is not None and not is_clean_window(w, spec.sanitize).ok:
                    continue
                if spec.normalize:
                    w = normalize_window(w)
                yield {
                    "data": w.astype(np.float32),
                    "source": src.slug,
                    "patient_id": pid,
                }


def summarize_sources(
    sources: Iterable[PpgH5Source],
    spec: WindowSpec,
    max_files_per_source: int | None = None,
) -> dict[str, dict[str, int | float]]:
    """Return per-source counts of total / kept / rejected windows.

    Useful for sanity-checking the sanitize thresholds before launching a long
    training run. ``max_files_per_source`` caps the audit cost.
    """
    win = spec.window_samples
    hop = spec.hop_samples
    out: dict[str, dict[str, int | float]] = {}
    for src in sources:
        files = src.files()
        if max_files_per_source is not None:
            files = files[:max_files_per_source]
        total = kept = 0
        reasons: dict[str, int] = {}
        for path in files:
            with h5py.File(path, "r") as h:
                if _should_skip(src, h):
                    continue
                fs_in = int(h.attrs.get("fs", spec.target_fs))
                signal = _resample(h["data"][0].astype(np.float32, copy=False), fs_in, spec.target_fs)
            n = signal.shape[0]
            for start in range(0, max(0, n - win + 1), hop):
                total += 1
                w = signal[start : start + win][np.newaxis, :]
                rep = is_clean_window(w, spec.sanitize) if spec.sanitize else None
                if rep is None or rep.ok:
                    kept += 1
                else:
                    reasons[rep.reason] = reasons.get(rep.reason, 0) + 1
        out[src.slug] = {
            "files": len(files),
            "total_windows": total,
            "kept_windows": kept,
            "reject_frac": (total - kept) / total if total else 0.0,
            **{f"reject_{k}": v for k, v in reasons.items()},
        }
    return out


def make_h5_ppg_dataset(
    sources: Iterable[PpgH5Source],
    spec: WindowSpec,
    *,
    batch_size: int = 32,
    shuffle_buffer: int = 1024,
    seed: int | None = None,
    split_cfg: SplitConfig | None = None,
) -> tf.data.Dataset:
    """Build a ``tf.data.Dataset`` of float32 windows shaped ``(B, 1, samples)``.

    The returned dataset yields a plain tensor (no labels), matching the
    autoencoder/RVQ training contract used elsewhere in the kit.
    """
    win = spec.window_samples
    sources = list(sources)

    def _gen():
        for sample in iter_windows(sources, spec, split_cfg=split_cfg):
            yield sample["data"]  # (1, win)

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=tf.TensorSpec(shape=(1, win), dtype=tf.float32),
    )
    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "PpgH5Source",
    "SplitConfig",
    "WindowSpec",
    "iter_windows",
    "make_h5_ppg_dataset",
    "patient_split",
    "summarize_sources",
]
