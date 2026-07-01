"""Unified per-source PPG TFRecord cache builder and loader.

Build once per dataset, train from any combination.  Each source slug gets
its own cache directory with ``train.tfrecord``, ``val.tfrecord``, and
``metadata.json``.  Windows are stored **raw** (unnormalized) so
normalization can be chosen at training time.

Supported source types:

* ``edf`` — MESA polysomnography EDF files (channel ``Pleth``).
* ``h5``  — Canonical h5 files (``bidmc``, ``ppg_dalia``, ``wesad``, ``butppg``).

Example CLI usage::

    python scripts/build_ppg_cache.py --sources mesa bidmc ppg_dalia wesad butppg

Example training YAML::

    data:
      unified_cache:
        enabled: true
        cache_root: datasets/ppg_cache
        sources:
          - slug: mesa
            weight: 0.4
          - slug: ppg_dalia
            weight: 0.3
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from compressionkit.configs.paths import default_datasets_dir

logger = logging.getLogger("ppg-cache")

# ---------------------------------------------------------------------------
# Known source registry
# ---------------------------------------------------------------------------

KNOWN_SOURCES: dict[str, dict[str, Any]] = {
    "mesa": {
        "type": "edf",
        "glob": "mesa-commercial-use/polysomnography/edfs/*.edf",
        "target_label": "Pleth",
        "offset_samples": 192,
    },
    "bidmc": {"type": "h5"},
    "ppg_dalia": {"type": "h5"},
    "wesad": {"type": "h5"},
    "butppg": {"type": "h5", "butppg_quality_only": True},
}


# ---------------------------------------------------------------------------
# Build-time config
# ---------------------------------------------------------------------------


@dataclass
class CacheBuildConfig:
    """Parameters for building one source cache."""

    slug: str
    datasets_root: str = field(default_factory=lambda: default_datasets_dir())
    cache_root: str = "datasets/ppg_cache"
    target_fs: int = 64
    frame_size: int = 320
    train_frac: float = 0.8
    val_frac: float = 0.1
    split_seed: int = 42
    max_windows_per_file: int | None = None
    force_rebuild: bool = False
    # Sanitization
    sanitize: bool = True
    min_std: float = 1e-4
    max_saturation_frac: float = 0.10
    max_abs_z: float = 8.0
    max_outlier_frac: float = 0.02


# ---------------------------------------------------------------------------
# Patient split (reusable hash-based)
# ---------------------------------------------------------------------------


def _patient_split(
    patient_id: str,
    source: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> str:
    """Deterministic patient-disjoint split via SHA-1 hash."""
    digest = hashlib.sha1(f"{source}|{patient_id}|{seed}".encode()).digest()
    val = int.from_bytes(digest[:8], "big") / 2**64
    if val < train_frac:
        return "train"
    if val < train_frac + val_frac:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# TFRecord serialization
# ---------------------------------------------------------------------------


def _serialize_window(signal: np.ndarray) -> bytes:
    """Serialize one 1-D float32 window into a TF Example."""
    flat = np.asarray(signal, dtype=np.float32).ravel()
    feat = {
        "signal": tf.train.Feature(float_list=tf.train.FloatList(value=flat.tolist())),
    }
    return tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()


# ---------------------------------------------------------------------------
# Window extraction helpers
# ---------------------------------------------------------------------------


def _subsample_evenly(windows: list[np.ndarray], max_n: int) -> list[np.ndarray]:
    """Deterministically pick *max_n* windows spread evenly across the list."""
    if len(windows) <= max_n:
        return windows
    step = len(windows) / max_n
    return [windows[int(i * step)] for i in range(max_n)]


def _extract_edf_windows(
    edf_path: Path,
    *,
    target_fs: int,
    frame_size: int,
    offset_samples: int,
    target_label: str,
    san_cfg: Any | None,
    max_windows: int | None,
) -> list[np.ndarray]:
    """Extract sliding windows from one EDF file."""
    from compressionkit.datasets.ppg import load_ppg_signal

    try:
        signal = load_ppg_signal(
            str(edf_path),
            target_rate=target_fs,
            offset_samples=offset_samples,
            target_label=target_label,
        )
    except Exception as exc:
        logger.warning("Skipping %s: %s", edf_path.name, exc)
        return []

    windows: list[np.ndarray] = []
    n = len(signal)
    for start in range(0, max(0, n - frame_size + 1), frame_size):
        w = signal[start : start + frame_size]
        if san_cfg is not None:
            from compressionkit.preprocessing.sanitize import is_clean_window

            if not is_clean_window(w[np.newaxis, :], san_cfg).ok:
                continue
        windows.append(w)

    if max_windows is not None:
        windows = _subsample_evenly(windows, max_windows)
    return windows


def _extract_h5_windows(
    h5_path: Path,
    *,
    target_fs: int,
    frame_size: int,
    slug: str,
    san_cfg: Any | None,
    max_windows: int | None,
    butppg_quality_only: bool = True,
) -> tuple[str, list[np.ndarray]]:
    """Extract sliding windows from one H5 file.

    Returns:
        ``(patient_id, windows)`` tuple.
    """
    import h5py
    from scipy.signal import resample_poly

    with h5py.File(h5_path, "r") as h:
        # Quality check for butppg
        if slug == "butppg" and butppg_quality_only:
            q = h.attrs.get("quality", 1)
            if int(q) == 0:
                return h5_path.stem, []

        fs_in = int(h.attrs.get("fs", target_fs))
        signal = h["data"][0].astype(np.float32, copy=False)

        pid = h.attrs.get("patient_id", h5_path.stem)
        if isinstance(pid, bytes):
            pid = pid.decode()
        pid = str(pid)

    # Resample
    if fs_in != target_fs:
        g = math.gcd(fs_in, target_fs)
        signal = resample_poly(signal, target_fs // g, fs_in // g).astype(np.float32)

    windows: list[np.ndarray] = []
    n = len(signal)
    for start in range(0, max(0, n - frame_size + 1), frame_size):
        w = signal[start : start + frame_size]
        if san_cfg is not None:
            from compressionkit.preprocessing.sanitize import is_clean_window

            if not is_clean_window(w[np.newaxis, :], san_cfg).ok:
                continue
        windows.append(w)

    if max_windows is not None:
        windows = _subsample_evenly(windows, max_windows)
    return pid, windows


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------


def build_source_cache(cfg: CacheBuildConfig) -> tuple[Path, dict[str, Any]]:
    """Build TFRecord cache for one PPG source.

    Args:
        cfg: Build configuration for this source.

    Returns:
        ``(cache_dir, metadata)`` tuple.
    """
    from compressionkit.preprocessing.sanitize import SanitizeConfig

    slug = cfg.slug
    source_info = KNOWN_SOURCES.get(slug)
    if source_info is None:
        raise ValueError(f"Unknown source slug {slug!r}. Known: {sorted(KNOWN_SOURCES)}")

    cache_dir = Path(cfg.cache_root) / slug
    if not cache_dir.is_absolute():
        cache_dir = cache_dir.resolve()
    metadata_path = cache_dir / "metadata.json"

    # Check for existing cache
    if metadata_path.exists() and not cfg.force_rebuild:
        with metadata_path.open() as f:
            metadata = json.load(f)
        # Validate key params match
        if (
            metadata.get("target_fs") == cfg.target_fs
            and metadata.get("frame_size") == cfg.frame_size
            and metadata.get("split_seed") == cfg.split_seed
        ):
            logger.info(
                "Cache exists for %s: %d train, %d val windows",
                slug,
                metadata["train_examples"],
                metadata["val_examples"],
            )
            return cache_dir, metadata
        logger.info("Cache params changed for %s, rebuilding", slug)

    cache_dir.mkdir(parents=True, exist_ok=True)

    san_cfg = (
        SanitizeConfig(
            min_std=cfg.min_std,
            max_saturation_frac=cfg.max_saturation_frac,
            max_abs_z=cfg.max_abs_z,
            max_outlier_frac=cfg.max_outlier_frac,
        )
        if cfg.sanitize
        else None
    )

    source_type = source_info["type"]
    datasets_root = Path(cfg.datasets_root)

    if source_type == "edf":
        train_count, val_count, n_files = _build_edf_source(
            slug=slug,
            cache_dir=cache_dir,
            datasets_root=datasets_root,
            target_fs=cfg.target_fs,
            frame_size=cfg.frame_size,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            split_seed=cfg.split_seed,
            max_windows_per_file=cfg.max_windows_per_file,
            san_cfg=san_cfg,
            target_label=source_info.get("target_label", "Pleth"),
            offset_samples=source_info.get("offset_samples", 0),
        )
    elif source_type == "h5":
        train_count, val_count, n_files = _build_h5_source(
            slug=slug,
            cache_dir=cache_dir,
            datasets_root=datasets_root,
            target_fs=cfg.target_fs,
            frame_size=cfg.frame_size,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            split_seed=cfg.split_seed,
            max_windows_per_file=cfg.max_windows_per_file,
            san_cfg=san_cfg,
            butppg_quality_only=source_info.get("butppg_quality_only", True),
        )
    else:
        raise ValueError(f"Unsupported source type: {source_type!r}")

    metadata: dict[str, Any] = {
        "version": 2,
        "slug": slug,
        "source_type": source_type,
        "target_fs": cfg.target_fs,
        "frame_size": cfg.frame_size,
        "split_seed": cfg.split_seed,
        "train_frac": cfg.train_frac,
        "val_frac": cfg.val_frac,
        "max_windows_per_file": cfg.max_windows_per_file,
        "train_examples": train_count,
        "val_examples": val_count,
        "num_source_files": n_files,
        "sanitize": cfg.sanitize,
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Built cache for %s: %d train, %d val windows from %d files",
        slug,
        train_count,
        val_count,
        n_files,
    )
    return cache_dir, metadata


def _build_edf_source(
    *,
    slug: str,
    cache_dir: Path,
    datasets_root: Path,
    target_fs: int,
    frame_size: int,
    train_frac: float,
    val_frac: float,
    split_seed: int,
    max_windows_per_file: int | None,
    san_cfg: Any | None,
    target_label: str,
    offset_samples: int,
) -> tuple[int, int, int]:
    """Build TFRecord cache from EDF files (MESA)."""
    source_info = KNOWN_SOURCES[slug]
    glob_pattern = source_info["glob"]
    edf_files = sorted(datasets_root.glob(glob_pattern))
    if not edf_files:
        raise FileNotFoundError(f"No EDF files for {slug}: {datasets_root / glob_pattern}")

    train_path = cache_dir / "train.tfrecord"
    val_path = cache_dir / "val.tfrecord"
    train_count = val_count = 0

    with (
        tf.io.TFRecordWriter(str(train_path)) as train_writer,
        tf.io.TFRecordWriter(str(val_path)) as val_writer,
    ):
        for i, edf_path in enumerate(edf_files):
            patient_id = edf_path.stem
            bucket = _patient_split(
                patient_id,
                slug,
                train_frac=train_frac,
                val_frac=val_frac,
                seed=split_seed,
            )
            if bucket == "test":
                continue

            writer = train_writer if bucket == "train" else val_writer
            windows = _extract_edf_windows(
                edf_path,
                target_fs=target_fs,
                frame_size=frame_size,
                offset_samples=offset_samples,
                target_label=target_label,
                san_cfg=san_cfg,
                max_windows=max_windows_per_file,
            )
            for w in windows:
                writer.write(_serialize_window(w))
                if bucket == "train":
                    train_count += 1
                else:
                    val_count += 1

            if (i + 1) % 100 == 0:
                logger.info(
                    "  %s: %d/%d files processed (%d train, %d val)",
                    slug,
                    i + 1,
                    len(edf_files),
                    train_count,
                    val_count,
                )

    return train_count, val_count, len(edf_files)


def _build_h5_source(
    *,
    slug: str,
    cache_dir: Path,
    datasets_root: Path,
    target_fs: int,
    frame_size: int,
    train_frac: float,
    val_frac: float,
    split_seed: int,
    max_windows_per_file: int | None,
    san_cfg: Any | None,
    butppg_quality_only: bool,
) -> tuple[int, int, int]:
    """Build TFRecord cache from H5 files."""
    h5_dir = datasets_root / slug
    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files for {slug}: {h5_dir}")

    train_path = cache_dir / "train.tfrecord"
    val_path = cache_dir / "val.tfrecord"
    train_count = val_count = 0

    with (
        tf.io.TFRecordWriter(str(train_path)) as train_writer,
        tf.io.TFRecordWriter(str(val_path)) as val_writer,
    ):
        for i, h5_path in enumerate(h5_files):
            pid, windows = _extract_h5_windows(
                h5_path,
                target_fs=target_fs,
                frame_size=frame_size,
                slug=slug,
                san_cfg=san_cfg,
                max_windows=max_windows_per_file,
                butppg_quality_only=butppg_quality_only,
            )
            if not windows:
                continue

            bucket = _patient_split(
                pid,
                slug,
                train_frac=train_frac,
                val_frac=val_frac,
                seed=split_seed,
            )
            if bucket == "test":
                continue

            writer = train_writer if bucket == "train" else val_writer
            for w in windows:
                writer.write(_serialize_window(w))
                if bucket == "train":
                    train_count += 1
                else:
                    val_count += 1

            if (i + 1) % 50 == 0:
                logger.info(
                    "  %s: %d/%d files processed (%d train, %d val)",
                    slug,
                    i + 1,
                    len(h5_files),
                    train_count,
                    val_count,
                )

    return train_count, val_count, len(h5_files)


# ---------------------------------------------------------------------------
# Unified dataset loader
# ---------------------------------------------------------------------------


@dataclass
class SourceWeight:
    """One source in a multi-source training mix."""

    slug: str
    weight: float | None = None  # None → proportional to window count


def load_cache_metadata(cache_root: Path, slug: str) -> dict[str, Any]:
    """Load metadata.json for a cached source."""
    meta_path = cache_root / slug / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No cache found for {slug!r} at {meta_path}. Run: python scripts/build_ppg_cache.py --source {slug}"
        )
    with meta_path.open() as f:
        return json.load(f)


def make_cached_ppg_dataset(
    sources: list[SourceWeight],
    *,
    cache_root: Path,
    frame_size: int,
    batch_size: int = 64,
    shuffle_buffer: int = 10_000,
    epsilon: float = 1e-3,
    split: str = "train",
    seed: int = 42,
    normalize: bool = True,
) -> tuple[tf.data.Dataset, dict[str, Any]]:
    """Build a tf.data pipeline from any combination of cached PPG sources.

    Each source is read from its TFRecord cache and optionally weighted.
    Windows are layer-normalized at training time (not stored normalized).

    Args:
        sources: List of source slugs with optional per-source weights.
        cache_root: Root directory containing per-slug cache subdirectories.
        frame_size: Expected window size (must match cache frame_size).
        batch_size: Batch size.
        shuffle_buffer: Shuffle buffer size (0 for val to keep deterministic).
        epsilon: Layer-norm epsilon.
        split: ``"train"`` or ``"val"``.
        seed: Random seed for shuffling and sampling.
        normalize: When ``True`` (default), windows are layer-normalized before
            being returned. Set ``False`` to return raw (unnormalized) pairs so a
            downstream augmenter can corrupt the raw signal and normalize itself
            (apply-before-norm policy).

    Returns:
        ``(dataset, info_dict)`` where dataset yields ``(x, x)`` tuples
        shaped ``(B, 1, frame_size, 1)`` and info_dict has per-source
        metadata and effective weights.
    """
    if not sources:
        raise ValueError("At least one source is required")

    cache_root = Path(cache_root)
    if not cache_root.is_absolute():
        cache_root = cache_root.resolve()

    datasets: list[tf.data.Dataset] = []
    weights: list[float] = []
    info: dict[str, Any] = {"sources": {}, "split": split}
    count_key = "train_examples" if split == "train" else "val_examples"

    for src in sources:
        meta = load_cache_metadata(cache_root, src.slug)

        # Validate frame_size matches
        if meta["frame_size"] != frame_size:
            raise ValueError(f"Cache frame_size for {src.slug} is {meta['frame_size']}, expected {frame_size}")

        tfrecord_path = cache_root / src.slug / f"{split}.tfrecord"
        if not tfrecord_path.exists():
            raise FileNotFoundError(f"Missing {tfrecord_path}")

        n_examples = meta[count_key]
        info["sources"][src.slug] = {
            "examples": n_examples,
            "weight_raw": src.weight,
        }

        ds = tf.data.TFRecordDataset(
            str(tfrecord_path),
            num_parallel_reads=tf.data.AUTOTUNE,
        )
        if split == "train":
            ds = ds.shuffle(
                buffer_size=min(shuffle_buffer, max(1, n_examples)),
                seed=seed,
                reshuffle_each_iteration=True,
            )
            ds = ds.repeat()

        datasets.append(ds)
        weights.append(src.weight if src.weight is not None else float(n_examples))

    # Normalize weights
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    for src, w in zip(sources, weights):
        info["sources"][src.slug]["weight_effective"] = round(w, 4)

    # Combine datasets
    if len(datasets) == 1:
        combined = datasets[0]
    elif split == "train":
        combined = tf.data.Dataset.sample_from_datasets(
            datasets,
            weights=weights,
            seed=seed,
            stop_on_empty_dataset=False,
        )
    else:
        # For validation: concatenate all sources (no sampling)
        combined = datasets[0]
        for ds in datasets[1:]:
            combined = combined.concatenate(ds)

    # Parse TFRecords
    spec = {"signal": tf.io.FixedLenFeature([frame_size], tf.float32)}

    def _parse(raw: tf.Tensor) -> tf.Tensor:
        rec = tf.io.parse_single_example(raw, spec)
        return rec["signal"]

    combined = combined.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    combined = combined.batch(batch_size, drop_remainder=True)

    # Layer-norm + reshape → (B, 1, frame_size, 1) autoencoder pairs
    def _normalize_and_reshape(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # x: (B, frame_size)
        x = tf.expand_dims(x, axis=-1)  # (B, frame_size, 1)
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
        x = (x - mean) / tf.sqrt(var + epsilon)
        x = tf.reshape(x, [-1, 1, frame_size, 1])  # (B, 1, frame_size, 1)
        return x, x

    def _reshape_only(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # Raw pairs for apply-before-norm augmentation downstream.
        x = tf.reshape(x, [-1, 1, frame_size, 1])
        return x, x

    map_fn = _normalize_and_reshape if normalize else _reshape_only
    combined = combined.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    combined = combined.prefetch(tf.data.AUTOTUNE)

    return combined, info


def load_cached_raw_windows(
    sources: list[SourceWeight],
    *,
    cache_root: Path,
    frame_size: int,
    split: str = "train",
    max_windows: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Load raw (unnormalized) windows from per-source caches as a numpy array.

    Useful for pipelines that need in-memory arrays (e.g. two-stream
    decomposition). Sources are sampled proportionally to their weights.

    Args:
        sources: List of source slugs with optional per-source weights.
        cache_root: Root directory containing per-slug cache subdirectories.
        frame_size: Expected window size (must match cache frame_size).
        split: ``"train"`` or ``"val"``.
        max_windows: Optional cap on total number of windows loaded.
        seed: Random seed for subsampling.

    Returns:
        Array of shape ``(N, frame_size)`` with raw float32 windows.
    """
    cache_root = Path(cache_root)
    if not cache_root.is_absolute():
        cache_root = cache_root.resolve()

    count_key = "train_examples" if split == "train" else "val_examples"
    spec = {"signal": tf.io.FixedLenFeature([frame_size], tf.float32)}

    # Determine per-source budgets
    raw_weights: list[float] = []
    metas: list[dict] = []
    for src in sources:
        meta = load_cache_metadata(cache_root, src.slug)
        if meta["frame_size"] != frame_size:
            raise ValueError(f"Cache frame_size for {src.slug} is {meta['frame_size']}, expected {frame_size}")
        metas.append(meta)
        raw_weights.append(src.weight if src.weight is not None else float(meta[count_key]))

    total_w = sum(raw_weights)
    norm_weights = [w / total_w for w in raw_weights]

    all_windows: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for src, meta, w in zip(sources, metas, norm_weights):
        tfrecord_path = cache_root / src.slug / f"{split}.tfrecord"
        if not tfrecord_path.exists():
            raise FileNotFoundError(f"Missing {tfrecord_path}")

        n_available = meta[count_key]
        if max_windows is not None:
            budget = math.ceil(max_windows * w)
        else:
            budget = n_available

        # Read all windows from this source
        ds = tf.data.TFRecordDataset(str(tfrecord_path))
        ds = ds.map(
            lambda raw: tf.io.parse_single_example(raw, spec)["signal"],
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(2048)

        source_windows: list[np.ndarray] = []
        for batch in ds:
            source_windows.append(batch.numpy())
        source_arr = np.concatenate(source_windows, axis=0)

        # Subsample if needed
        if len(source_arr) > budget:
            idx = rng.choice(len(source_arr), budget, replace=False)
            source_arr = source_arr[idx]

        all_windows.append(source_arr)
        logger.info(
            "Loaded %d %s windows from %s cache",
            len(source_arr),
            split,
            src.slug,
        )

    combined = np.concatenate(all_windows, axis=0)

    # Final cap
    if max_windows is not None and len(combined) > max_windows:
        idx = rng.choice(len(combined), max_windows, replace=False)
        combined = combined[idx]

    # Shuffle combined array
    rng.shuffle(combined)
    return combined
