"""Dataset acquisition contract for golden experiments.

The :data:`DATASET_REGISTRY` maps the ``dataset_id`` declared by each
:class:`compressionkit.experiments.GoldenExperiment` to a factory that
returns a configured dataset instance. The golden lifecycle runner
(#25) uses :func:`ensure_dataset_available` as a pre-flight check
before training: if data is missing, it surfaces the exact remediation
command instead of failing deep inside the training loop.

See also issue #26.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from compressionkit.datasets.defines import DatasetInfo


class DatasetNotAvailableError(FileNotFoundError):
    """Raised when a dataset is not present on disk.

    Carries the remediation command a user should run to fetch it.
    """

    def __init__(self, dataset_id: str, path: Path, remediation: str) -> None:
        self.dataset_id = dataset_id
        self.path = path
        self.remediation = remediation
        super().__init__(f"Dataset {dataset_id!r} not available at {path}. To fetch it run:\n  {remediation}")


class _DatasetLike(Protocol):
    info: DatasetInfo
    path: Path

    def ensure_available(self) -> None: ...


def _ptbxl_factory(root: Path | None = None) -> _DatasetLike:
    from compressionkit.datasets.ptbxl import PtbxlDataset

    path = root / "ptbxl" if root is not None else Path("datasets/ptbxl")
    return PtbxlDataset(path=path)


def _mesa_factory(root: Path | None = None) -> _DatasetLike:
    from compressionkit.datasets.mesa import MesaDataset

    path = root / "mesa-commercial-use" if root is not None else Path("datasets/mesa-commercial-use")
    return MesaDataset(path=path)


_PPG_UNIFIED_STRICT_SANITIZE_INFO = DatasetInfo(
    name="ppg-unified-strict-sanitize-v1",
    sampling_rate=64,
    num_leads=1,
    description=(
        "Strict-sanitized unified 64 Hz PPG cache used by the faithful v1 PPG goldens "
        "(bidmc, butppg, ppg_dalia, wesad)."
    ),
    license="mixed/open-source terms",
    license_tier="open",
    requires_agreement=False,
)


@dataclass(frozen=True)
class UnifiedPpgCacheDataset:
    """Cache-backed PPG dataset contract for unified strict-sanitized goldens."""

    path: Path
    required_sources: tuple[str, ...] = ("bidmc", "butppg", "ppg_dalia", "wesad")
    required_splits: tuple[str, ...] = ("train.tfrecord", "val.tfrecord", "metadata.json")
    info: DatasetInfo = field(default_factory=lambda: _PPG_UNIFIED_STRICT_SANITIZE_INFO)

    def ensure_available(self) -> None:
        missing: list[str] = []
        for slug in self.required_sources:
            for rel in self.required_splits:
                candidate = self.path / slug / rel
                if not candidate.exists():
                    missing.append(str(candidate))
        if missing:
            remediation = (
                "uv run python scripts/build_ppg_cache.py --sources bidmc butppg ppg_dalia wesad "
                "--cache-root datasets/ppg_cache_strict_sanitize"
            )
            raise DatasetNotAvailableError(
                "ppg-unified-strict-sanitize-v1",
                self.path,
                remediation,
            )


def _ppg_unified_strict_sanitize_factory(root: Path | None = None) -> _DatasetLike:
    path = root / "ppg_cache_strict_sanitize" if root is not None else Path("datasets/ppg_cache_strict_sanitize")
    return UnifiedPpgCacheDataset(path=path)


# dataset_id → factory returning a dataset instance.
DATASET_REGISTRY: dict[str, Callable[[Path | None], _DatasetLike]] = {
    "ptb-xl": _ptbxl_factory,
    "mesa": _mesa_factory,
    "ppg-unified-strict-sanitize-v1": _ppg_unified_strict_sanitize_factory,
}


def resolve_dataset(dataset_id: str, root: Path | None = None) -> _DatasetLike:
    """Instantiate the dataset class registered for ``dataset_id``."""
    try:
        factory = DATASET_REGISTRY[dataset_id]
    except KeyError as err:
        raise KeyError(f"Unknown dataset_id {dataset_id!r}. Known: {sorted(DATASET_REGISTRY)}") from err
    return factory(root)


def ensure_dataset_available(dataset_id: str, root: Path | None = None) -> None:
    """Verify the dataset is on disk; raise :class:`DatasetNotAvailableError` otherwise.

    The dataset's own ``ensure_available()`` is invoked. Dataset
    implementations are expected to raise :class:`DatasetNotAvailableError`
    with a clear remediation message when data is missing.
    """
    resolve_dataset(dataset_id, root).ensure_available()


__all__ = [
    "DATASET_REGISTRY",
    "DatasetNotAvailableError",
    "ensure_dataset_available",
    "resolve_dataset",
]
