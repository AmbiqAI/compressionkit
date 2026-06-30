"""Run a catalog method on a user-supplied signal.

This is the reusable engine behind ``compressionkit playbook run`` and
``compressionkit playbook compare``. It frames a 1-D signal, runs any
*runnable* (self-contained) :class:`~compressionkit.playbook.catalog.MethodCard`
through encode/decode, and reports the dual-reference scorecard:

* ``faithful_prd`` — PRD against the raw input (how faithful, noise included).
* ``proxy_prd`` — PRD against a band-limited clean proxy (how well the
  underlying physiological signal is recovered).
* ``true_cr`` — realized compression ratio from actual encoded bits.

Imprint / hallucination probes require a controlled clean+artifact reference
and live in the benchmark harness, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compressionkit.dsp.filters import clean_proxy
from compressionkit.evaluation.metrics import compute_signal_metrics
from compressionkit.playbook.catalog import MethodCard

# Modality defaults used when the caller does not override them.
_DEFAULTS: dict[str, tuple[int, int]] = {
    # modality: (sample_rate, frame_size)
    "ecg": (256, 512),
    "ppg": (64, 320),
}


@dataclass(frozen=True)
class RunResult:
    """Outcome of running one method on one signal."""

    method_id: str
    modality: str
    sample_rate: int
    frame_size: int
    target_cr: float
    true_cr: float
    faithful_prd: float
    proxy_prd: float
    n_frames: int
    original: np.ndarray
    reconstruction: np.ndarray
    proxy: np.ndarray


def resolve_defaults(
    modality: str,
    sample_rate: int | None,
    frame_size: int | None,
) -> tuple[int, int]:
    """Fill in modality-default ``(sample_rate, frame_size)`` where unset."""
    sr_default, fs_default = _DEFAULTS.get(modality, _DEFAULTS["ecg"])
    return sample_rate or sr_default, frame_size or fs_default


def _frame_signal(signal: np.ndarray, frame_size: int) -> np.ndarray:
    """Split a 1-D signal into non-overlapping frames, zero-padding the tail."""
    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    n_frames = int(np.ceil(arr.size / frame_size))
    padded = np.zeros(n_frames * frame_size, dtype=np.float32)
    padded[: arr.size] = arr
    return padded.reshape(n_frames, frame_size)


def run_method_on_signal(
    card: MethodCard,
    signal: np.ndarray,
    *,
    modality: str,
    target_cr: float,
    sample_rate: int | None = None,
    frame_size: int | None = None,
) -> RunResult:
    """Encode/decode ``signal`` with ``card`` and score the reconstruction.

    Args:
        card: A *runnable* method card (``card.runnable`` must be True).
        signal: 1-D input samples.
        modality: ``"ecg"`` or ``"ppg"`` (selects clean-proxy band + defaults).
        target_cr: Target compression ratio passed to the codec.
        sample_rate: Override the modality default sample rate.
        frame_size: Override the modality default frame size.

    Returns:
        A :class:`RunResult` with the dual-reference scorecard and signals.

    Raises:
        ValueError: If the method is not runnable without trained weights.
    """
    if not card.runnable:
        raise ValueError(
            f"Method {card.id!r} needs trained weights and cannot be built standalone. "
            "Run its golden experiment or load it from a run directory."
        )

    sample_rate, frame_size = resolve_defaults(modality, sample_rate, frame_size)
    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    codec = card.builder(  # type: ignore[misc]
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
        modality=modality,
    )

    frames = _frame_signal(arr, frame_size)
    recon_frames: list[np.ndarray] = []
    total_bits = 0
    for frame in frames:
        encoded = codec.encode(frame)
        recon_frames.append(np.asarray(codec.decode(encoded), dtype=np.float32).reshape(-1))
        total_bits += int(encoded.nbits)

    reconstruction = np.concatenate(recon_frames)[: arr.size]
    raw_bits = arr.size * 16
    true_cr = float(raw_bits / total_bits) if total_bits > 0 else float("inf")

    proxy = clean_proxy(arr, sample_rate, modality)
    faithful_prd = compute_signal_metrics(arr, reconstruction)["prd_percent"]
    proxy_prd = compute_signal_metrics(proxy, reconstruction)["prd_percent"]

    return RunResult(
        method_id=card.id,
        modality=modality,
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
        true_cr=true_cr,
        faithful_prd=faithful_prd,
        proxy_prd=proxy_prd,
        n_frames=frames.shape[0],
        original=arr,
        reconstruction=reconstruction,
        proxy=proxy,
    )


def load_signal(path: str) -> np.ndarray:
    """Load a 1-D signal from ``.npy``, ``.npz`` (first/``signal`` array), or ``.csv``."""
    from pathlib import Path

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p)
    elif suffix == ".npz":
        npz = np.load(p)
        key = "signal" if "signal" in npz else next(iter(npz.keys()))
        arr = npz[key]
    elif suffix in (".csv", ".txt"):
        arr = np.loadtxt(p, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"Unsupported data format {suffix!r}; use .npy, .npz, or .csv")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[:, 0] if arr.shape[0] >= arr.shape[1] else arr[0]
    return arr.reshape(-1)


__all__ = [
    "RunResult",
    "load_signal",
    "resolve_defaults",
    "run_method_on_signal",
]
