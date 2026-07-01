"""Trustworthy triple-metric benchmark on synthetic clean-truth ECG.

The generic ``run``/``compare`` path scores against a band-pass *proxy*, which
is biased toward band-pass denoisers. This module removes that bias by using a
*known* clean signal: it generates clean ECG with the McSharry model
(:func:`compressionkit.synthetic.ecg_mcsharry`), corrupts it with controlled
contact artifacts
(:func:`compressionkit.synthetic.ecg_contact_artifacts.simulate_contact_artifact_batch`),
and scores three complementary numbers:

* ``truth_prd`` — PRD vs the *true clean* signal (fidelity to the underlying
  physiology; unbiased — this is the metric the proxy could only approximate).
* ``faithful_prd`` — PRD vs the noisy *input* (fidelity to what was recorded,
  noise included).
* ``imprint`` — on pure-artifact input (no ECG present), how much periodic
  ECG-like structure the codec *invents*. Lower is better; a positive value
  means hallucination.

These three disagree by design; reporting all three is what lets us justify a
method as "robust" rather than chasing a single, gameable number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compressionkit.evaluation.metrics import compute_signal_metrics
from compressionkit.playbook.catalog import MethodCard
from compressionkit.synthetic.ecg_contact_artifacts import normalize_window, simulate_contact_artifact_batch
from compressionkit.synthetic.ecg_mcsharry import ecg_mcsharry


def make_clean_windows(
    n_windows: int,
    frame_size: int,
    sample_rate: int,
    *,
    hr_low: float = 50.0,
    hr_high: float = 90.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate ``n_windows`` clean ECG frames with varied heart rate."""
    rng = np.random.default_rng(seed)
    duration_s = frame_size / sample_rate
    out = np.empty((n_windows, frame_size), dtype=np.float32)
    for i in range(n_windows):
        hr = float(rng.uniform(hr_low, hr_high))
        sig = ecg_mcsharry(duration_s, sample_rate, hr_mean=hr, hr_std=1.0, seed=int(rng.integers(1 << 31)))
        out[i] = np.asarray(sig, dtype=np.float32)[:frame_size]
    return out


def autocorr_structure(x: np.ndarray, sample_rate: float, *, hr_low: float = 40.0, hr_high: float = 180.0) -> float:
    """Peak normalized autocorrelation in the plausible heart-rate lag band.

    A near-zero value means no periodic structure (pure noise); a high value
    means strong beat-like periodicity. Used to detect invented structure.
    """
    arr = np.asarray(x, dtype=np.float64)
    arr = arr - arr.mean()
    n = arr.size
    ac = np.correlate(arr, arr, mode="full")[n - 1 :]
    ac = ac / (ac[0] + 1e-12)
    lo_lag = max(1, int(sample_rate * 60.0 / hr_high))
    hi_lag = min(n - 1, int(sample_rate * 60.0 / hr_low))
    if lo_lag >= hi_lag:
        return 0.0
    return float(np.max(ac[lo_lag:hi_lag]))


@dataclass(frozen=True)
class BenchmarkResult:
    """Triple-metric outcome for one method on the synthetic-truth battery."""

    method_id: str
    truth_prd: float
    faithful_prd: float
    imprint: float
    true_cr: float
    family: str
    severity: float


def benchmark_method(
    card: MethodCard,
    clean: np.ndarray,
    corrupted: np.ndarray,
    pure_artifact: np.ndarray,
    *,
    sample_rate: int,
    frame_size: int,
    target_cr: float,
    family: str,
    severity: float,
) -> BenchmarkResult:
    """Score one runnable method with the triple metric (medians over windows)."""
    if not card.runnable:
        raise ValueError(f"Method {card.id!r} needs trained weights; not runnable standalone.")
    codec = card.builder(  # type: ignore[misc]
        sample_rate=sample_rate, frame_size=frame_size, target_cr=target_cr, modality="ecg"
    )

    truth, faithful, crs = [], [], []
    for i in range(clean.shape[0]):
        enc = codec.encode(corrupted[i])
        recon = np.asarray(codec.decode(enc), dtype=np.float32).reshape(-1)[:frame_size]
        # truth_prd is scale-invariant (shape fidelity to the true clean morphology):
        # the synthetic clean is in raw units while the input is unit-normalized.
        truth.append(compute_signal_metrics(normalize_window(clean[i]), normalize_window(recon))["prd_percent"])
        faithful.append(compute_signal_metrics(corrupted[i], recon)["prd_percent"])
        crs.append((frame_size * 16) / enc.nbits if enc.nbits > 0 else np.inf)

    imprints = []
    for i in range(pure_artifact.shape[0]):
        enc = codec.encode(pure_artifact[i])
        recon = np.asarray(codec.decode(enc), dtype=np.float32).reshape(-1)[:frame_size]
        s_in = autocorr_structure(pure_artifact[i], sample_rate)
        s_out = autocorr_structure(recon, sample_rate)
        imprints.append(max(0.0, s_out - s_in))

    return BenchmarkResult(
        method_id=card.id,
        truth_prd=float(np.median(truth)),
        faithful_prd=float(np.median(faithful)),
        imprint=float(np.median(imprints)),
        true_cr=float(np.median(crs)),
        family=family,
        severity=severity,
    )


def run_benchmark(
    method_ids: list[str],
    *,
    n_windows: int = 64,
    frame_size: int = 512,
    sample_rate: int = 256,
    target_cr: float = 8.0,
    family: str = "motion",
    severity: float = 0.5,
    seed: int = 0,
) -> list[BenchmarkResult]:
    """Build the synthetic-truth battery once and score each method on it."""
    from compressionkit.playbook.catalog import get_method

    clean = make_clean_windows(n_windows, frame_size, sample_rate, seed=seed)
    corrupted = simulate_contact_artifact_batch(
        clean, family=family, severity=severity, sample_rate=sample_rate, seed=seed + 1
    )
    pure = simulate_contact_artifact_batch(
        clean, family="pure_artifact", severity=1.0, sample_rate=sample_rate, seed=seed + 2
    )

    results: list[BenchmarkResult] = []
    for method_id in method_ids:
        card = get_method(method_id)
        if not card.runnable:
            continue
        try:
            results.append(
                benchmark_method(
                    card,
                    clean,
                    corrupted,
                    pure,
                    sample_rate=sample_rate,
                    frame_size=frame_size,
                    target_cr=target_cr,
                    family=family,
                    severity=severity,
                )
            )
        except FileNotFoundError:
            continue
    return results


def condition_snr_db(clean: np.ndarray, corrupted: np.ndarray) -> float:
    """Median per-window SNR (dB) of a corrupted batch vs its clean source."""
    from compressionkit.synthetic.ecg_contact_artifacts import measured_artifact_fraction

    snrs = []
    for i in range(clean.shape[0]):
        f = float(measured_artifact_fraction(clean[i], corrupted[i]))
        f = min(max(f, 1e-4), 1.0 - 1e-4)
        snrs.append(10.0 * np.log10((1.0 - f) / f))
    return float(np.median(snrs))


def run_benchmark_grid(
    method_ids: list[str],
    *,
    families: list[str],
    severities: list[float],
    n_windows: int = 64,
    frame_size: int = 512,
    sample_rate: int = 256,
    target_cr: float = 8.0,
    seed: int = 0,
) -> tuple[dict[str, dict[tuple[str, float], BenchmarkResult]], dict[tuple[str, float], float]]:
    """Sweep the triple metric over ``families x severities`` for each method.

    Codecs are built once and reused across all conditions (so a learned model
    is loaded a single time). Returns ``(grid, snr_by_condition)`` where
    ``grid[method_id][(family, severity)]`` is a :class:`BenchmarkResult` and
    ``snr_by_condition[(family, severity)]`` is the median input SNR in dB.
    """
    from compressionkit.playbook.catalog import get_method

    clean = make_clean_windows(n_windows, frame_size, sample_rate, seed=seed)
    pure = simulate_contact_artifact_batch(
        clean, family="pure_artifact", severity=1.0, sample_rate=sample_rate, seed=seed + 2
    )

    cards = [get_method(m) for m in method_ids]
    cards = [c for c in cards if c.runnable]

    grid: dict[str, dict[tuple[str, float], BenchmarkResult]] = {c.id: {} for c in cards}
    snr_by_condition: dict[tuple[str, float], float] = {}

    for family in families:
        for severity in severities:
            corrupted = simulate_contact_artifact_batch(
                clean, family=family, severity=severity, sample_rate=sample_rate, seed=seed + 1
            )
            snr_by_condition[(family, severity)] = condition_snr_db(clean, corrupted)
            for card in cards:
                try:
                    grid[card.id][(family, severity)] = benchmark_method(
                        card,
                        clean,
                        corrupted,
                        pure,
                        sample_rate=sample_rate,
                        frame_size=frame_size,
                        target_cr=target_cr,
                        family=family,
                        severity=severity,
                    )
                except FileNotFoundError:
                    grid.pop(card.id, None)
                    break
    return grid, snr_by_condition


__all__ = [
    "BenchmarkResult",
    "autocorr_structure",
    "benchmark_method",
    "condition_snr_db",
    "make_clean_windows",
    "run_benchmark",
    "run_benchmark_grid",
]
