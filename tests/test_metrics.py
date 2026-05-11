"""Metric correctness tests — numpy-only, no Keras model required."""

from __future__ import annotations

import numpy as np

from compressionkit.evaluation import metrics
from compressionkit.evaluation.metrics import compute_signal_metrics


def test_compute_signal_metrics_identical_signals() -> None:
    """An identical reconstruction should give zero error and unit cosine."""
    signal = np.sin(np.linspace(0, 4 * np.pi, 512)).astype(np.float32)

    out = compute_signal_metrics(signal, signal)

    assert out["mse"] == 0.0
    assert out["mae"] == 0.0
    assert np.isclose(out["cosine_similarity"], 1.0, atol=1e-6)
    assert np.isclose(out["prd_percent"], 0.0, atol=1e-6)


def test_compute_signal_metrics_scaled_reconstruction() -> None:
    """Scaling preserves cosine similarity and yields non-zero MSE."""
    signal = np.sin(np.linspace(0, 4 * np.pi, 512)).astype(np.float32)
    recon = 0.5 * signal

    out = compute_signal_metrics(signal, recon)

    assert out["mse"] > 0.0
    assert out["mae"] > 0.0
    assert np.isclose(out["cosine_similarity"], 1.0, atol=1e-5)
    assert out["prd_percent"] > 0.0


def test_summarize_ppg_peak_alignment_matches_unique_peaks(monkeypatch) -> None:
    """PPG peak matching should report timing, precision/recall, and IBI error."""
    peak_sets = {
        1: [10, 50, 90, 130, 170],
        2: [11, 51, 91, 131, 210],
    }

    def fake_ppg_metrics(signal, **_kwargs):
        peaks = peak_sets[int(signal[0])]
        return {
            "hr_bpm": 60.0,
            "hr_qos": 1.0,
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "mean_nn_ms": 1000.0,
            "num_peaks": len(peaks),
            "peak_locations": peaks,
        }

    monkeypatch.setattr(metrics, "compute_ppg_physiokit_metrics", fake_ppg_metrics)

    originals = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    recons = np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32)
    summary, per_sample = metrics.summarize_ppg_peak_alignment(
        originals,
        recons,
        sample_rate=64,
        min_peaks=5,
        timing_tolerance_ms=125.0,
    )

    assert summary is not None
    assert per_sample[0] is not None
    assert summary["matched_total_peaks"] == 4.0
    assert summary["total_missed_peaks"] == 1.0
    assert summary["total_extra_peaks"] == 1.0
    assert np.isclose(summary["peak_precision_pct"], 80.0)
    assert np.isclose(summary["peak_recall_pct"], 80.0)
    assert np.isclose(summary["peak_timing_mae_ms"], 1000.0 / 64.0)
    assert np.isclose(summary["ibi_mae_ms"], 0.0)
