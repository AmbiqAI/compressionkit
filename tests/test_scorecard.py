"""Unit tests for compressionkit.evaluation.scorecard — quality scorecard builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from compressionkit.evaluation.scorecard import build_quality_scorecard


@pytest.fixture()
def ppg_run_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic PPG run directory with sample CSVs."""
    rng = np.random.default_rng(123)
    fs = 64
    duration_s = 8.0
    n = int(fs * duration_s)
    t = np.arange(n) / fs

    for i in range(10):
        # Mild frequency variation per sample for diversity
        freq = 1.0 + 0.1 * i
        original = np.sin(2 * np.pi * freq * t).astype(np.float32)
        # Reconstruction = original + small noise (simulates codec error)
        noise = 0.05 * rng.standard_normal(n).astype(np.float32)
        reconstructed = original + noise

        csv_path = tmp_path / f"sample_{i:03d}.csv"
        csv_path.write_text(
            "original,reconstructed\n"
            + "\n".join(
                f"{o:.6f},{r:.6f}" for o, r in zip(original, reconstructed)
            )
        )

    # Minimal summary.json
    summary = {
        "compression": {"compression_ratio": 4.0, "downsample_factor": 4},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    return tmp_path


@pytest.fixture()
def ecg_run_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic ECG run directory."""
    rng = np.random.default_rng(456)
    fs = 256
    duration_s = 4.0
    n = int(fs * duration_s)
    t = np.arange(n) / fs

    for i in range(6):
        freq = 1.0 + 0.2 * i
        original = np.sin(2 * np.pi * freq * t).astype(np.float32)
        noise = 0.03 * rng.standard_normal(n).astype(np.float32)
        reconstructed = original + noise

        csv_path = tmp_path / f"sample_{i:03d}.csv"
        csv_path.write_text(
            "original,reconstructed\n"
            + "\n".join(
                f"{o:.6f},{r:.6f}" for o, r in zip(original, reconstructed)
            )
        )

    (tmp_path / "summary.json").write_text(json.dumps({}))
    return tmp_path


class TestBuildQualityScorecard:
    def test_ppg_scorecard_structure(self, ppg_run_dir: Path) -> None:
        """Scorecard should contain all expected top-level sections."""
        card = build_quality_scorecard(ppg_run_dir, modality="ppg", sample_rate=64)

        assert card["modality"] == "ppg"
        assert card["sample_rate"] == 64
        assert card["num_samples"] == 10
        assert "time_domain" in card
        assert "spectral" in card
        assert "physiology" in card
        assert "bitrate" in card
        assert "context" in card

    def test_ppg_time_domain_metrics(self, ppg_run_dir: Path) -> None:
        """Time-domain section should have aggregate stats for PRD, RMSE, etc."""
        card = build_quality_scorecard(ppg_run_dir, modality="ppg", sample_rate=64)
        td = card["time_domain"]

        assert td["prd_percent"]["n"] == 10
        assert td["prd_percent"]["mean"] > 0.0
        assert "std" in td["prd_percent"]
        assert "median" in td["prd_percent"]
        assert "p10" in td["prd_percent"]
        assert "p90" in td["prd_percent"]

        assert td["rmse"]["n"] == 10
        assert td["cosine_similarity"]["mean"] > 0.9

    def test_ppg_spectral_metrics(self, ppg_run_dir: Path) -> None:
        """Spectral section should have band error, wf-PRD, coherence."""
        card = build_quality_scorecard(ppg_run_dir, modality="ppg", sample_rate=64)
        sp = card["spectral"]

        assert sp["band_total_rel_error"]["n"] == 10
        assert sp["weighted_freq_prd_percent"]["n"] == 10
        assert sp["coherence"]["n"] == 10
        assert sp["coherence"]["mean"] > 0.3  # should be decent for signal + small noise

    def test_ppg_noise_context(self, ppg_run_dir: Path) -> None:
        """Context section should contain noise estimates."""
        card = build_quality_scorecard(ppg_run_dir, modality="ppg", sample_rate=64)
        ctx = card["context"]

        assert ctx["noise_rms"]["n"] == 10
        assert ctx["noise_power"]["n"] == 10

    def test_ppg_prdn_noise_present(self, ppg_run_dir: Path) -> None:
        """PRDN-noise should be computed when noise estimator is available."""
        card = build_quality_scorecard(ppg_run_dir, modality="ppg", sample_rate=64)
        td = card["time_domain"]
        assert td["prdn_noise_percent"]["n"] > 0

    def test_ecg_scorecard_structure(self, ecg_run_dir: Path) -> None:
        """ECG scorecard should have all sections including physiology."""
        card = build_quality_scorecard(ecg_run_dir, modality="ecg", sample_rate=256)

        assert card["modality"] == "ecg"
        assert card["num_samples"] == 6
        assert "time_domain" in card
        assert "spectral" in card
        assert "physiology" in card

    def test_bitrate_from_summary(self, ppg_run_dir: Path) -> None:
        """Bitrate section should pull compression_ratio from summary.json."""
        card = build_quality_scorecard(ppg_run_dir, modality="ppg", sample_rate=64)
        br = card["bitrate"]
        assert br["codec_compression_ratio"] == 4.0

    def test_rejected_flat_samples(self, tmp_path: Path) -> None:
        """Near-flat samples should be rejected by min_signal_std."""
        n = 512
        # One valid sample
        csv_valid = tmp_path / "sample_000.csv"
        orig = np.sin(np.linspace(0, 4 * np.pi, n)).astype(np.float32)
        recon = orig + 0.01 * np.random.default_rng(0).standard_normal(n).astype(np.float32)
        csv_valid.write_text(
            "original,reconstructed\n"
            + "\n".join(f"{o:.6f},{r:.6f}" for o, r in zip(orig, recon))
        )
        # One flat sample
        csv_flat = tmp_path / "sample_001.csv"
        flat = np.full(n, 0.0001, dtype=np.float32)
        csv_flat.write_text(
            "original,reconstructed\n"
            + "\n".join(f"{o:.6f},{r:.6f}" for o, r in zip(flat, flat))
        )
        (tmp_path / "summary.json").write_text("{}")

        card = build_quality_scorecard(
            tmp_path, modality="ppg", sample_rate=64, min_signal_std=0.01,
        )
        assert card["num_samples_loaded"] == 2
        assert card["num_samples"] == 1
        assert card["num_samples_rejected"] == 1

    def test_unknown_modality_raises(self, ppg_run_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unknown modality"):
            build_quality_scorecard(ppg_run_dir, modality="emg", sample_rate=64)

    def test_noise_estimator_options(self, ppg_run_dir: Path) -> None:
        """Different noise estimators should produce valid scorecards."""
        for estimator in ("bp", "hf"):
            card = build_quality_scorecard(
                ppg_run_dir, modality="ppg", sample_rate=64,
                noise_estimator=estimator,
            )
            assert card["noise_estimator"] == estimator
            assert card["time_domain"]["prd_percent"]["n"] == 10

    def test_by_noise_tertile_structure(self, ppg_run_dir: Path) -> None:
        """Scorecard should include time-domain + spectral noise-tertile breakdown."""
        card = build_quality_scorecard(ppg_run_dir, modality="ppg", sample_rate=64)
        bnt = card["by_noise_tertile"]

        assert "thresholds_bp_noise_rms" in bnt
        assert bnt["thresholds_bp_noise_rms"]["clean_max"] <= bnt["thresholds_bp_noise_rms"]["median_max"]
        assert set(bnt["buckets"].keys()) == {"clean", "median", "noisy"}

        for name in ("clean", "median", "noisy"):
            b = bnt["buckets"][name]
            assert b["n"] > 0
            assert "time_domain" in b
            assert "spectral" in b
            td = b["time_domain"]
            assert "prd_percent" in td
            assert "prdn_noise_percent" in td
            assert "cosine_similarity" in td
            sp = b["spectral"]
            assert "band_total_rel_error" in sp
            assert "weighted_freq_prd_percent" in sp
            assert "coherence" in sp

    def test_by_noise_tertile_requires_enough_samples(self, tmp_path: Path) -> None:
        """With fewer than 6 samples, by_noise_tertile should be empty."""
        n = 512
        for i in range(3):
            orig = np.sin(np.linspace(0, 4 * np.pi, n)).astype(np.float32)
            recon = orig + 0.01 * np.random.default_rng(i).standard_normal(n).astype(np.float32)
            csv_path = tmp_path / f"sample_{i:03d}.csv"
            csv_path.write_text(
                "original,reconstructed\n"
                + "\n".join(f"{o:.6f},{r:.6f}" for o, r in zip(orig, recon))
            )
        (tmp_path / "summary.json").write_text("{}")
        card = build_quality_scorecard(tmp_path, modality="ppg", sample_rate=64)
        assert card["by_noise_tertile"] == {}
