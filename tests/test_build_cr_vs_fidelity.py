"""Tests for the B4 CR-vs-fidelity decision artefact builder (issue #2)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "build_cr_vs_fidelity",
    Path(__file__).resolve().parent.parent / "scripts" / "build_cr_vs_fidelity.py",
)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules["build_cr_vs_fidelity"] = mod
SPEC.loader.exec_module(mod)


def _scorecard(
    *,
    codec_cr: float,
    effective_cr: float,
    prd: float,
    prdn: float,
    hr_mae: float,
    qrs_err: float,
    coh: float,
    seam: float | None = None,
) -> dict:
    sc = {
        "num_samples": 1000,
        "bitrate": {
            "codec_compression_ratio": codec_cr,
            "cr_codec_uniform": codec_cr,
            "cr_codec_learned": effective_cr,
            "val_bits_per_token": 4.0,
        },
        "time_domain": {
            "prd_percent": {"mean": prd},
            "prdn_noise_percent": {"mean": prdn},
        },
        "spectral": {
            "coherence": {"mean": coh},
            "per_band_rel_error": {
                "band_5_15_rel_error": {"mean": qrs_err},
                "band_0.5_3_rel_error": {"mean": qrs_err},
            },
        },
        "physiology": {
            "vs_raw_original": {"hr_mae_bpm": hr_mae},
            "by_noise_tertile": {
                "buckets": {
                    t: {"hr_mae_bpm": {"mean": hr_mae * (0.5 + i * 0.5)}}
                    for i, t in enumerate(("clean", "median", "noisy"))
                }
            },
        },
        "by_noise_tertile": {
            "buckets": {
                t: {
                    "n": 333,
                    "time_domain": {
                        "prd_percent": {"mean": prd * (0.7 + i * 0.3)},
                        "prdn_noise_percent": {"mean": prdn * (0.5 + i * 0.5)},
                    },
                    "spectral": {
                        "coherence": {"mean": coh},
                        "per_band_rel_error": {
                            "band_5_15_rel_error": {"mean": qrs_err},
                            "band_0.5_3_rel_error": {"mean": qrs_err},
                        },
                    },
                }
                for i, t in enumerate(("clean", "median", "noisy"))
            }
        },
    }
    if seam is not None:
        sc["long_recording"] = {"seam_ratio": seam}
    return sc


def test_headline_row_pulls_codec_and_effective_cr():
    sc = _scorecard(codec_cr=8.0, effective_cr=15.2, prd=5.0, prdn=0.1, hr_mae=0.8, qrs_err=0.02, coh=0.998, seam=1.01)
    row = mod.build_headline_row(sc, "ecg_rvq_256hz_08x_golden", "ecg")
    assert row["cr_label"] == "08x"
    assert row["codec_cr"] == 8.0
    assert row["effective_cr"] == 15.2
    assert row["prd_percent"] == 5.0
    assert row["hr_mae_bpm"] == 0.8
    assert row["qrs_band_err"] == 0.02
    assert row["stitching_seam_ratio"] == 1.01


def test_tertile_rows_emit_three_per_run():
    sc = _scorecard(codec_cr=8.0, effective_cr=15.2, prd=5.0, prdn=0.1, hr_mae=0.8, qrs_err=0.02, coh=0.998)
    rows = mod.build_tertile_rows(sc, "ecg_rvq_256hz_08x_golden", "ecg")
    assert [r["tertile"] for r in rows] == ["clean", "median", "noisy"]
    assert all(r["cr_label"] == "08x" for r in rows)
    assert rows[0]["prd_percent"] < rows[2]["prd_percent"]


def test_document_renders_both_sections_and_modality_band_label():
    sc_ecg = _scorecard(codec_cr=8.0, effective_cr=15.0, prd=5.0, prdn=0.1, hr_mae=0.8, qrs_err=0.02, coh=0.998)
    sc_ppg = _scorecard(codec_cr=4.0, effective_cr=6.5, prd=3.0, prdn=0.05, hr_mae=0.4, qrs_err=0.01, coh=0.999)

    doc_ecg = mod.build_document([("ecg_rvq_256hz_08x_golden", sc_ecg)], "ecg")
    assert "# ECG CR vs. Fidelity" in doc_ecg
    assert "Headline summary" in doc_ecg
    assert "Noise-stratified detail" in doc_ecg
    assert "QRS-band PSD err" in doc_ecg
    assert "15.00" in doc_ecg  # effective CR formatted

    doc_ppg = mod.build_document([("ppg_rvq_64hz_04x_golden", sc_ppg)], "ppg")
    assert "Pulse-band PSD err" in doc_ppg
    assert "QRS-band" not in doc_ppg


def test_tertile_filter_subset():
    sc_a = _scorecard(codec_cr=8.0, effective_cr=15.0, prd=5.0, prdn=0.1, hr_mae=0.8, qrs_err=0.02, coh=0.998)
    sc_b = _scorecard(codec_cr=32.0, effective_cr=60.0, prd=15.0, prdn=1.0, hr_mae=2.3, qrs_err=0.08, coh=0.99)
    doc = mod.build_document(
        [("ecg_rvq_256hz_08x_golden", sc_a), ("ecg_rvq_256hz_32x_golden", sc_b)],
        "ecg",
        tertile_crs=["08x"],
    )
    # Headline still includes both runs; tertile section is filtered to 08x only.
    headline_idx = doc.index("Headline summary")
    tertile_idx = doc.index("Noise-stratified detail")
    tertile_body = doc[tertile_idx:]
    assert "32x" in doc[headline_idx:tertile_idx]
    assert "08x" in tertile_body
    assert "32x" not in tertile_body


def test_main_writes_markdown(tmp_path: Path):
    results = tmp_path / "results"
    run = results / "ecg_rvq_256hz_08x_golden"
    run.mkdir(parents=True)
    sc = _scorecard(codec_cr=8.0, effective_cr=15.0, prd=5.0, prdn=0.1, hr_mae=0.8, qrs_err=0.02, coh=0.998)
    import json as _json

    (run / "quality_scorecard.json").write_text(_json.dumps(sc))

    out = tmp_path / "out.md"
    rc = mod.main(
        [
            "--modality",
            "ecg",
            "--results-dir",
            str(results),
            "--runs",
            "ecg_rvq_256hz_08x_golden",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    text = out.read_text()
    assert "# ECG CR vs. Fidelity" in text
    assert "08x" in text
