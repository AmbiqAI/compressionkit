"""Tests for 12-lead Holter reporting and LiteRT verification (issue #10)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

# Load report_12lead_holter.py
_REPORT_SPEC = importlib.util.spec_from_file_location(
    "report_12lead_holter",
    Path(__file__).resolve().parent.parent / "scripts" / "report_12lead_holter.py",
)
assert _REPORT_SPEC is not None and _REPORT_SPEC.loader is not None
report_mod = importlib.util.module_from_spec(_REPORT_SPEC)
sys.modules["report_12lead_holter"] = report_mod
_REPORT_SPEC.loader.exec_module(report_mod)

# Load verify_xlead_litert.py
_VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_xlead_litert",
    Path(__file__).resolve().parent.parent / "scripts" / "verify_xlead_litert.py",
)
assert _VERIFY_SPEC is not None and _VERIFY_SPEC.loader is not None
verify_mod = importlib.util.module_from_spec(_VERIFY_SPEC)
sys.modules["verify_xlead_litert"] = verify_mod
_VERIFY_SPEC.loader.exec_module(verify_mod)


def _write_entropy_report(
    run_dir: Path, tag: str, *, bpt: float, codec_cr: float, vocab: int = 256, tpf: int = 4
) -> None:
    p = run_dir / "entropy_prior" / tag
    p.mkdir(parents=True, exist_ok=True)
    (p / "entropy_report.json").write_text(
        json.dumps(
            {
                "prior": {"type": "cnn"},
                "codec": {"cr_codec_uniform": codec_cr, "tokens_per_frame": tpf},
                "metrics": {"val_bits_per_token": bpt},
            }
        )
    )


def _write_xlead_report(
    run_dir: Path, tag: str, *, bpt: float, codec_cr: float, prior_type: str, vocab: int = 256, tpf: int = 4
) -> None:
    p = run_dir / "entropy_prior" / tag
    p.mkdir(parents=True, exist_ok=True)
    (p / "entropy_report.json").write_text(
        json.dumps(
            {
                "prior": {"type": prior_type},
                "codec": {"cr_codec_uniform": codec_cr, "tokens_per_frame": tpf},
                "metrics": {"val_bits_per_token": bpt},
            }
        )
    )


class TestPerLeadStats:
    def test_basic(self, tmp_path: Path):
        _write_entropy_report(tmp_path, "cnn_xl", bpt=4.0, codec_cr=32.0, tpf=4)
        stats = report_mod.compute_per_lead_stats(tmp_path, "cnn_xl", num_leads=12, vocab_size=256)
        assert stats["bpt_per_lead"] == 4.0
        assert stats["tokens_per_frame"] == 4
        assert stats["total_bits_per_frame_12lead"] == 4.0 * 4 * 12
        assert stats["cr_codec_uniform"] == 32.0
        assert stats["cr_uplift_vs_uniform"] == 8.0 / 4.0  # log2(256)/4.0 = 2.0
        assert stats["effective_cr_12lead"] == 32.0 * 2.0

    def test_missing_report(self, tmp_path: Path):
        stats = report_mod.compute_per_lead_stats(tmp_path, "missing", num_leads=12)
        assert "error" in stats


class TestXleadStats:
    def test_basic(self, tmp_path: Path):
        _write_xlead_report(tmp_path, "xl_concat", bpt=3.5, codec_cr=32.0, prior_type="xlead_concat", tpf=4)
        stats = report_mod.compute_xlead_stats(tmp_path, "xl_concat", num_leads=12, vocab_size=256)
        assert stats["bpt_cross_channel"] == 3.5
        assert stats["total_tokens_per_frame_12lead"] == 48
        assert stats["total_bits_per_frame_12lead"] == 3.5 * 48


class TestComparison:
    def test_bpt_reduction(self, tmp_path: Path):
        _write_entropy_report(tmp_path, "cnn_xl", bpt=4.0, codec_cr=32.0, tpf=4)
        _write_xlead_report(tmp_path, "xl_concat", bpt=3.0, codec_cr=32.0, prior_type="xlead_concat", tpf=4)

        ind = report_mod.compute_per_lead_stats(tmp_path, "cnn_xl", num_leads=12, vocab_size=256)
        xl = report_mod.compute_xlead_stats(tmp_path, "xl_concat", num_leads=12, vocab_size=256)
        comp = report_mod.build_comparison(ind, xl)

        assert comp["comparison"] is not None
        # Independent: 4.0 * 4 * 12 = 192; xlead: 3.0 * 48 = 144
        # Reduction: (1 - 144/192) * 100 = 25%
        assert abs(comp["comparison"]["bpt_reduction_percent"] - 25.0) < 0.01
        assert comp["comparison"]["meets_10pct_threshold"] is True

    def test_no_reduction(self, tmp_path: Path):
        _write_entropy_report(tmp_path, "cnn_xl", bpt=4.0, codec_cr=32.0, tpf=4)
        _write_xlead_report(tmp_path, "xl_concat", bpt=3.8, codec_cr=32.0, prior_type="xlead_concat", tpf=4)

        ind = report_mod.compute_per_lead_stats(tmp_path, "cnn_xl", num_leads=12, vocab_size=256)
        xl = report_mod.compute_xlead_stats(tmp_path, "xl_concat", num_leads=12, vocab_size=256)
        comp = report_mod.build_comparison(ind, xl)

        # (1 - 3.8*48 / 4*48) * 100 = 5% — below threshold
        assert comp["comparison"]["meets_10pct_threshold"] is False


class TestMarkdownRender:
    def test_output_has_table(self, tmp_path: Path):
        _write_entropy_report(tmp_path, "cnn_xl", bpt=4.0, codec_cr=32.0, tpf=4)
        _write_xlead_report(tmp_path, "xl_concat", bpt=3.0, codec_cr=32.0, prior_type="xlead_concat", tpf=4)

        ind = report_mod.compute_per_lead_stats(tmp_path, "cnn_xl", num_leads=12, vocab_size=256)
        xl = report_mod.compute_xlead_stats(tmp_path, "xl_concat", num_leads=12, vocab_size=256)
        report = report_mod.build_comparison(ind, xl)
        md = report_mod.render_markdown(report)

        assert "12-Lead Holter" in md
        assert "Per-lead Independent" in md
        assert "Cross-channel" in md
        assert "25.0%" in md


class TestMainCLI:
    def test_writes_output(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_entropy_report(run_dir, "cnn_xl", bpt=4.0, codec_cr=32.0, tpf=4)
        out = tmp_path / "report.json"
        rc = report_mod.main(["--run-dir", str(run_dir), "--independent-tag", "cnn_xl", "--output", str(out)])
        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "independent" in data


class TestLiteRTVerify:
    def test_verify_prior_export_builds_model(self):
        """Test that the verify function at least builds models correctly."""
        from compressionkit.generative.xlead_prior import build_xlead_concat_prior

        model = build_xlead_concat_prior(
            vocab_size=32, context_length=4, num_leads=2, embed_dim=8, num_layers=2, kernel_size=3
        )
        assert model.count_params() > 0
        # Just validate the model builds — actual TFLite export depends on helia_edge
        out = model(np.zeros((1, 4, 2), dtype=np.int32), training=False)
        assert out.shape == (1, 4, 2, 32)
