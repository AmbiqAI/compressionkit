"""Tests for INT8 RVQ encoder release gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from compressionkit.export.quantization import (
    RvqEncoderQuantizationReport,
    require_rvq_encoder_quantization_report,
)
from compressionkit.trainers.common import collect_disjoint_quantization_datasets


def _report(*, p90: float = 8.0, maximum: float = 14.0, saturation: float = 0.0) -> RvqEncoderQuantizationReport:
    """Build a representative parity result for threshold tests."""
    return RvqEncoderQuantizationReport(
        code_index_match_fraction_median=0.5,
        encoder_input_saturation_fraction_max=saturation,
        encoder_input_saturation_fraction_p90=0.0,
        frames_checked=128,
        latent_prd_percent_p90=3.0,
        reconstruction_prd_percent_p90=p90,
        reconstruction_prd_percent_max=maximum,
    )


def test_quantization_report_accepts_reconstruction_parity_without_index_equality() -> None:
    """Equivalent RVQ entries need not produce identical index streams."""
    assert _report().passed()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"saturation": 0.0101}, "saturation"),
    ],
)
def test_quantization_report_rejects_saturation(kwargs: dict[str, float], expected: str) -> None:
    """The publication gate rejects invalid INT8 input ranges."""
    assert not _report(**kwargs).passed(), expected


def test_quantization_report_keeps_worst_frame_as_diagnostic() -> None:
    """An isolated RVQ decision-boundary crossing does not block release."""
    assert _report(maximum=80.0).passed()


def test_quantization_report_keeps_p90_as_recommendation() -> None:
    """High P90 is reported to users rather than suppressing the release."""
    assert _report(p90=80.0).passed()


def test_publisher_guard_requires_passed_report_for_int8_rvq(tmp_path: Path) -> None:
    """Hub releases cannot bypass the parity gate with a hand-written manifest."""
    (tmp_path / "deploy_manifest.json").write_text(
        json.dumps(
            {
                "family": "rvq",
                "quantization": "INT8",
                "quantization_validation": {"report": "quantization_report.json", "passed": True},
            }
        )
    )
    (tmp_path / "quantization_report.json").write_text(
        json.dumps({"passed": True, "metrics": _report().__dict__})
    )

    require_rvq_encoder_quantization_report(tmp_path)

    (tmp_path / "quantization_report.json").write_text(json.dumps({"passed": False, "metrics": _report().__dict__}))
    with pytest.raises(ValueError, match="not passed"):
        require_rvq_encoder_quantization_report(tmp_path)


def test_quantization_partitions_are_disjoint_and_seeded() -> None:
    """Calibration ranges and parity measurements must not reuse a frame."""
    frames = np.arange(20, dtype=np.float32).reshape(20, 1, 1, 1)
    dataset = tf.data.Dataset.from_tensor_slices((frames, frames)).batch(4)

    calibration, validation = collect_disjoint_quantization_datasets(
        dataset,
        calibration_frames=8,
        validation_frames=6,
        sampling_pool_frames=20,
        seed=123,
    )

    assert calibration.shape[0] == 8
    assert validation.shape[0] == 6
    assert set(calibration[:, 0, 0, 0]).isdisjoint(validation[:, 0, 0, 0])
