"""Tests for the quantized INT8 TFLite denoiser export + LiteRT runtime path (#47)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from compressionkit.dsp.wavelet import dwt_forward
from compressionkit.models.wavelet_denoiser import (
    DenoiserMode,
    as_coeff_denoiser,
    build_wavelet_denoiser_v2,
    build_wavelet_gain_denoiser,
    detect_denoiser_mode,
)

_FRAME_SIZE = 512
_WAVELET = "bior4.4"
_LEVELS = 6


def _packed_frames(n: int, seed: int = 0) -> np.ndarray:
    """A handful of zero-mean/unit-std packed-coefficient windows for calibration/eval."""
    rng = np.random.default_rng(seed)
    packed = []
    for _ in range(n):
        frame = rng.standard_normal(_FRAME_SIZE).astype(np.float32)
        frame = (frame - frame.mean()) / (frame.std() + 1e-6)
        coeffs = dwt_forward(frame, levels=_LEVELS, wavelet=_WAVELET)
        packed.append(np.concatenate([coeffs.approx, *coeffs.details]).astype(np.float32))
    return np.stack(packed)


def test_detect_denoiser_mode_gain_net() -> None:
    model = build_wavelet_gain_denoiser(frame_size=_FRAME_SIZE, in_ch=2)
    mode = detect_denoiser_mode(model, frame_size=_FRAME_SIZE, wavelet=_WAVELET, levels=_LEVELS)
    assert mode.expects_level is True
    assert mode.feature_kind == "level"
    assert mode.gain_mode is True


def test_detect_denoiser_mode_direct_net() -> None:
    model = build_wavelet_denoiser_v2(frame_size=_FRAME_SIZE, in_ch=2)
    mode = detect_denoiser_mode(model, frame_size=_FRAME_SIZE, wavelet=_WAVELET, levels=_LEVELS)
    assert mode.expects_level is True
    assert mode.feature_kind == "level"
    assert mode.gain_mode is False


def test_denoiser_mode_round_trips_through_dict() -> None:
    mode = DenoiserMode(start=5, length=160, expects_level=True, feature_kind="band_ratio", gain_mode=True)
    restored = DenoiserMode.from_dict(json.loads(json.dumps(mode.to_dict())))
    assert restored == mode


def test_export_denoiser_tflite_matches_keras_closely(tmp_path: Path) -> None:
    """INT8 TFLite output should closely track the float32 Keras reference on a gain-mode model."""
    from compressionkit.export.tflite import export_denoiser_tflite
    from compressionkit.pipeline.learned_stages import load_wavelet_gain_preprocessor_tflite

    model = build_wavelet_gain_denoiser(frame_size=_FRAME_SIZE, in_ch=2)
    mode = detect_denoiser_mode(model, frame_size=_FRAME_SIZE, wavelet=_WAVELET, levels=_LEVELS)

    packed = _packed_frames(16)
    rep_dataset = np.stack([mode.compute_feature(p)[0, 0] for p in packed])[:, None, :, :]
    tflite_path, header_path = export_denoiser_tflite(model, rep_dataset=rep_dataset, output_dir=tmp_path)
    assert tflite_path.exists()
    assert header_path.exists()

    keras_denoise = as_coeff_denoiser(model, frame_size=_FRAME_SIZE, wavelet=_WAVELET, levels=_LEVELS)
    tflite_pre = load_wavelet_gain_preprocessor_tflite(
        tflite_path,
        frame_size=_FRAME_SIZE,
        feature_kind=mode.feature_kind,
        gain_mode=mode.gain_mode,
        wavelet=_WAVELET,
        levels=_LEVELS,
    )

    prds = []
    for p in _packed_frames(8, seed=1):
        keras_out = keras_denoise(p)
        tflite_out = tflite_pre.coeff_denoiser(p)
        sse = float(np.sum((keras_out - tflite_out) ** 2))
        sig_pow = float(np.sum(keras_out**2))
        prds.append(100.0 * np.sqrt(sse / (sig_pow + 1e-8)))

    # A near-identity toy gain model quantizes cleanly; generous but not
    # meaningless bound (real trained gain-mode models measured ~0.001-0.01%).
    assert max(prds) < 5.0


def test_load_wavelet_gain_preprocessor_tflite_rejects_bad_feature_kind(tmp_path: Path) -> None:
    from compressionkit.pipeline.learned_stages import load_wavelet_gain_preprocessor_tflite

    fake_tflite = tmp_path / "denoiser.tflite"
    fake_tflite.write_bytes(b"not a real tflite file")

    with pytest.raises(ValueError, match="feature_kind"):
        load_wavelet_gain_preprocessor_tflite(fake_tflite, frame_size=_FRAME_SIZE, feature_kind="bogus", gain_mode=True)


def test_load_wavelet_gain_preprocessor_tflite_missing_file(tmp_path: Path) -> None:
    from compressionkit.pipeline.learned_stages import load_wavelet_gain_preprocessor_tflite

    with pytest.raises(FileNotFoundError):
        load_wavelet_gain_preprocessor_tflite(
            tmp_path / "does_not_exist.tflite", frame_size=_FRAME_SIZE, feature_kind="level", gain_mode=True
        )


def test_litert_quantize_dequantize_round_trips_int16() -> None:
    """INT16X8 support in the shared quantize/dequantize helpers (see #55).

    Direct-mode (non-gain) denoisers need finer-than-INT8 resolution to stay
    within the hybrid reference-vector PRD tolerance; this only works if the
    shared quantize()/dequantize() helpers actually apply the scale/zero-point
    formula for int16 tensors instead of silently truncating to the wrong
    dtype (the original bug this test guards against).
    """
    from compressionkit.runtime._litert import dequantize, quantize

    details = {
        "dtype": np.int16,
        "quantization_parameters": {"scales": np.array([0.001], dtype=np.float32), "zero_points": np.array([0])},
    }
    data = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    quantized = quantize(data, details)
    assert quantized.dtype == np.int16
    # A 0.001 scale should use much more of the int16 range than int8 would.
    assert np.max(np.abs(quantized)) > 127
    restored = dequantize(quantized, details)
    np.testing.assert_allclose(restored, data, atol=1e-3)


def test_export_denoiser_tflite_int16x8_tracks_keras_on_direct_mode_model(tmp_path: Path) -> None:
    """Direct-mode (non-gain) denoisers should quantize far more accurately under
    INT16X8 than INT8 (see #55): this is the fix for the ECG hybrid denoiser's
    tail-risk quantization error."""
    from compressionkit.export.tflite import export_denoiser_tflite
    from compressionkit.pipeline.learned_stages import load_wavelet_gain_preprocessor_tflite

    model = build_wavelet_denoiser_v2(frame_size=_FRAME_SIZE, in_ch=2)
    mode = detect_denoiser_mode(model, frame_size=_FRAME_SIZE, wavelet=_WAVELET, levels=_LEVELS)
    assert mode.gain_mode is False

    packed = _packed_frames(16)
    rep_dataset = np.stack([mode.compute_feature(p)[0, 0] for p in packed])[:, None, :, :]
    tflite_path, _ = export_denoiser_tflite(
        model, rep_dataset=rep_dataset, output_dir=tmp_path, quantization="INT16X8", io_type="int16"
    )

    keras_denoise = as_coeff_denoiser(model, frame_size=_FRAME_SIZE, wavelet=_WAVELET, levels=_LEVELS)
    tflite_pre = load_wavelet_gain_preprocessor_tflite(
        tflite_path,
        frame_size=_FRAME_SIZE,
        feature_kind=mode.feature_kind,
        gain_mode=mode.gain_mode,
        wavelet=_WAVELET,
        levels=_LEVELS,
    )

    prds = []
    for p in _packed_frames(8, seed=1):
        keras_out = keras_denoise(p)
        tflite_out = tflite_pre.coeff_denoiser(p)
        sse = float(np.sum((keras_out - tflite_out) ** 2))
        sig_pow = float(np.sum(keras_out**2))
        prds.append(100.0 * np.sqrt(sse / (sig_pow + 1e-8)))

    assert max(prds) < 5.0


def _write_hybrid_deploy_fixture(tmp_path: Path, *, denoise_stage: dict) -> Path:
    """A minimal hybrid deploy dir sufficient for generate_spiht_model_card()."""
    manifest = {
        "family": "hybrid",
        "method": "hybrid",
        "model_name": "ecg_hybrid_test",
        "codec": {"modality": "ecg", "sample_rate": 256, "frame_size": 512, "target_cr": 8.0},
    }
    (tmp_path / "deploy_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "codec_spec.json").write_text(json.dumps({"codec": manifest["codec"]}))
    (tmp_path / "hybrid_manifest.json").write_text(
        json.dumps({"pipeline": "hybrid", "stages": [denoise_stage, {"stage": "codec", "type": "spiht"}]})
    )
    return tmp_path


def test_model_card_reports_int16x8_for_direct_mode_denoiser(tmp_path: Path) -> None:
    from compressionkit.export.model_card import generate_spiht_model_card

    _write_hybrid_deploy_fixture(
        tmp_path,
        denoise_stage={
            "stage": "denoise",
            "artifact": "denoiser_gain_model.keras",
            "artifact_tflite": "denoiser_gain_model.tflite",
            "mode": {"start": 8, "length": 256, "expects_level": True, "feature_kind": "level", "gain_mode": False},
        },
    )

    card = generate_spiht_model_card(tmp_path)
    assert "INT16X8 `denoiser_gain_model.tflite`" in card


def test_model_card_reports_int8_for_gain_mode_denoiser(tmp_path: Path) -> None:
    from compressionkit.export.model_card import generate_spiht_model_card

    _write_hybrid_deploy_fixture(
        tmp_path,
        denoise_stage={
            "stage": "denoise",
            "artifact": "denoiser_gain_model.keras",
            "artifact_tflite": "denoiser_gain_model.tflite",
            "mode": {"start": 5, "length": 160, "expects_level": True, "feature_kind": "level", "gain_mode": True},
        },
    )

    card = generate_spiht_model_card(tmp_path)
    assert "INT8 `denoiser_gain_model.tflite`" in card


def test_model_card_falls_back_to_keras_only_text_when_tflite_artifact_missing(tmp_path: Path) -> None:
    """Older packages / failed exports must not claim a .tflite denoiser that doesn't exist (#58 review)."""
    from compressionkit.export.model_card import generate_spiht_model_card

    _write_hybrid_deploy_fixture(
        tmp_path,
        denoise_stage={"stage": "denoise", "artifact": "denoiser_gain_model.keras"},
    )

    card = generate_spiht_model_card(tmp_path)
    assert "denoiser_gain_model.tflite" not in card
    assert "Python/TFLite runtime path" in card
