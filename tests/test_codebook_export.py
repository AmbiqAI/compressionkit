"""Codebook exports must preserve the trained quantizer, including EMA state."""

from __future__ import annotations

import re

import keras
import numpy as np
import pytest

from compressionkit.export.codebook import (
    export_codebooks_header,
    export_codebooks_npz,
    extract_codebooks,
    load_rvq_weights,
)
from compressionkit.layers.ema_residual_vector_quantizer import EmaResidualVectorQuantizer
from compressionkit.layers.residual_vector_quantizer import ResidualVectorQuantizer
from compressionkit.runtime.codec import RVQCodec


@pytest.mark.parametrize("use_ema,kmeans_init", [(False, False), (True, False), (True, True)])
def test_export_preserves_trained_quantizer(tmp_path, use_ema, kmeans_init):
    rng = np.random.default_rng(74)
    kwargs = {"num_levels": 4, "num_embeddings": 8, "embedding_dim": 3}
    layer = (
        EmaResidualVectorQuantizer(**kwargs, kmeans_init=kmeans_init) if use_ema else ResidualVectorQuantizer(**kwargs)
    )
    layer.build((None, 1, 16, 3))
    for level, cb in enumerate(layer._codebooks):
        cb.assign(rng.normal(size=(8, 3)).astype(np.float32) / (level + 1))
    if use_ema:
        for count, total in zip(layer._ema_counts, layer._ema_weights):
            count.assign(np.full(8, 13.0, dtype=np.float32))
            total.assign(rng.normal(100, 5, size=(8, 3)).astype(np.float32))
    weights = layer.get_weights()
    archive = tmp_path / "rvq_weights.npz"
    np.savez(archive, *weights)  # Includes arr_10 and arr_11 for EMA.
    codebooks = extract_codebooks(load_rvq_weights(archive), num_levels=4, use_ema=use_ema, kmeans_init=kmeans_init)
    npz = export_codebooks_npz(codebooks, tmp_path / "codebook.npz")
    header = export_codebooks_header(codebooks, tmp_path / "codebook.h")
    with np.load(npz) as saved:
        assert saved.files == [f"level_{i}" for i in range(4)]
        for i, cb in enumerate(layer._codebooks):
            np.testing.assert_array_equal(saved[f"level_{i}"], keras.ops.convert_to_numpy(cb))

    # C literals must preserve the same float32 values as the NPZ tables.
    literals = re.findall(r"(?<![\w.])(-?\d+(?:\.\d*)?(?:e[+-]?\d+)?)f\b", header.read_text())
    np.testing.assert_array_equal(np.array(literals, dtype=np.float32), np.concatenate(codebooks).ravel())
    codec = object.__new__(RVQCodec)
    codec._codebooks = codebooks
    codec._num_levels = 4
    codec._embedding_dim = 3
    latents = rng.normal(size=(2, 1, 16, 3)).astype(np.float32)
    trained_indices = layer.encode(latents)
    expected = np.stack([keras.ops.convert_to_numpy(i) for i in trained_indices], axis=-1).reshape(2, 1, 16, 4)
    actual = codec.quantize_latent(latents)
    np.testing.assert_array_equal(actual, expected)
    trained_latents = keras.ops.convert_to_numpy(layer.decode(trained_indices, latents.shape))
    np.testing.assert_array_equal(codec.dequantize_indices(actual), trained_latents)


def test_ema_weights_cannot_be_exported_as_plain_codebooks():
    weights = [np.ones((8, 3), np.float32), np.ones(8, np.float32), np.ones((8, 3), np.float32)]
    with pytest.raises(ValueError, match="codebook matrix"):
        extract_codebooks(weights)
    with pytest.raises(ValueError, match="requires num_levels"):
        extract_codebooks(weights, use_ema=True)


@pytest.mark.parametrize("case", ["missing", "counts", "sums", "flag", "dimension", "nonfinite"])
def test_reject_invalid_checkpoint_layout(case):
    weights = [np.ones((8, 3), np.float32), np.ones(8, np.float32), np.ones((8, 3), np.float32)] * 2
    weights = [w.copy() for w in weights] + [np.array(1.0, np.float32)]
    if case == "missing":
        weights.pop(2)
    elif case == "counts":
        weights[1] = np.ones(7, np.float32)
    elif case == "sums":
        weights[2] = np.ones((8, 2), np.float32)
    elif case == "flag":
        weights[-1] = np.ones(1, np.float32)
    elif case == "dimension":
        weights[3] = np.ones((8, 4), np.float32)
    else:
        weights[0][0, 0] = np.nan
    with pytest.raises(ValueError):
        extract_codebooks(weights, num_levels=2, use_ema=True, kmeans_init=True)


def test_reject_noncontiguous_weight_archive(tmp_path):
    path = tmp_path / "weights.npz"
    np.savez(path, arr_0=np.ones((4, 2)), arr_2=np.ones((4, 2)))
    with pytest.raises(ValueError, match="contiguous"):
        load_rvq_weights(path)


def test_runtime_loads_numeric_levels_and_rejects_contract_mismatch(tmp_path, monkeypatch):
    import json
    from unittest.mock import MagicMock

    from compressionkit.runtime import codec as codec_module

    monkeypatch.setattr(codec_module, "_Interpreter", MagicMock())
    books = {f"level_{i}": np.full((4, 2), i / 16, np.float32) for i in range(12)}
    np.savez(tmp_path / "codebook.npz", **books)
    manifest = {
        "encoder": {"tflite": "encoder.tflite"},
        "codebook": {"npz": "codebook.npz", "num_levels": 12, "num_embeddings": 4, "embedding_dim": 2},
    }
    (tmp_path / "deploy_manifest.json").write_text(json.dumps(manifest))
    codec = RVQCodec(tmp_path)
    for i, cb in enumerate(codec._codebooks):
        np.testing.assert_array_equal(cb, books[f"level_{i}"])
    manifest["codebook"]["num_levels"] = 11
    (tmp_path / "deploy_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="levels do not match"):
        RVQCodec(tmp_path)


def test_c_header_compiles_and_roundtrips_float32(tmp_path):
    import shutil
    import subprocess

    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable")
    values = np.array([[0.0, 1.0, -1.0, 0.123456789, np.finfo(np.float32).tiny, np.finfo(np.float32).max]], np.float32)
    export_codebooks_header([values], tmp_path / "codebook.h")
    source = tmp_path / "check.c"
    source.write_text(
        '#include <stdio.h>\n#include "codebook.h"\nint main(void) {\n'
        "return fwrite(rvq_codebook_l0, sizeof(float), 6, stdout) == 6 ? 0 : 1;\n}\n"
    )
    executable = tmp_path / "check"
    subprocess.run(
        [cc, "-std=c99", "-Wall", "-Werror", str(source), "-o", str(executable)], check=True, capture_output=True
    )
    result = subprocess.run([str(executable)], check=True, capture_output=True)
    np.testing.assert_array_equal(np.frombuffer(result.stdout, dtype=np.float32), values.ravel())
