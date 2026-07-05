"""Tests for compressionkit.generative.spiht_bit_prior.

Covers three layers:
  1. Numerical correctness of the pure-numpy WaveNet forward pass against the
     reference Keras model (`build_wavenet_bit_prior`).
  2. The incremental `SpihtBitPredictor`'s ring-buffer bookkeeping exactly
     reproducing the batched/windowed computation, both in the "growing"
     regime (frame shorter than context_length) and the "sliding" regime
     (frame longer than context_length).
  3. `NeuralAcSink`/`NeuralAcSource` as a standalone entropy coder (isolated
     bit-exact roundtrip, independent of SPIHT), the probability-clipping
     safety margin (mirrors the RVQ rANS zero-width-interval bug fix), and
     full integration through `spiht_encode`/`spiht_decode`'s generic
     `sink=`/`source=` injection point.
"""

from __future__ import annotations

import numpy as np
import pytest

from compressionkit.dsp.spiht import spiht_decode, spiht_encode
from compressionkit.dsp.wavelet import WaveletCoeffs, dwt_forward, dwt_inverse
from compressionkit.generative.causal_priors import build_wavenet_bit_prior
from compressionkit.generative.spiht_bit_prior import (
    PREVBIT_START,
    NeuralAcSink,
    NeuralAcSource,
    SpihtBitPredictor,
    extract_wavenet_bit_prior_weights,
    wavenet_forward_numpy,
)

# SPIHT has 6 fixed symbol classes (CTX_LIP_SIG, CTX_LIS_A_SIG, CTX_LIS_B_SIG,
# CTX_CHILD_SIG, CTX_SIGN, CTX_REFINE — see compressionkit/dsp/spiht.py).
_N_CTX = 6
_BP_VOCAB = 16
_PREVBIT_VOCAB = 3


def _build_test_model(context_length: int, *, embed_dim: int = 8, num_layers: int = 3, kernel_size: int = 3):
    return build_wavenet_bit_prior(
        context_length,
        ctx_vocab=_N_CTX,
        bp_vocab=_BP_VOCAB,
        prevbit_vocab=_PREVBIT_VOCAB,
        embed_dim=embed_dim,
        num_layers=num_layers,
        kernel_size=kernel_size,
    )


# ---------------------------------------------------------------------------
# 1. Numpy forward pass vs Keras
# ---------------------------------------------------------------------------


def test_numpy_forward_matches_keras_batched():
    rng = np.random.default_rng(0)
    cl = 20
    model = _build_test_model(cl)
    weights = extract_wavenet_bit_prior_weights(model, context_length=cl, kernel_size=3, bp_vocab=_BP_VOCAB)

    ctx = rng.integers(0, _N_CTX, size=cl)
    bp = rng.integers(0, _BP_VOCAB, size=cl)
    prev = rng.integers(0, _PREVBIT_VOCAB, size=cl)

    keras_logits = model.predict([ctx[None, :], bp[None, :], prev[None, :]], verbose=0)[0]
    keras_p1 = 1.0 / (1.0 + np.exp(-keras_logits))
    numpy_p1 = wavenet_forward_numpy(ctx, bp, prev, weights)

    assert np.max(np.abs(keras_p1 - numpy_p1)) < 1e-5


# ---------------------------------------------------------------------------
# 2. Incremental predictor vs batched (growing + sliding regimes)
# ---------------------------------------------------------------------------


def test_incremental_predictor_matches_batched_within_context_length():
    """Frame shorter than context_length: window only ever grows."""
    rng = np.random.default_rng(1)
    cl = 20
    model = _build_test_model(cl)
    weights = extract_wavenet_bit_prior_weights(model, context_length=cl, kernel_size=3, bp_vocab=_BP_VOCAB)

    ctx = rng.integers(0, _N_CTX, size=cl)
    bp = rng.integers(0, _BP_VOCAB, size=cl)
    true_bits = rng.integers(0, 2, size=cl)

    prev = np.empty(cl, dtype=np.int32)
    prev[0] = PREVBIT_START
    prev[1:] = true_bits[:-1]
    batched_p1 = wavenet_forward_numpy(ctx, bp, prev, weights)

    predictor = SpihtBitPredictor(weights)
    incremental_p1 = []
    for t in range(cl):
        incremental_p1.append(predictor.predict(int(ctx[t]), int(bp[t])))
        predictor.observe(int(ctx[t]), int(bp[t]), int(true_bits[t]))

    assert np.allclose(incremental_p1, batched_p1, atol=1e-9)


def test_incremental_predictor_matches_windowed_reference_when_sliding():
    """Frame longer than context_length: window slides (drops oldest position)."""
    rng = np.random.default_rng(2)
    cl = 16
    n = 50
    model = _build_test_model(cl)
    weights = extract_wavenet_bit_prior_weights(model, context_length=cl, kernel_size=3, bp_vocab=_BP_VOCAB)

    ctx = rng.integers(0, _N_CTX, size=n)
    bp = rng.integers(0, _BP_VOCAB, size=n)
    true_bits = rng.integers(0, 2, size=n)

    reference_p1 = []
    for t in range(n):
        start = max(0, t - cl + 1)
        ctx_win = ctx[start : t + 1]
        bp_win = bp[start : t + 1]
        prev_win = np.empty(t - start + 1, dtype=np.int32)
        for i, pos in enumerate(range(start, t + 1)):
            prev_win[i] = PREVBIT_START if pos == 0 else true_bits[pos - 1]
        reference_p1.append(wavenet_forward_numpy(ctx_win, bp_win, prev_win, weights)[-1])
    reference_p1 = np.array(reference_p1)

    predictor = SpihtBitPredictor(weights)
    incremental_p1 = []
    for t in range(n):
        incremental_p1.append(predictor.predict(int(ctx[t]), int(bp[t])))
        predictor.observe(int(ctx[t]), int(bp[t]), int(true_bits[t]))

    assert np.allclose(incremental_p1, reference_p1, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. NeuralAcSink/NeuralAcSource — standalone entropy coder correctness
# ---------------------------------------------------------------------------


def test_neural_ac_bit_exact_roundtrip_isolated_from_spiht():
    """Encode/decode a known (ctx, bitplane, bit) sequence directly via the
    sink/source pair, with NO SPIHT tree-traversal involved. This isolates
    correctness of the entropy-coding primitive itself."""
    rng = np.random.default_rng(3)
    cl = 24
    model = _build_test_model(cl)
    weights = extract_wavenet_bit_prior_weights(model, context_length=cl, kernel_size=3, bp_vocab=_BP_VOCAB)

    n_symbols = 500
    ctx_seq = rng.integers(0, _N_CTX, size=n_symbols)
    bp_seq = rng.integers(0, _BP_VOCAB, size=n_symbols)
    bit_seq = rng.integers(0, 2, size=n_symbols)

    enc_predictor = SpihtBitPredictor(weights)
    sink = NeuralAcSink(capacity_bits=n_symbols * 8, predictor=enc_predictor)  # generous budget
    for ctx, bp, bit in zip(ctx_seq, bp_seq, bit_seq):
        sink.set_bitplane(int(bp))
        sink.write(int(bit), int(ctx))
    bitstream = sink.to_bytes()

    dec_predictor = SpihtBitPredictor(weights)
    source = NeuralAcSource(data=bitstream, total_bits=len(bitstream) * 8, predictor=dec_predictor)
    decoded_bits = []
    for ctx, bp in zip(ctx_seq, bp_seq):
        source.set_bitplane(int(bp))
        decoded_bits.append(source.read(int(ctx)))

    assert decoded_bits == list(bit_seq)


def test_neural_ac_handles_extreme_probabilities_without_hanging():
    """A predictor that always returns p1 near 0 or 1 must not create a
    zero-width coding interval (the exact class of bug fixed for the RVQ
    rANS/AC coder — see repo memory entropy-prior-sweep.md)."""

    class _ExtremePredictor:
        def __init__(self, p1: float):
            self.p1 = p1

        def predict(self, ctx: int, bitplane: int) -> float:
            return self.p1

        def observe(self, ctx: int, bitplane: int, bit: int) -> None:
            pass

    for p1, bits in [(1e-9, [0] * 50 + [1] * 5), (1.0 - 1e-9, [1] * 50 + [0] * 5)]:
        sink = NeuralAcSink(capacity_bits=2000, predictor=_ExtremePredictor(p1))
        for bit in bits:
            sink.write(bit, ctx=0)
        bitstream = sink.to_bytes()

        source = NeuralAcSource(data=bitstream, total_bits=len(bitstream) * 8, predictor=_ExtremePredictor(p1))
        decoded = [source.read(ctx=0) for _ in bits]
        assert decoded == bits


# ---------------------------------------------------------------------------
# 3b. Full integration through spiht_encode/spiht_decode's generic sink=/source=
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_cr", [4.0, 8.0, 16.0])
def test_spiht_neural_ac_end_to_end_roundtrip(target_cr: float):
    rng = np.random.default_rng(4)
    cl = 32
    model = _build_test_model(cl)
    weights = extract_wavenet_bit_prior_weights(model, context_length=cl, kernel_size=3, bp_vocab=_BP_VOCAB)

    t = np.linspace(0, 1, 512, endpoint=False)
    sig = (np.sin(2 * np.pi * 5 * t) + 0.1 * rng.standard_normal(512)).astype(np.float32)
    coeffs = dwt_forward(sig, levels=6, wavelet="bior4.4")
    max_bits = int(512 * 16 / target_cr)

    enc_predictor = SpihtBitPredictor(weights)
    sink = NeuralAcSink(max_bits, enc_predictor)
    bitstream, meta = spiht_encode(coeffs.approx, coeffs.details, max_bits=max_bits, use_ac=True, sink=sink)

    assert meta["use_ac"] is True
    assert meta["n_bits"] <= max_bits + 64  # renorm spill tolerance, mirrors existing AC codec test

    # Decode twice with FRESH predictors — must be fully deterministic/reproducible.
    recons = []
    for _ in range(2):
        dec_predictor = SpihtBitPredictor(weights)
        source = NeuralAcSource(data=bitstream, total_bits=len(bitstream) * 8, predictor=dec_predictor)
        approx_r, details_r = spiht_decode(bitstream, meta, source=source)
        recon = dwt_inverse(WaveletCoeffs(approx=approx_r, details=details_r), wavelet="bior4.4")
        recons.append(recon)

    assert np.array_equal(recons[0], recons[1]), "decode must be deterministic given the same bitstream+weights"

    recon = recons[0]
    assert np.all(np.isfinite(recon))
    err = sig - recon[: len(sig)]
    prd = 100.0 * np.sqrt(np.sum(err**2) / (np.sum(sig**2) + 1e-12))
    assert prd < 80.0, f"PRD {prd:.1f}% implausibly high at CR {target_cr}x (untrained random model, sanity bound only)"


def test_spiht_neural_ac_all_zero_frame():
    """All-zero coefficients must short-circuit cleanly (matches existing SPIHT behavior)."""
    cl = 16
    model = _build_test_model(cl)
    weights = extract_wavenet_bit_prior_weights(model, context_length=cl, kernel_size=3, bp_vocab=_BP_VOCAB)

    approx = np.zeros(8, dtype=np.float32)
    details = [np.zeros(8, dtype=np.float32), np.zeros(16, dtype=np.float32)]

    enc_predictor = SpihtBitPredictor(weights)
    sink = NeuralAcSink(1000, enc_predictor)
    bitstream, meta = spiht_encode(approx, details, max_bits=1000, use_ac=True, sink=sink)
    assert bitstream == b""
    assert meta["n_bits"] == 0

    dec_predictor = SpihtBitPredictor(weights)
    source = NeuralAcSource(data=bitstream, total_bits=0, predictor=dec_predictor)
    approx_r, details_r = spiht_decode(bitstream, meta, source=source)
    assert np.all(approx_r == 0)
    for band in details_r:
        assert np.all(band == 0)
