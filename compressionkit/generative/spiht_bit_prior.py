"""Pure-numpy inference + pluggable AC sink/source for the SPIHT neural bit-prior.

Phase 2 of the neural-entropy-prior-for-SPIHT effort (see repo memory
``spiht-wavelet-findings.md``). The model is trained via Keras via
:func:`compressionkit.generative.causal_priors.build_wavenet_bit_prior`
(embeddings for context-class/bitplane/previous-bit + a gated-residual
causal dilated-Conv1D stack, sigmoid head). This module re-implements its
forward pass in plain numpy — no TF/Keras dependency at inference time — so
per-symbol incremental probability prediction during real SPIHT encode/decode
is fast and (eventually) portable to embedded C.

Everything neural-specific lives HERE, not in :mod:`compressionkit.dsp.spiht`.
That module only exposes a generic ``sink``/``source`` override point on
``spiht_encode``/``spiht_decode`` (dependency injection) — it has no notion of
"neural" anything. :class:`NeuralAcSink`/:class:`NeuralAcSource` here are one
possible implementation of that generic seam; the pipeline stays modular
(SPIHT's tree-traversal algorithm doesn't know or care what's behind the sink).

:class:`SpihtBitPredictor` is the incremental interface consumed by
:class:`NeuralAcSink`/:class:`NeuralAcSource`: one fresh predictor instance
per frame, ``predict(ctx, bitplane)`` before coding a bit, ``observe(ctx,
bitplane, bit)`` after. Its ring-buffer bookkeeping exactly reproduces the
causal windowing convention used at training time (verified against the
reference Keras model — see ``tests/test_spiht_bit_prior.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from compressionkit.dsp.spiht import BitReader, BitWriter

__all__ = [
    "NeuralAcSink",
    "NeuralAcSource",
    "PREVBIT_START",
    "SpihtBitPredictor",
    "WavenetBitPriorWeights",
    "extract_wavenet_bit_prior_weights",
    "wavenet_forward_numpy",
]

#: Sentinel "previous bit" value used for the very first position of a frame
#: (matches ``_PREVBIT_VOCAB - 1`` in the training script's ``build_features``
#: / ``build_sequence_windows``).
PREVBIT_START = 2

# Standard 32-bit binary range-coder constants (same scheme as
# `compressionkit.dsp.spiht.ArithEncoder`/`ArithDecoder` — these are a fixed
# mathematical convention, not a SPIHT-specific design choice, so they're
# defined locally rather than importing spiht.py's private copies).
_AC_TOP = 0xFFFFFFFF
_AC_HALF = 0x80000000
_AC_QTR = 0x40000000
_AC_3QTR = 0xC0000000

#: Total for quantizing a float probability into an integer split. Kept tiny
#: relative to `_AC_QTR` (a ~16k× safety margin) so even a probability very
#: close to 0 or 1 still gets a non-zero-width coding interval — this is the
#: exact fix for the zero-width-interval bug found earlier in the RVQ rANS/AC
#: work (see repo memory `entropy-prior-sweep.md`).
_NEURAL_AC_TOTAL = 1 << 16


@dataclass
class WavenetBitPriorWeights:
    """Numpy weights extracted from a trained ``build_wavenet_bit_prior`` model."""

    ctx_embed: np.ndarray  # (ctx_vocab, embed_dim)
    bp_embed: np.ndarray  # (bp_vocab, embed_dim)
    prevbit_embed: np.ndarray  # (prevbit_vocab, embed_dim)
    layers: list[dict[str, Any]]  # per-layer: dilation, gated_kernel/bias, resid_kernel/bias
    head_kernel: np.ndarray  # (embed_dim, 1)
    head_bias: np.ndarray  # (1,)
    kernel_size: int
    bp_vocab: int
    context_length: int


def extract_wavenet_bit_prior_weights(
    model: Any,
    *,
    context_length: int,
    kernel_size: int = 3,
    bp_vocab: int = 16,
) -> WavenetBitPriorWeights:
    """Pull numpy weight arrays out of a trained ``build_wavenet_bit_prior`` Keras model.

    Args:
        model: A compiled/trained model from ``build_wavenet_bit_prior``.
        context_length: The window length the model was trained/will be used with.
        kernel_size: Must match the model's ``kernel_size`` (default 3).
        bp_vocab: Must match the model's ``bp_vocab`` (default 16).
    """
    ctx_embed = model.get_layer("ctx_embed").get_weights()[0]
    bp_embed = model.get_layer("bp_embed").get_weights()[0]
    prevbit_embed = model.get_layer("prevbit_embed").get_weights()[0]

    layers: list[dict[str, Any]] = []
    i = 0
    while True:
        d = 2**i
        try:
            gated_kernel, gated_bias = model.get_layer(f"gated_conv_d{d}").get_weights()
        except ValueError:
            break
        resid_kernel, resid_bias = model.get_layer(f"resid_proj_d{d}").get_weights()
        layers.append(
            {
                "dilation": d,
                "gated_kernel": gated_kernel,  # (kernel_size, embed_dim, 2*embed_dim)
                "gated_bias": gated_bias,  # (2*embed_dim,)
                "resid_kernel": resid_kernel[0],  # (embed_dim, embed_dim) — squeeze the 1x1 conv's kernel_size axis
                "resid_bias": resid_bias,  # (embed_dim,)
            }
        )
        i += 1
    if not layers:
        raise ValueError("Could not find any gated_conv_d* layers on the given model.")

    head_kernel, head_bias = model.get_layer("bit_logit").get_weights()
    return WavenetBitPriorWeights(
        ctx_embed=ctx_embed,
        bp_embed=bp_embed,
        prevbit_embed=prevbit_embed,
        layers=layers,
        head_kernel=head_kernel,
        head_bias=head_bias,
        kernel_size=kernel_size,
        bp_vocab=bp_vocab,
        context_length=context_length,
    )


def wavenet_forward_numpy(
    ctx_ids: np.ndarray,
    bp_ids: np.ndarray,
    prevbit_ids: np.ndarray,
    weights: WavenetBitPriorWeights,
) -> np.ndarray:
    """Vectorized causal forward pass over one window; returns P(bit=1) at every position.

    Args:
        ctx_ids, bp_ids, prevbit_ids: ``(L,)`` int arrays (same window length L).
        weights: Extracted model weights.

    Returns:
        ``(L,)`` float64 array of P(bit=1) for each position in the window.
    """
    x = (
        weights.ctx_embed[ctx_ids]
        + weights.bp_embed[bp_ids]
        + weights.prevbit_embed[prevbit_ids]
    ).astype(np.float64)
    length, channels = x.shape
    k = weights.kernel_size

    for layer in weights.layers:
        d = int(layer["dilation"])
        pad = (k - 1) * d
        x_padded = np.pad(x, ((pad, 0), (0, 0)))
        acc = np.zeros((length, 2 * channels), dtype=np.float64)
        for j in range(k):
            shift = (k - 1 - j) * d
            acc += x_padded[pad - shift : pad - shift + length] @ layer["gated_kernel"][j]
        acc += layer["gated_bias"]
        a, b = acc[:, :channels], acc[:, channels:]
        gated = np.tanh(a) * (1.0 / (1.0 + np.exp(-b)))
        resid = gated @ layer["resid_kernel"] + layer["resid_bias"]
        x = x + resid

    logits = x @ weights.head_kernel + weights.head_bias  # (L, 1)
    return 1.0 / (1.0 + np.exp(-logits[:, 0]))


@dataclass
class SpihtBitPredictor:
    """Incremental, causal P(bit=1) predictor for ONE SPIHT frame's emission stream.

    A fresh instance must be constructed per frame — state (ring buffers) is
    scoped to a single frame and is NOT reset automatically. The window grows
    from length 1 at the start of a frame up to ``weights.context_length``,
    then slides (drops the oldest position) — this exactly reproduces the
    fixed-length teacher-forced windows used at training time (see
    ``build_sequence_windows`` in the training script), just computed one
    position at a time instead of many windows in a batch.
    """

    weights: WavenetBitPriorWeights
    _ctx_hist: list[int] = field(default_factory=list)
    _bp_hist: list[int] = field(default_factory=list)
    _bit_hist: list[int] = field(default_factory=list)

    def predict(self, ctx: int, bitplane: int) -> float:
        """Return P(bit=1) for the NEXT position, given (ctx, bitplane) at that position."""
        cl = self.weights.context_length
        bp_clip = int(np.clip(bitplane, 0, self.weights.bp_vocab - 1))
        ctx_win = (self._ctx_hist + [ctx])[-cl:]
        bp_win = (self._bp_hist + [bp_clip])[-cl:]
        prev_win = ([PREVBIT_START] + self._bit_hist)[-cl:]
        p1_all = wavenet_forward_numpy(
            np.asarray(ctx_win, dtype=np.int32),
            np.asarray(bp_win, dtype=np.int32),
            np.asarray(prev_win, dtype=np.int32),
            self.weights,
        )
        return float(p1_all[-1])

    def observe(self, ctx: int, bitplane: int, bit: int) -> None:
        """Record the actual (ctx, bitplane, bit) once it's known (encoded or decoded)."""
        bp_clip = int(np.clip(bitplane, 0, self.weights.bp_vocab - 1))
        self._ctx_hist.append(int(ctx))
        self._bp_hist.append(bp_clip)
        self._bit_hist.append(int(bit))


# ---------------------------------------------------------------------------
# Pluggable AC sink/source — implements the generic ``sink``/``source`` seam
# on ``spiht_encode``/``spiht_decode`` (see compressionkit/dsp/spiht.py).
# spiht.py has NO knowledge of these classes; they just happen to satisfy its
# generic ``write(bit, ctx)`` / ``read(ctx)`` / ``bits_out`` / ``to_bytes()``
# / ``set_bitplane(n)`` duck-typed protocol. Range-coding math (renormalize
# loop) mirrors `ArithEncoder`/`ArithDecoder` exactly; the only difference is
# where the probability comes from.
# ---------------------------------------------------------------------------


class NeuralAcSink:
    """AC sink driven by a :class:`SpihtBitPredictor` instead of fixed per-context counts.

    Satisfies the generic sink protocol expected by ``spiht_encode(..., sink=)``.
    Construct a FRESH instance per frame — pass ``use_ac=True`` alongside it so
    ``spiht_encode`` uses symbol-count budget accounting (matches any AC-style,
    variable-bits-per-symbol coder).
    """

    def __init__(self, capacity_bits: int, predictor: SpihtBitPredictor):
        self.writer = BitWriter(capacity_bits=capacity_bits + 64)
        self.capacity_bits = capacity_bits
        self.predictor = predictor
        self.low = 0
        self.high = _AC_TOP
        self.pending = 0
        self._cur_bitplane = 0

    def set_bitplane(self, n: int) -> None:
        self._cur_bitplane = n

    @property
    def bits_out(self) -> int:
        return self.writer.bit_pos + self.pending

    def _emit(self, bit: int) -> None:
        self.writer.write_bit(bit)
        for _ in range(self.pending):
            self.writer.write_bit(1 - bit)
        self.pending = 0

    def write(self, bit: int, ctx: int) -> None:
        p1 = self.predictor.predict(ctx, self._cur_bitplane)
        c1 = min(max(int(round(p1 * _NEURAL_AC_TOTAL)), 1), _NEURAL_AC_TOTAL - 1)
        c0 = _NEURAL_AC_TOTAL - c1
        rng = self.high - self.low + 1
        split = self.low + (rng * c0) // _NEURAL_AC_TOTAL - 1
        if bit:
            self.low = split + 1
        else:
            self.high = split
        while True:
            if self.high < _AC_HALF:
                self._emit(0)
            elif self.low >= _AC_HALF:
                self._emit(1)
                self.low -= _AC_HALF
                self.high -= _AC_HALF
            elif self.low >= _AC_QTR and self.high < _AC_3QTR:
                self.pending += 1
                self.low -= _AC_QTR
                self.high -= _AC_QTR
            else:
                break
            self.low = (self.low << 1) & _AC_TOP
            self.high = ((self.high << 1) | 1) & _AC_TOP
        self.predictor.observe(ctx, self._cur_bitplane, bit)

    def to_bytes(self) -> bytes:
        self.pending += 1
        if self.low < _AC_QTR:
            self._emit(0)
        else:
            self._emit(1)
        return self.writer.to_bytes()


class NeuralAcSource:
    """AC source mirroring :class:`NeuralAcSink` for decode.

    Satisfies the generic source protocol expected by ``spiht_decode(...,
    source=)``. Construct a FRESH instance per frame, using a FRESH
    :class:`SpihtBitPredictor` with the SAME weights used at encode time (its
    history starts empty and is rebuilt purely from decoded bits, so encode
    and decode stay in lock-step as long as decoding is correct).
    """

    def __init__(self, data: bytes, total_bits: int, predictor: SpihtBitPredictor):
        self.reader = BitReader(data=data, total_bits=total_bits)
        self.predictor = predictor
        self.low = 0
        self.high = _AC_TOP
        self.code = 0
        self._cur_bitplane = 0
        for _ in range(32):
            self.code = (self.code << 1) | self._read_input()

    def set_bitplane(self, n: int) -> None:
        self._cur_bitplane = n

    def _read_input(self) -> int:
        if self.reader.bit_pos >= self.reader.total_bits:
            return 0
        return self.reader.read_bit()

    def read(self, ctx: int) -> int:
        p1 = self.predictor.predict(ctx, self._cur_bitplane)
        c1 = min(max(int(round(p1 * _NEURAL_AC_TOTAL)), 1), _NEURAL_AC_TOTAL - 1)
        c0 = _NEURAL_AC_TOTAL - c1
        rng = self.high - self.low + 1
        split = self.low + (rng * c0) // _NEURAL_AC_TOTAL - 1
        if self.code <= split:
            bit = 0
            self.high = split
        else:
            bit = 1
            self.low = split + 1
        while True:
            if self.high < _AC_HALF:
                pass
            elif self.low >= _AC_HALF:
                self.low -= _AC_HALF
                self.high -= _AC_HALF
                self.code -= _AC_HALF
            elif self.low >= _AC_QTR and self.high < _AC_3QTR:
                self.low -= _AC_QTR
                self.high -= _AC_QTR
                self.code -= _AC_QTR
            else:
                break
            self.low = (self.low << 1) & _AC_TOP
            self.high = ((self.high << 1) | 1) & _AC_TOP
            self.code = ((self.code << 1) | self._read_input()) & _AC_TOP
        self.predictor.observe(ctx, self._cur_bitplane, bit)
        return bit

