"""Two-stage codec: RVQ codec + entropy prior for maximum compression.

Combines :class:`RVQCodec` (encoder → VQ → decoder) with
:class:`EntropyPrior` (causal model that predicts token probabilities)
to achieve compression beyond the uniform-codebook baseline.

The two stages:

1. **Codec stage** — ``RVQCodec.encode(signal)`` → RVQ indices.
2. **Prior stage** — ``EntropyPrior`` predicts per-token probabilities;
   an entropy coder uses those probabilities to compress the indices
   into a compact bitstream. Two backends are available:

   * ``"arithmetic"`` (default) — classical bit-level arithmetic coding.
   * ``"rans"`` — range Asymmetric Numeral Systems, the table-driven,
     byte-oriented successor used by Zstd/JPEG XL. Same compression ratio
     (up to quantization) as arithmetic coding, but the standard choice for
     production/embedded entropy coders since it avoids a per-bit
     renormalization loop.

This module requires only ``numpy`` and a LiteRT interpreter (plus
``compressionkit.runtime.codec`` and ``compressionkit.runtime.prior``).
"""

from __future__ import annotations

import bisect
import logging
import struct
from dataclasses import dataclass

import numpy as np

from compressionkit.runtime.codec import RVQCodec
from compressionkit.runtime.prior import EntropyPrior

logger = logging.getLogger(__name__)

# Arithmetic coder precision constants
_PRECISION = 24
_WHOLE = 1 << _PRECISION
_HALF = _WHOLE >> 1
_QUARTER = _WHOLE >> 2
_MASK = _WHOLE - 1

# CDF quantization total for arithmetic coding. Must be safely smaller than
# _QUARTER (the coder's minimum renormalized range) so that even a
# minimum-count-1 symbol maps to a non-zero-width interval:
# rng * count // _CDF_TOTAL >= _QUARTER // _CDF_TOTAL >= 1. Using _WHOLE here
# (as a prior version of this code did) is a bug: it allows zero-width
# intervals -- and an infinite low==high renormalization loop -- whenever a
# confident prior assigns a rare symbol a count of exactly 1 while rng has
# shrunk to _QUARTER. This was found via real (non-synthetic) trained-prior
# probabilities, which are peaked enough to trigger it; toy/uniform test
# probabilities never did.
_CDF_TOTAL_BITS = 16
_CDF_TOTAL = 1 << _CDF_TOTAL_BITS

# rANS precision constants
_RANS_SCALE_BITS = 16
_RANS_TOTAL = 1 << _RANS_SCALE_BITS
_RANS_L = 1 << 23  # lower bound of the normalized state interval


@dataclass
class CompressionResult:
    """Result of two-stage compression."""

    bitstream: bytes
    """Compressed bitstream."""

    num_tokens: int
    """Total number of RVQ tokens (T' × num_levels)."""

    indices_shape: tuple[int, ...]
    """Original shape of the RVQ indices array."""

    bits_per_token_prior: float
    """Effective bits-per-token estimated by the entropy prior."""

    bits_per_token_actual: float
    """Actual bits-per-token of the compressed bitstream."""

    bits_per_token_uniform: float
    """Bits-per-token under uniform coding (log2(K))."""

    backend: str = "arithmetic"
    """Entropy coder backend used to produce ``bitstream`` ("arithmetic" or "rans")."""

    @property
    def cr_uniform(self) -> float:
        """Compression ratio vs uniform coding."""
        if self.bits_per_token_actual == 0:
            return float("inf")
        return self.bits_per_token_uniform / self.bits_per_token_actual

    @property
    def cr_uplift(self) -> float:
        """CR uplift factor from entropy coding (>1 means improvement)."""
        if self.bits_per_token_actual == 0:
            return float("inf")
        return self.bits_per_token_uniform / self.bits_per_token_actual


class TwoStageCodec:
    """Two-stage compression: RVQ codec + entropy-coded prior.

    Args:
        codec: An :class:`RVQCodec` instance.
        prior: An :class:`EntropyPrior` instance.

    Example::

        from compressionkit.runtime import RVQCodec
        from compressionkit.runtime.prior import EntropyPrior
        from compressionkit.runtime.two_stage import TwoStageCodec

        codec = RVQCodec("deploy/")
        prior = EntropyPrior("prior.tflite")
        two_stage = TwoStageCodec(codec, prior)

        result = two_stage.compress(signal)
        print(f"CR uplift: {result.cr_uplift:.2f}x")

        recon = two_stage.decompress(result)
    """

    def __init__(self, codec: RVQCodec, prior: EntropyPrior) -> None:
        self._codec = codec
        self._prior = prior

    @property
    def codec(self) -> RVQCodec:
        """The underlying RVQ codec."""
        return self._codec

    @property
    def prior(self) -> EntropyPrior:
        """The entropy prior model."""
        return self._prior

    def compress(self, signal: np.ndarray) -> CompressionResult:
        """Compress a signal using two-stage coding.

        1. Encode signal → RVQ indices via the codec.
        2. Compute token probabilities via the entropy prior.
        3. Arithmetic-code the indices using those probabilities.

        Args:
            signal: Input signal matching codec encoder input shape
                (e.g. ``(1, 1, 320, 1)`` float32).

        Returns:
            A :class:`CompressionResult` with the bitstream and metrics.
        """
        # Stage 1: Codec encode
        indices = self._codec.encode(signal)

        # Stage 2: Get prior probabilities and arithmetic-code
        return self.compress_indices(indices)

    def compress_indices(self, indices: np.ndarray, backend: str = "arithmetic") -> CompressionResult:
        """Compress pre-computed RVQ indices using entropy coding.

        Args:
            indices: RVQ indices from ``codec.encode()``.
            backend: ``"arithmetic"`` (default) or ``"rans"``.

        Returns:
            A :class:`CompressionResult`.
        """
        # Flatten to token sequence
        tokens = self._flatten(indices)
        num_tokens = tokens.shape[0]

        indices_4d = indices
        if indices.ndim == 3:
            indices_4d = indices[np.newaxis]

        # Single prior pass: `_get_probs` already computes the exact
        # per-position probability table `predict_log_probs`/`bits_per_token`
        # would recompute from scratch (same windowed-context convention, same
        # token order — see `_get_probs`'s docstring). Deriving `bpt_prior`
        # from it directly avoids a second full pass over the prior (each
        # position is a separate TFLite interpreter invocation, so a second
        # pass roughly doubles this method's runtime for no new information).
        probs_per_position = self._get_probs(indices_4d)
        token_probs = np.clip(probs_per_position[np.arange(num_tokens), tokens], 1e-12, None)
        bpt_prior = float(-np.mean(np.log(token_probs)) / np.log(2))

        if backend == "arithmetic":
            bitstream = _arithmetic_encode(tokens, probs_per_position, self._prior.vocab_size)
        elif backend == "rans":
            bitstream = _rans_encode(tokens, probs_per_position, self._prior.vocab_size)
        else:
            raise ValueError(f"Unknown backend {backend!r}; expected 'arithmetic' or 'rans'")

        bits_uniform = float(np.log2(self._codec.num_embeddings))
        bpt_actual = len(bitstream) * 8 / max(num_tokens, 1)

        return CompressionResult(
            bitstream=bitstream,
            num_tokens=num_tokens,
            indices_shape=indices.shape,
            bits_per_token_prior=bpt_prior,
            bits_per_token_actual=bpt_actual,
            bits_per_token_uniform=bits_uniform,
            backend=backend,
        )

    def decompress(self, result: CompressionResult) -> np.ndarray:
        """Decompress a bitstream back to a signal.

        1. Arithmetic-decode the bitstream → RVQ indices.
        2. Decode indices → signal via the codec.

        Args:
            result: A :class:`CompressionResult` from ``compress()``.

        Returns:
            Reconstructed signal array.
        """
        indices = self.decompress_indices(result)
        return self._codec.decode(indices)

    def decompress_indices(self, result: CompressionResult) -> np.ndarray:
        """Decompress a bitstream back to RVQ indices.

        Args:
            result: A :class:`CompressionResult` from ``compress()``.

        Returns:
            RVQ indices array with shape ``result.indices_shape``.
        """
        # We need to decode autoregressively: each token's probability
        # depends on all previous tokens via the prior.
        if result.backend == "arithmetic":
            decode_fn = _arithmetic_decode_with_prior
        elif result.backend == "rans":
            decode_fn = _rans_decode_with_prior
        else:
            raise ValueError(f"Unknown backend {result.backend!r}; expected 'arithmetic' or 'rans'")
        indices = decode_fn(
            bitstream=result.bitstream,
            num_tokens=result.num_tokens,
            indices_shape=result.indices_shape,
            prior=self._prior,
            vocab_size=self._prior.vocab_size,
        )
        return indices

    def estimate_bitrate(self, indices: np.ndarray) -> dict[str, float]:
        """Estimate compression metrics without actually encoding.

        Faster than ``compress()`` — only runs the prior, no arithmetic
        coding.

        Args:
            indices: RVQ indices from ``codec.encode()``.

        Returns:
            Dict with ``bits_per_token_prior``, ``bits_per_token_uniform``,
            ``cr_uplift``, and ``cr_codec_learned``.
        """
        if indices.ndim == 3:
            indices = indices[np.newaxis]
        bpt = self._prior.bits_per_token(indices)
        bpt_uniform = float(np.log2(self._codec.num_embeddings))
        cr_uplift = bpt_uniform / bpt if bpt > 0 else float("inf")

        return {
            "bits_per_token_prior": bpt,
            "bits_per_token_uniform": bpt_uniform,
            "cr_uplift": cr_uplift,
        }

    def _flatten(self, indices: np.ndarray) -> np.ndarray:
        """Flatten indices to a 1-D token sequence."""
        if indices.ndim == 4:
            indices = indices[0]
        if indices.ndim == 3:
            indices = indices[0]
        # Now (T', num_levels) → flatten
        return indices.reshape(-1).astype(np.int32)

    def _get_probs(self, indices: np.ndarray) -> np.ndarray:
        """Get per-position probability distributions from the prior.

        Returns:
            Array of shape ``(num_tokens, vocab_size)`` with probabilities.
        """
        tokens = self._prior._flatten_indices(indices)  # (1, seq_len)
        num_tokens = tokens.shape[1]
        probs = np.full((num_tokens, self._prior.vocab_size), 1.0 / self._prior.vocab_size, dtype=np.float32)

        for pos in range(1, num_tokens):
            start = max(0, pos - self._prior.context_length)
            probs[pos] = self._prior.predict_next_probs(tokens[:, start:pos])[0]

        return probs


# ── Arithmetic coding ────────────────────────────────────────────


def _probs_to_cdf(probs: np.ndarray) -> list[int]:
    """Convert a probability vector to an integer CDF for arithmetic coding.

    Returns a list of length ``len(probs) + 1`` with values in ``[0, _CDF_TOTAL]``,
    where ``cdf[0] = 0`` and ``cdf[-1] = _CDF_TOTAL``.  Each symbol is guaranteed
    at least 1 count to avoid zero-width intervals.
    """
    n = len(probs)
    # Cast to plain Python floats so quantization is bit-identical regardless
    # of whether the caller passes a numpy float32 array (encode side) or a
    # Python list from `.tolist()` (decode side). Without this, numpy
    # float32 arithmetic vs Python float64 arithmetic can round `p * total`
    # differently right at a .5 boundary, making the encoder and decoder
    # quantize to different counts for the same probability -- silently
    # corrupting the roundtrip.
    probs = [float(p) for p in probs]
    # Quantize to integer counts, ensuring each symbol gets ≥ 1
    counts = [max(1, round(p * (_CDF_TOTAL - n))) for p in probs]
    # Build CDF
    cdf = [0] * (n + 1)
    for i in range(n):
        cdf[i + 1] = cdf[i] + counts[i]
    # Normalize so cdf[-1] == _CDF_TOTAL by adjusting the largest bucket
    diff = _CDF_TOTAL - cdf[-1]
    if diff != 0:
        # Add the residual to the largest-count symbol
        max_idx = max(range(n), key=lambda i: counts[i])
        for j in range(max_idx + 1, n + 1):
            cdf[j] += diff
    return cdf


def _arithmetic_encode(tokens: np.ndarray, probs: np.ndarray, vocab_size: int) -> bytes:
    """Arithmetic-encode a token sequence given per-position probabilities.

    Args:
        tokens: 1-D int32 array of token values.
        probs: ``(num_tokens, vocab_size)`` probability table.
        vocab_size: Size of the token alphabet.

    Returns:
        Compressed bytes (4-byte big-endian token count header + coded bits).
    """
    num_tokens = len(tokens)

    # Precompute CDFs (Python int lists — no overflow risk)
    cdfs = [_probs_to_cdf(probs[t]) for t in range(num_tokens)]

    low = 0
    high = _WHOLE
    pending_bits = 0
    output_bits: list[int] = []

    for t in range(num_tokens):
        tok = int(tokens[t])
        cdf = cdfs[t]
        rng = high - low

        high = low + rng * cdf[tok + 1] // _CDF_TOTAL
        low = low + rng * cdf[tok] // _CDF_TOTAL

        while True:
            if high <= _HALF:
                output_bits.append(0)
                output_bits.extend([1] * pending_bits)
                pending_bits = 0
                low <<= 1
                high <<= 1
            elif low >= _HALF:
                output_bits.append(1)
                output_bits.extend([0] * pending_bits)
                pending_bits = 0
                low = (low - _HALF) << 1
                high = (high - _HALF) << 1
            elif low >= _QUARTER and high <= 3 * _QUARTER:
                pending_bits += 1
                low = (low - _QUARTER) << 1
                high = (high - _QUARTER) << 1
            else:
                break

    # Flush
    pending_bits += 1
    if low < _QUARTER:
        output_bits.append(0)
        output_bits.extend([1] * pending_bits)
    else:
        output_bits.append(1)
        output_bits.extend([0] * pending_bits)

    # Pack bits → bytes with 4-byte header
    header = struct.pack(">I", num_tokens)
    bit_bytes = bytearray()
    for i in range(0, len(output_bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(output_bits):
                byte = (byte << 1) | output_bits[i + j]
            else:
                byte <<= 1
        bit_bytes.append(byte)

    return header + bytes(bit_bytes)


def _arithmetic_decode_with_prior(
    bitstream: bytes,
    num_tokens: int,
    indices_shape: tuple[int, ...],
    prior: EntropyPrior,
    vocab_size: int,
) -> np.ndarray:
    """Arithmetic-decode a bitstream autoregressively using the prior.

    Each decoded token is fed back into the prior to get the probability
    distribution for the next token.

    Args:
        bitstream: Compressed bytes from :func:`_arithmetic_encode`.
        num_tokens: Number of tokens to decode.
        indices_shape: Original shape of the RVQ indices.
        prior: The entropy prior model.
        vocab_size: Token vocabulary size.

    Returns:
        Decoded RVQ indices with shape ``indices_shape``.
    """
    header_size = 4
    stored_num = struct.unpack(">I", bitstream[:header_size])[0]
    if stored_num != num_tokens:
        raise ValueError(f"Token count mismatch: header={stored_num}, expected={num_tokens}")

    # Unpack bits
    bit_data = bitstream[header_size:]
    bits: list[int] = []
    for byte in bit_data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    def _read_bit() -> int:
        nonlocal bit_idx
        b = bits[bit_idx] if bit_idx < len(bits) else 0
        bit_idx += 1
        return b

    decoded_tokens = np.zeros(num_tokens, dtype=np.int32)
    low = 0
    high = _WHOLE
    bit_idx = 0

    # Bootstrap value register
    value = 0
    for _ in range(_PRECISION):
        value = (value << 1) | _read_bit()

    for t in range(num_tokens):
        # Get probability distribution for this position
        if t == 0:
            probs_t = [1.0 / vocab_size] * vocab_size
        else:
            start = max(0, t - prior.context_length)
            tokens_so_far = decoded_tokens[start:t].reshape(1, -1)
            probs_t = prior.predict_next_probs(tokens_so_far)[0].tolist()

        cdf = _probs_to_cdf(probs_t)

        # Decode symbol
        rng = high - low
        scaled = ((value - low + 1) * _CDF_TOTAL - 1) // rng

        # Binary search for symbol
        sym = 0
        for s in range(vocab_size):
            if cdf[s + 1] > scaled:
                sym = s
                break

        decoded_tokens[t] = sym

        # Update interval
        high = low + rng * cdf[sym + 1] // _CDF_TOTAL
        low = low + rng * cdf[sym] // _CDF_TOTAL

        # Renormalize
        while True:
            if high <= _HALF:
                low <<= 1
                high <<= 1
                value = (value << 1) | _read_bit()
            elif low >= _HALF:
                low = (low - _HALF) << 1
                high = (high - _HALF) << 1
                value = ((value - _HALF) << 1) | _read_bit()
            elif low >= _QUARTER and high <= 3 * _QUARTER:
                low = (low - _QUARTER) << 1
                high = (high - _QUARTER) << 1
                value = ((value - _QUARTER) << 1) | _read_bit()
            else:
                break

    return decoded_tokens.reshape(indices_shape)


# ── rANS (range Asymmetric Numeral Systems) coding ──────────────
#
# The table-driven, byte-oriented successor to classical arithmetic coding
# (same compression ratio up to quantization; no per-bit renormalization
# loop). Standard reference structure (Fabian Giesen's public-domain
# "rans_byte.h"), adapted here to support a *different* probability table
# per position (needed since our prior is autoregressive/context-dependent,
# unlike the static single-table case rANS is usually demonstrated with).
#
# Key property that makes this work with an autoregressive prior: rANS is a
# LIFO stack, so encoding must process symbols in *reverse* order (this is
# fine — during encoding we already know every token, so we can precompute
# every position's probability table with one forward pass first). Decoding
# then naturally proceeds in *forward* order, which is exactly what we need
# to feed each decoded token back into the prior for the next position.


def _probs_to_rans_freqs(probs: np.ndarray, total: int = _RANS_TOTAL) -> tuple[list[int], list[int]]:
    """Quantize a probability vector into integer ``(starts, freqs)`` summing to ``total``.

    Same quantization convention as :func:`_probs_to_cdf` (every symbol gets
    at least one count, residual assigned to the largest bucket), just
    returned as separate start/freq tables since that's the natural rANS
    representation.
    """
    n = len(probs)
    # See _probs_to_cdf for why this cast matters: encode passes raw numpy
    # float32 rows while decode passes float64-cast arrays, and quantizing
    # each in its native precision can round `p * total` differently at a
    # .5 boundary, corrupting the roundtrip.
    probs = [float(p) for p in probs]
    counts = [max(1, round(p * (total - n))) for p in probs]
    diff = total - sum(counts)
    if diff != 0:
        max_idx = max(range(n), key=lambda i: counts[i])
        counts[max_idx] += diff
    starts = [0] * n
    acc = 0
    for i in range(n):
        starts[i] = acc
        acc += counts[i]
    return starts, counts


def _rans_encode(tokens: np.ndarray, probs: np.ndarray, vocab_size: int) -> bytes:
    """rANS-encode a token sequence given per-position probabilities.

    Same interface/framing convention as :func:`_arithmetic_encode` (4-byte
    big-endian token-count header + coded payload).

    Args:
        tokens: 1-D int32 array of token values.
        probs: ``(num_tokens, vocab_size)`` probability table.
        vocab_size: Size of the token alphabet.

    Returns:
        Compressed bytes (4-byte big-endian token count header + rANS payload).
    """
    num_tokens = len(tokens)
    freq_tables = [_probs_to_rans_freqs(probs[t]) for t in range(num_tokens)]

    x = _RANS_L
    out_bytes: list[int] = []
    for t in reversed(range(num_tokens)):
        tok = int(tokens[t])
        starts, freqs = freq_tables[t]
        start, freq = starts[tok], freqs[tok]
        x_max = ((_RANS_L >> _RANS_SCALE_BITS) << 8) * freq
        while x >= x_max:
            out_bytes.append(x & 0xFF)
            x >>= 8
        x = ((x // freq) << _RANS_SCALE_BITS) + (x % freq) + start

    # Flush the final state as 4 bytes (same renormalization byte order).
    for _ in range(4):
        out_bytes.append(x & 0xFF)
        x >>= 8
    out_bytes.reverse()

    header = struct.pack(">I", num_tokens)
    return header + bytes(out_bytes)


def _rans_decode_with_prior(
    bitstream: bytes,
    num_tokens: int,
    indices_shape: tuple[int, ...],
    prior: EntropyPrior,
    vocab_size: int,
) -> np.ndarray:
    """rANS-decode a bitstream autoregressively using the prior.

    Each decoded token is fed back into the prior to get the probability
    distribution for the next token — same autoregressive contract as
    :func:`_arithmetic_decode_with_prior`.

    Args:
        bitstream: Compressed bytes from :func:`_rans_encode`.
        num_tokens: Number of tokens to decode.
        indices_shape: Original shape of the RVQ indices.
        prior: The entropy prior model.
        vocab_size: Token vocabulary size.

    Returns:
        Decoded RVQ indices with shape ``indices_shape``.
    """
    header_size = 4
    stored_num = struct.unpack(">I", bitstream[:header_size])[0]
    if stored_num != num_tokens:
        raise ValueError(f"Token count mismatch: header={stored_num}, expected={num_tokens}")

    buf = bitstream[header_size:]
    pos = 0

    def _read_byte() -> int:
        nonlocal pos
        b = buf[pos] if pos < len(buf) else 0
        pos += 1
        return b

    x = 0
    for _ in range(4):
        x = (x << 8) | _read_byte()

    decoded_tokens = np.zeros(num_tokens, dtype=np.int32)
    for t in range(num_tokens):
        if t == 0:
            probs_t = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)
        else:
            start = max(0, t - prior.context_length)
            tokens_so_far = decoded_tokens[start:t].reshape(1, -1)
            probs_t = prior.predict_next_probs(tokens_so_far)[0].astype(np.float64)
        starts, freqs = _probs_to_rans_freqs(probs_t)

        slot = x & (_RANS_TOTAL - 1)
        # `starts` is a strictly increasing cumulative sum (every symbol gets
        # at least one count in _probs_to_rans_freqs), so the symbol whose
        # bucket contains `slot` can be found with a binary search instead of
        # an O(vocab_size) linear scan — matters since this runs once per
        # decoded token.
        sym = bisect.bisect_right(starts, slot) - 1
        decoded_tokens[t] = sym

        x = freqs[sym] * (x >> _RANS_SCALE_BITS) + slot - starts[sym]
        while x < _RANS_L:
            x = (x << 8) | _read_byte()

    return decoded_tokens.reshape(indices_shape)
