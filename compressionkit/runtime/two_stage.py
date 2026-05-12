"""Two-stage codec: RVQ codec + entropy prior for maximum compression.

Combines :class:`RVQCodec` (encoder → VQ → decoder) with
:class:`EntropyPrior` (causal model that predicts token probabilities)
to achieve compression beyond the uniform-codebook baseline.

The two stages:

1. **Codec stage** — ``RVQCodec.encode(signal)`` → RVQ indices.
2. **Prior stage** — ``EntropyPrior`` predicts per-token probabilities;
   an arithmetic coder uses those probabilities to compress the indices
   into a compact bitstream.

This module requires only ``numpy`` and a LiteRT interpreter (plus
``compressionkit.runtime.codec`` and ``compressionkit.runtime.prior``).
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

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

    def compress_indices(self, indices: np.ndarray) -> CompressionResult:
        """Compress pre-computed RVQ indices using entropy coding.

        Args:
            indices: RVQ indices from ``codec.encode()``.

        Returns:
            A :class:`CompressionResult`.
        """
        # Flatten to token sequence
        tokens = self._flatten(indices)
        num_tokens = tokens.shape[0]

        # Get prior log-probs
        indices_4d = indices
        if indices.ndim == 3:
            indices_4d = indices[np.newaxis]
        bpt_prior = self._prior.bits_per_token(indices_4d)

        # Get probability table for arithmetic coding
        probs_per_position = self._get_probs(indices_4d)

        # Arithmetic encode
        bitstream = _arithmetic_encode(tokens, probs_per_position, self._prior.vocab_size)

        bits_uniform = float(np.log2(self._codec.num_embeddings))
        bpt_actual = len(bitstream) * 8 / max(num_tokens, 1)

        return CompressionResult(
            bitstream=bitstream,
            num_tokens=num_tokens,
            indices_shape=indices.shape,
            bits_per_token_prior=bpt_prior,
            bits_per_token_actual=bpt_actual,
            bits_per_token_uniform=bits_uniform,
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
        indices = _arithmetic_decode_with_prior(
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
        logits = self._prior.predict_logits(tokens)  # (1, seq_len, vocab_size)
        logits = logits[0]  # (seq_len, vocab_size)

        # Convert to probabilities via softmax
        # For position t, logits[t-1] predict token[t]
        # Position 0 uses uniform prior
        num_tokens = logits.shape[0]
        probs = np.full((num_tokens, logits.shape[1]), 1.0 / logits.shape[1], dtype=np.float32)

        if num_tokens > 1:
            shifted = logits[:-1]  # (seq_len-1, vocab_size)
            max_l = np.max(shifted, axis=-1, keepdims=True)
            exp_l = np.exp(shifted - max_l)
            probs[1:] = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

        # Clamp to avoid zero probabilities
        probs = np.clip(probs, 1e-8, None)
        probs = probs / probs.sum(axis=-1, keepdims=True)

        return probs


# ── Arithmetic coding ────────────────────────────────────────────


def _probs_to_cdf(probs: np.ndarray) -> list[int]:
    """Convert a probability vector to an integer CDF for arithmetic coding.

    Returns a list of length ``len(probs) + 1`` with values in ``[0, _WHOLE]``,
    where ``cdf[0] = 0`` and ``cdf[-1] = _WHOLE``.  Each symbol is guaranteed
    at least 1 count to avoid zero-width intervals.
    """
    n = len(probs)
    # Quantize to integer counts, ensuring each symbol gets ≥ 1
    counts = [max(1, int(round(p * (_WHOLE - n)))) for p in probs]
    total = sum(counts)
    # Build CDF
    cdf = [0] * (n + 1)
    for i in range(n):
        cdf[i + 1] = cdf[i] + counts[i]
    # Normalize so cdf[-1] == _WHOLE by adjusting the largest bucket
    diff = _WHOLE - cdf[-1]
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

        high = low + rng * cdf[tok + 1] // _WHOLE
        low = low + rng * cdf[tok] // _WHOLE

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
    prior: "EntropyPrior",
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
            tokens_so_far = decoded_tokens[:t].reshape(1, -1)
            logits = prior.predict_logits(tokens_so_far)
            logit_last = logits[0, -1, :]
            max_l = float(np.max(logit_last))
            exp_l = np.exp(logit_last - max_l)
            softmax = exp_l / np.sum(exp_l)
            softmax = np.clip(softmax, 1e-8, None)
            softmax = softmax / softmax.sum()
            probs_t = softmax.tolist()

        cdf = _probs_to_cdf(probs_t)

        # Decode symbol
        rng = high - low
        scaled = ((value - low + 1) * _WHOLE - 1) // rng

        # Binary search for symbol
        sym = 0
        for s in range(vocab_size):
            if cdf[s + 1] > scaled:
                sym = s
                break

        decoded_tokens[t] = sym

        # Update interval
        high = low + rng * cdf[sym + 1] // _WHOLE
        low = low + rng * cdf[sym] // _WHOLE

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
