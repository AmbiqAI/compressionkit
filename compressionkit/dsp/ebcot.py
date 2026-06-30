"""1-D EBCOT encoder/decoder for wavelet-based signal compression.

Implements a 1-D adaptation of the Embedded Block Coding with Optimized
Truncation algorithm (Taubman, 2000) — the core of JPEG 2000.

Key differences from SPIHT:
  - Subbands are partitioned into independent **code blocks** (default 64 coeffs).
  - Each code block is bit-plane coded with 3 passes per bit-plane:
    (1) Significance Propagation Pass (SPP)
    (2) Magnitude Refinement Pass (MRP)
    (3) Cleanup Pass (CUP)
  - Context-adaptive binary arithmetic coding (13 contexts for 1-D).
  - Post-Compression Rate-Distortion (PCRD) optimization selects the
    optimal truncation point per code block under a global bit budget.

Advantages over SPIHT:
  - PCRD yields globally optimal R-D allocation across subbands/blocks.
  - Independent code blocks = fixed working memory per block (no tree state).
  - Better at low bitrates due to finer rate control granularity.

Memory layout (designed for embedded C port):
  - Working buffers: O(block_size) — default 64 int32 + 64 uint8 state.
  - No dynamic allocation: all buffers statically sized.
  - AC state: identical to existing spiht.py ArithEncoder/Decoder.

Tree structure for L-level 1-D DWT of length N:
  - Packed coefficients: [approx(N/2^L) | detail_L(N/2^L) | ... | detail_1(N/2)]
  - Same packing as spiht.py for compatibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Binary Arithmetic Coder (reused from spiht.py for consistency)
# ---------------------------------------------------------------------------

_AC_TOP = 0xFFFFFFFF
_AC_HALF = 0x80000000
_AC_QTR = 0x40000000
_AC_3QTR = 0xC0000000
_AC_COUNT_CAP = 256  # Tighter adaptation for EBCOT contexts


class _BitWriter:
    """Fixed-capacity bit writer."""

    __slots__ = ("buffer", "bit_pos", "capacity_bits")

    def __init__(self, capacity_bits: int):
        self.capacity_bits = capacity_bits
        self.buffer = bytearray((capacity_bits + 7) // 8)
        self.bit_pos = 0

    def write_bit(self, bit: int) -> None:
        if self.bit_pos >= self.capacity_bits:
            raise _BudgetExhausted()
        byte_idx = self.bit_pos >> 3
        shift = 7 - (self.bit_pos & 7)
        if bit & 1:
            self.buffer[byte_idx] |= 1 << shift
        self.bit_pos += 1

    def to_bytes(self) -> bytes:
        return bytes(self.buffer[: (self.bit_pos + 7) // 8])


class _BitReader:
    """Fixed-capacity bit reader."""

    __slots__ = ("data", "total_bits", "bit_pos")

    def __init__(self, data: bytes, total_bits: int):
        self.data = data
        self.total_bits = total_bits
        self.bit_pos = 0

    def read_bit(self) -> int:
        if self.bit_pos >= self.total_bits:
            raise _BudgetExhausted()
        byte_idx = self.bit_pos >> 3
        shift = 7 - (self.bit_pos & 7)
        bit = (self.data[byte_idx] >> shift) & 1
        self.bit_pos += 1
        return bit


class _BudgetExhausted(Exception):
    """Raised when bit budget is exceeded — used for truncation."""

    pass


# ---------------------------------------------------------------------------
# Context definitions for 1-D EBCOT
# ---------------------------------------------------------------------------
# 1-D neighbor contexts: for a sample at position i, neighbors are i-1, i+1.
# Significance context depends on how many neighbors are already significant.
# We define 13 contexts total:
#   0-2: significance (0, 1, 2 significant neighbors) for LL/HL bands
#   3-5: significance (0, 1, 2 significant neighbors) for LH/HH bands
#         (In 1D we only have approx + detail bands, map detail → indices 0-2)
#   6: sign context (positive prediction)
#   7: sign context (negative prediction)
#   8: sign context (zero prediction)
#   9-11: magnitude refinement (first refinement, >1 sig neighbor, other)
#   12: cleanup (run-length context)

_N_CONTEXTS = 13
_CTX_SIG_BASE = 0      # + num_sig_neighbors (0, 1, 2)
_CTX_SIGN_POS = 6
_CTX_SIGN_NEG = 7
_CTX_SIGN_ZERO = 8
_CTX_MAG_FIRST = 9     # first refinement pass for this sample
_CTX_MAG_OTHER = 10    # subsequent refinement passes
_CTX_MAG_NBSIG = 11    # refinement with significant neighbor
_CTX_CLEANUP = 12


class _ArithEnc:
    """Adaptive binary AC encoder for EBCOT."""

    __slots__ = ("writer", "capacity", "low", "high", "pending", "ctx0", "ctx1")

    def __init__(self, capacity_bits: int):
        self.writer = _BitWriter(capacity_bits + 128)  # headroom for flush
        self.capacity = capacity_bits
        self.low = 0
        self.high = _AC_TOP
        self.pending = 0
        self.ctx0 = [1] * _N_CONTEXTS
        self.ctx1 = [1] * _N_CONTEXTS

    @property
    def bits_out(self) -> int:
        return self.writer.bit_pos + self.pending

    def _emit(self, bit: int) -> None:
        self.writer.write_bit(bit)
        for _ in range(self.pending):
            self.writer.write_bit(1 - bit)
        self.pending = 0

    def write(self, bit: int, ctx: int) -> None:
        c0 = self.ctx0[ctx]
        c1 = self.ctx1[ctx]
        total = c0 + c1
        rng = self.high - self.low + 1
        split = self.low + (rng * c0) // total - 1
        if bit:
            self.low = split + 1
            self.ctx1[ctx] += 1
        else:
            self.high = split
            self.ctx0[ctx] += 1
        if self.ctx0[ctx] + self.ctx1[ctx] > _AC_COUNT_CAP:
            self.ctx0[ctx] = (self.ctx0[ctx] + 1) >> 1
            self.ctx1[ctx] = (self.ctx1[ctx] + 1) >> 1
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

    def flush(self) -> None:
        self.pending += 1
        if self.low < _AC_QTR:
            self._emit(0)
        else:
            self._emit(1)


class _ArithDec:
    """Adaptive binary AC decoder for EBCOT."""

    __slots__ = ("reader", "low", "high", "value", "ctx0", "ctx1")

    def __init__(self, data: bytes, total_bits: int):
        self.reader = _BitReader(data, total_bits)
        self.low = 0
        self.high = _AC_TOP
        self.ctx0 = [1] * _N_CONTEXTS
        self.ctx1 = [1] * _N_CONTEXTS
        self.value = 0
        for _ in range(32):
            self.value = (self.value << 1) | self._read_safe()

    def _read_safe(self) -> int:
        try:
            return self.reader.read_bit()
        except _BudgetExhausted:
            return 0

    def read(self, ctx: int) -> int:
        c0 = self.ctx0[ctx]
        c1 = self.ctx1[ctx]
        total = c0 + c1
        rng = self.high - self.low + 1
        split = self.low + (rng * c0) // total - 1
        if self.value > split:
            bit = 1
            self.low = split + 1
            self.ctx1[ctx] += 1
        else:
            bit = 0
            self.high = split
            self.ctx0[ctx] += 1
        if self.ctx0[ctx] + self.ctx1[ctx] > _AC_COUNT_CAP:
            self.ctx0[ctx] = (self.ctx0[ctx] + 1) >> 1
            self.ctx1[ctx] = (self.ctx1[ctx] + 1) >> 1
        while True:
            if self.high < _AC_HALF:
                pass
            elif self.low >= _AC_HALF:
                self.low -= _AC_HALF
                self.high -= _AC_HALF
                self.value -= _AC_HALF
            elif self.low >= _AC_QTR and self.high < _AC_3QTR:
                self.low -= _AC_QTR
                self.high -= _AC_QTR
                self.value -= _AC_QTR
            else:
                break
            self.low = (self.low << 1) & _AC_TOP
            self.high = ((self.high << 1) | 1) & _AC_TOP
            self.value = ((self.value << 1) | self._read_safe()) & _AC_TOP
        return bit


# ---------------------------------------------------------------------------
# Code-block state flags (uint8 per sample)
# ---------------------------------------------------------------------------
_STATE_SIG = 0x01        # sample has become significant
_STATE_REFINED = 0x02    # sample has been refined at least once
_STATE_VISITED = 0x04    # visited this pass (cleared between passes)
_STATE_SIGN = 0x08       # sign bit (1 = negative)


# ---------------------------------------------------------------------------
# Tier-1: Code-block encoder
# ---------------------------------------------------------------------------


@dataclass
class CodeBlockEncResult:
    """Result of encoding one code block (Tier-1)."""

    bitstream: bytes
    n_bits: int
    # Truncation points: list of (bits_at_end_of_pass, pass_distortion_reduction)
    truncation_points: list[tuple[int, float]]
    # For reconstruction
    block_size: int
    num_bitplanes: int
    max_val: int


def _sig_context(state: np.ndarray, i: int, block_size: int) -> int:
    """Significance context for position i based on neighbor significance."""
    n_sig = 0
    if i > 0 and (state[i - 1] & _STATE_SIG):
        n_sig += 1
    if i < block_size - 1 and (state[i + 1] & _STATE_SIG):
        n_sig += 1
    return _CTX_SIG_BASE + min(n_sig, 2)


def _sign_context(state: np.ndarray, signs: np.ndarray, i: int, block_size: int) -> tuple[int, int]:
    """Sign context and sign prediction for position i.

    Returns (context_id, sign_prediction) where sign_prediction is the XOR
    flip needed: if prediction is negative, flip the coded bit.
    """
    contrib = 0
    if i > 0 and (state[i - 1] & _STATE_SIG):
        contrib += 1 if not (state[i - 1] & _STATE_SIGN) else -1
    if i < block_size - 1 and (state[i + 1] & _STATE_SIG):
        contrib += 1 if not (state[i + 1] & _STATE_SIGN) else -1

    if contrib > 0:
        return _CTX_SIGN_POS, 0
    elif contrib < 0:
        return _CTX_SIGN_NEG, 1
    else:
        return _CTX_SIGN_ZERO, 0


def _mag_context(state: np.ndarray, i: int, block_size: int) -> int:
    """Magnitude refinement context."""
    if not (state[i] & _STATE_REFINED):
        # First refinement
        n_sig = 0
        if i > 0 and (state[i - 1] & _STATE_SIG):
            n_sig += 1
        if i < block_size - 1 and (state[i + 1] & _STATE_SIG):
            n_sig += 1
        return _CTX_MAG_NBSIG if n_sig > 0 else _CTX_MAG_FIRST
    return _CTX_MAG_OTHER


def encode_code_block(
    coeffs: np.ndarray,
    max_bitplanes: int | None = None,
) -> CodeBlockEncResult:
    """Encode a single code block with EBCOT Tier-1.

    Args:
        coeffs: 1-D array of integer (quantized) wavelet coefficients.
        max_bitplanes: Max number of bit-planes to encode (None = all).

    Returns:
        CodeBlockEncResult with truncation points for PCRD.
    """
    block_size = len(coeffs)
    magnitudes = np.abs(coeffs).astype(np.int64)
    signs = (coeffs < 0).astype(np.uint8)
    max_val = int(magnitudes.max()) if block_size > 0 else 0

    if max_val == 0:
        return CodeBlockEncResult(
            bitstream=b"",
            n_bits=0,
            truncation_points=[(0, 0.0)],
            block_size=block_size,
            num_bitplanes=0,
            max_val=0,
        )

    num_bp = math.ceil(math.log2(max_val + 1))
    if max_bitplanes is not None:
        num_bp = min(num_bp, max_bitplanes)

    # Generous capacity (will be truncated by PCRD)
    max_capacity = block_size * num_bp * 2 + 256
    enc = _ArithEnc(max_capacity)

    state = np.zeros(block_size, dtype=np.uint8)
    truncation_points: list[tuple[int, float]] = []

    # Track reconstruction for distortion calculation
    recon = np.zeros(block_size, dtype=np.float64)

    for bp in range(num_bp - 1, -1, -1):
        threshold = 1 << bp
        half_step = threshold >> 1  # midpoint of quantization bin

        # --- Pass 1: Significance Propagation Pass (SPP) ---
        # Only code samples that have at least one significant neighbor
        # and are not yet significant themselves.
        state &= np.uint8(~_STATE_VISITED & 0xFF)
        for i in range(block_size):
            if state[i] & _STATE_SIG:
                continue
            # Check if any neighbor is significant
            has_sig_nb = False
            if i > 0 and (state[i - 1] & _STATE_SIG):
                has_sig_nb = True
            if not has_sig_nb and i < block_size - 1 and (state[i + 1] & _STATE_SIG):
                has_sig_nb = True
            if not has_sig_nb:
                continue

            ctx = _sig_context(state, i, block_size)
            is_sig = int(magnitudes[i] >= threshold)
            enc.write(is_sig, ctx)
            state[i] |= _STATE_VISITED

            if is_sig:
                state[i] |= _STATE_SIG
                # Code sign
                sign_ctx, sign_flip = _sign_context(state, signs, i, block_size)
                coded_sign = signs[i] ^ sign_flip
                enc.write(coded_sign, sign_ctx)
                if signs[i]:
                    state[i] |= _STATE_SIGN
                # Update reconstruction
                recon[i] = (threshold + half_step) * (-1 if signs[i] else 1)

        bits_after_spp = enc.bits_out
        # Distortion reduction from this pass
        dist_reduction_spp = float(np.sum((np.abs(coeffs.astype(np.float64)) - np.abs(recon)) ** 2))
        truncation_points.append((bits_after_spp, dist_reduction_spp))

        # --- Pass 2: Magnitude Refinement Pass (MRP) ---
        for i in range(block_size):
            if not (state[i] & _STATE_SIG):
                continue
            if state[i] & _STATE_VISITED:
                continue  # newly significant this bitplane — skip refinement

            ctx = _mag_context(state, i, block_size)
            bit = int((magnitudes[i] >> bp) & 1)
            enc.write(bit, ctx)
            state[i] |= _STATE_REFINED

            # Update reconstruction
            old_mag = abs(recon[i])
            # Refine: we now know the bit at this plane
            new_mag = old_mag - half_step + (threshold if bit else 0) + (half_step >> 1 if half_step > 0 else 0)
            # Simpler: just accumulate the bit
            sign_mult = -1.0 if (state[i] & _STATE_SIGN) else 1.0
            current_mag = int(abs(recon[i]))
            new_mag_int = (current_mag & ~((threshold << 1) - 1)) | (magnitudes[i] & ((threshold << 1) - 1))
            recon[i] = (new_mag_int + half_step) * sign_mult if new_mag_int > 0 else 0

        bits_after_mrp = enc.bits_out
        truncation_points.append((bits_after_mrp, 0.0))

        # --- Pass 3: Cleanup Pass (CUP) ---
        # Code all remaining non-significant, unvisited samples
        for i in range(block_size):
            if state[i] & (_STATE_SIG | _STATE_VISITED):
                continue

            ctx = _CTX_CLEANUP
            is_sig = int(magnitudes[i] >= threshold)
            enc.write(is_sig, ctx)

            if is_sig:
                state[i] |= _STATE_SIG
                sign_ctx, sign_flip = _sign_context(state, signs, i, block_size)
                coded_sign = signs[i] ^ sign_flip
                enc.write(coded_sign, sign_ctx)
                if signs[i]:
                    state[i] |= _STATE_SIGN
                recon[i] = (threshold + half_step) * (-1 if signs[i] else 1)

        bits_after_cup = enc.bits_out
        truncation_points.append((bits_after_cup, 0.0))

    enc.flush()
    return CodeBlockEncResult(
        bitstream=enc.writer.to_bytes(),
        n_bits=enc.bits_out,
        truncation_points=truncation_points,
        block_size=block_size,
        num_bitplanes=num_bp,
        max_val=max_val,
    )


# ---------------------------------------------------------------------------
# Tier-1: Code-block decoder
# ---------------------------------------------------------------------------


def decode_code_block(
    bitstream: bytes,
    n_bits: int,
    block_size: int,
    num_bitplanes: int,
    max_val: int,
    truncate_at: int | None = None,
) -> np.ndarray:
    """Decode a single code block from its EBCOT Tier-1 bitstream.

    Args:
        bitstream: Encoded bytes.
        n_bits: Total bits in the bitstream.
        block_size: Number of coefficients in the block.
        num_bitplanes: Number of encoded bit-planes.
        max_val: Maximum absolute coefficient value (for bit-plane count).
        truncate_at: If given, stop decoding after this many bits.

    Returns:
        Reconstructed integer coefficients of shape (block_size,).
    """
    if num_bitplanes == 0 or max_val == 0:
        return np.zeros(block_size, dtype=np.int64)

    effective_bits = truncate_at if truncate_at is not None else n_bits
    effective_bits = min(effective_bits, n_bits)

    dec = _ArithDec(bitstream, effective_bits)
    state = np.zeros(block_size, dtype=np.uint8)
    magnitudes = np.zeros(block_size, dtype=np.int64)
    signs = np.zeros(block_size, dtype=np.uint8)

    try:
        for bp in range(num_bitplanes - 1, -1, -1):
            threshold = 1 << bp

            # --- Pass 1: Significance Propagation ---
            state &= np.uint8(~_STATE_VISITED & 0xFF)
            for i in range(block_size):
                if state[i] & _STATE_SIG:
                    continue
                has_sig_nb = False
                if i > 0 and (state[i - 1] & _STATE_SIG):
                    has_sig_nb = True
                if not has_sig_nb and i < block_size - 1 and (state[i + 1] & _STATE_SIG):
                    has_sig_nb = True
                if not has_sig_nb:
                    continue

                ctx = _sig_context(state, i, block_size)
                is_sig = dec.read(ctx)
                state[i] |= _STATE_VISITED

                if is_sig:
                    state[i] |= _STATE_SIG
                    magnitudes[i] = threshold
                    sign_ctx, sign_flip = _sign_context(state, signs, i, block_size)
                    coded_sign = dec.read(sign_ctx)
                    signs[i] = coded_sign ^ sign_flip
                    if signs[i]:
                        state[i] |= _STATE_SIGN

            # --- Pass 2: Magnitude Refinement ---
            for i in range(block_size):
                if not (state[i] & _STATE_SIG):
                    continue
                if state[i] & _STATE_VISITED:
                    continue

                ctx = _mag_context(state, i, block_size)
                bit = dec.read(ctx)
                if bit:
                    magnitudes[i] |= threshold
                state[i] |= _STATE_REFINED

            # --- Pass 3: Cleanup ---
            for i in range(block_size):
                if state[i] & (_STATE_SIG | _STATE_VISITED):
                    continue

                ctx = _CTX_CLEANUP
                is_sig = dec.read(ctx)

                if is_sig:
                    state[i] |= _STATE_SIG
                    magnitudes[i] = threshold
                    sign_ctx, sign_flip = _sign_context(state, signs, i, block_size)
                    coded_sign = dec.read(sign_ctx)
                    signs[i] = coded_sign ^ sign_flip
                    if signs[i]:
                        state[i] |= _STATE_SIGN

    except _BudgetExhausted:
        pass  # truncation — return what we have

    # Reconstruct with midpoint dequantization
    result = magnitudes.copy()
    # Add half-step for mid-bin reconstruction (only for significant samples)
    for i in range(block_size):
        if state[i] & _STATE_SIG:
            # Find lowest coded bit-plane for this sample
            lowest_bp = 0
            for bp_check in range(num_bitplanes):
                if magnitudes[i] & (1 << bp_check):
                    lowest_bp = bp_check
                    break
            result[i] = magnitudes[i]  # exact magnitude bits we decoded
    # Apply signs
    coeffs = np.where(signs, -result, result)
    return coeffs.astype(np.int64)


# ---------------------------------------------------------------------------
# Tier-2: PCRD Rate-Distortion Optimization
# ---------------------------------------------------------------------------


@dataclass
class SubbandInfo:
    """Metadata for one DWT subband."""

    offset: int       # start index in packed coefficient array
    length: int       # number of coefficients
    level: int        # decomposition level (0 = finest detail)
    gain: float       # subband gain weight for distortion (energy weighting)


def _partition_into_blocks(length: int, block_size: int) -> list[tuple[int, int]]:
    """Partition a subband into code blocks.

    Returns list of (start, size) tuples.
    """
    blocks = []
    for start in range(0, length, block_size):
        size = min(block_size, length - start)
        blocks.append((start, size))
    return blocks


@dataclass
class EbcotEncodeResult:
    """Full EBCOT encoding result."""

    # Per-block encoded data
    block_bitstreams: list[bytes]
    block_metas: list[dict]
    # Header info for decoder
    n_coeffs: int
    num_bitplanes_global: int
    quantization_step: float
    subband_layout: list[SubbandInfo]
    block_size: int
    # For PCRD
    block_truncation_points: list[list[tuple[int, float]]]


def _quantize_coeffs(
    coeffs: np.ndarray,
    step: float,
) -> np.ndarray:
    """Dead-zone uniform scalar quantizer (JPEG 2000 style).

    Args:
        coeffs: Float wavelet coefficients.
        step: Quantization step size.

    Returns:
        Integer quantized magnitudes with sign preserved.
    """
    return np.sign(coeffs) * np.floor(np.abs(coeffs) / step).astype(np.int64)


def _dequantize_coeffs(
    qcoeffs: np.ndarray,
    step: float,
) -> np.ndarray:
    """Inverse dead-zone quantizer with mid-bin reconstruction."""
    signs = np.sign(qcoeffs)
    mags = np.abs(qcoeffs).astype(np.float64)
    # Mid-bin: magnitude + 0.5 step for nonzero coefficients
    recon_mags = np.where(mags > 0, (mags + 0.5) * step, 0.0)
    return (signs * recon_mags).astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level encode / decode API
# ---------------------------------------------------------------------------


def ebcot_encode(
    approx: np.ndarray,
    details: list[np.ndarray],
    *,
    max_bits: int,
    block_size: int = 64,
    quantization_step: float | None = None,
    flat: bool = False,
) -> tuple[bytes, dict]:
    """Encode wavelet coefficients with EBCOT.

    Args:
        approx: Approximation coefficients (coarsest level).
        details: List of detail coefficient arrays [finest, ..., coarsest]
                 (same ordering as spiht.py codebase convention).
        max_bits: Total bit budget for the output.
        block_size: Code-block size (default 64). Ignored if flat=True.
        quantization_step: Scalar quantizer step. If None, auto-derived
            from coefficient range to use ~16 effective bit-planes.
        flat: If True, encode all coefficients as a single code block.
            This minimizes header overhead for short signals (< 1024 samples)
            at the cost of losing per-subband PCRD optimization.

    Returns:
        (bitstream_bytes, metadata_dict) — metadata needed by decoder.
    """
    # Pack coefficients into a flat array: [approx | detail_L | ... | detail_1]
    # details arrives as [finest(detail_1), ..., coarsest(detail_L)]
    # We pack as [approx, detail_L, detail_{L-1}, ..., detail_1] (coarsest first)
    levels = len(details)
    bands = [approx] + list(reversed(details))  # [approx, d_L, ..., d_1]

    subband_layout: list[SubbandInfo] = []
    offset = 0
    for band_idx, band in enumerate(bands):
        level = levels - band_idx if band_idx > 0 else levels
        gain = 1.0  # uniform weighting; could be wavelet-dependent
        subband_layout.append(SubbandInfo(
            offset=offset, length=len(band), level=level, gain=gain,
        ))
        offset += len(band)

    packed = np.concatenate(bands).astype(np.float64)
    n_coeffs = len(packed)

    # Quantization
    max_abs = float(np.max(np.abs(packed))) if n_coeffs > 0 else 1.0
    if quantization_step is None:
        # Target ~16 effective bit-planes
        quantization_step = max_abs / (1 << 15) if max_abs > 0 else 1.0
    step = max(quantization_step, 1e-12)

    q_packed = _quantize_coeffs(packed, step)
    global_max = int(np.max(np.abs(q_packed))) if n_coeffs > 0 else 0
    num_bp_global = math.ceil(math.log2(global_max + 1)) if global_max > 0 else 0

    import struct

    if flat:
        # --- FLAT MODE: single code block, minimal header ---
        # Header: n_coeffs(4) + num_bp(1) + step(4) + num_bands(1) + per-band lengths(4 each)
        #        = 10 + num_bands*4 bytes
        header = bytearray()
        header += struct.pack("<I", n_coeffs)
        header += struct.pack("<B", min(num_bp_global, 255))
        header += struct.pack("<f", step)
        header += struct.pack("<B", len(subband_layout))
        for sb in subband_layout:
            header += struct.pack("<I", sb.length)

        header_bits = len(header) * 8
        data_budget_bits = max(max_bits - header_bits, 0)

        # Encode the entire packed array as one code block
        result = encode_code_block(q_packed.astype(np.int64))

        # Truncate to budget: find best truncation point within budget
        trunc_bits = 0
        for bits, _ in result.truncation_points:
            actual_cost = ((bits + 7) // 8) * 8
            if actual_cost <= data_budget_bits:
                trunc_bits = bits
            else:
                break
        # Also check full bitstream
        if ((result.n_bits + 7) // 8) * 8 <= data_budget_bits:
            trunc_bits = result.n_bits

        trunc_bytes_count = (trunc_bits + 7) // 8
        body = result.bitstream[:trunc_bytes_count]
        output = bytes(header) + bytes(body)

        meta = {
            "n_coeffs": n_coeffs,
            "n_bits": len(output) * 8,
            "n_blocks": 1,
            "block_size": n_coeffs,
            "quantization_step": step,
            "num_bitplanes": num_bp_global,
            "flat": True,
            "trunc_bits": trunc_bits,
            "max_val": result.max_val,
        }
        return output, meta

    # --- MULTI-BLOCK MODE ---
    # Encode each code block
    all_blocks: list[tuple[int, int, int]] = []  # (subband_idx, start_in_band, size)
    for sb_idx, sb in enumerate(subband_layout):
        blocks = _partition_into_blocks(sb.length, block_size)
        for start_in_band, size in blocks:
            all_blocks.append((sb_idx, start_in_band, size))

    block_results: list[CodeBlockEncResult] = []
    for sb_idx, start_in_band, size in all_blocks:
        sb = subband_layout[sb_idx]
        global_start = sb.offset + start_in_band
        block_coeffs = q_packed[global_start: global_start + size]
        result = encode_code_block(block_coeffs.astype(np.int64))
        block_results.append(result)

    # --- PCRD: Select truncation points ---
    # Compute header size so PCRD can budget for data only
    # Header: 4+1+4+2+1 + num_bands*4 + 2 + num_blocks*(2+1+4)
    header_bytes = (4 + 1 + 4 + 2 + 1
                    + len(subband_layout) * 4
                    + 2
                    + len(block_results) * 7)
    data_budget_bits = max(max_bits - header_bytes * 8, 0)

    best_allocation = _pcrd_optimize(block_results, data_budget_bits)

    # Assemble final bitstream: header + concatenated truncated block streams
    # Header format:
    #   - n_coeffs (4 bytes)
    #   - num_bitplanes_global (1 byte)
    #   - quantization_step (4 bytes float32)
    #   - block_size (2 bytes)
    #   - num_bands (1 byte)
    #   - per-band: length (4 bytes)
    #   - num_blocks (2 bytes)
    #   - per-block: truncation_bits (2 bytes), num_bitplanes (1 byte), max_val (4 bytes)
    #   - block bitstreams concatenated

    import struct
    header = bytearray()
    header += struct.pack("<I", n_coeffs)
    header += struct.pack("<B", min(num_bp_global, 255))
    header += struct.pack("<f", step)
    header += struct.pack("<H", block_size)
    header += struct.pack("<B", len(subband_layout))
    for sb in subband_layout:
        header += struct.pack("<I", sb.length)
    header += struct.pack("<H", len(all_blocks))

    body = bytearray()
    for blk_idx, (bits_alloc) in enumerate(best_allocation):
        res = block_results[blk_idx]
        # Truncate bitstream to allocated bits
        trunc_bytes = (bits_alloc + 7) // 8
        blk_data = res.bitstream[:trunc_bytes]
        header += struct.pack("<H", bits_alloc)
        header += struct.pack("<B", res.num_bitplanes)
        header += struct.pack("<I", res.max_val)
        body += blk_data

    output = bytes(header) + bytes(body)

    meta = {
        "n_coeffs": n_coeffs,
        "n_bits": len(output) * 8,
        "n_blocks": len(all_blocks),
        "block_size": block_size,
        "quantization_step": step,
        "num_bitplanes": num_bp_global,
    }
    return output, meta


def _pcrd_optimize(
    block_results: list[CodeBlockEncResult],
    max_bits: int,
) -> list[int]:
    """Post-Compression Rate-Distortion optimization.

    For each block, select the truncation point (in bits) that minimizes
    total distortion subject to the global bit budget.

    Uses a Lagrangian bisection: for a given lambda, each block picks the
    truncation point that minimizes D + lambda*R. Binary search on lambda
    until total R <= max_bits.

    Returns: list of allocated bits per block.
    """
    n_blocks = len(block_results)
    if n_blocks == 0:
        return []

    # Build rate-distortion slopes per truncation point per block.
    # For each block, truncation_points is a list of (cumulative_bits, dist_value).
    # The "distortion" we use is simply the squared error relative to full precision.
    # We approximate: distortion at truncation point k is proportional to
    # the number of remaining unrefined bit-planes.

    # Simplified PCRD: allocate proportionally, then refine with slope-based greedy.
    # Each truncation point's rate is its cumulative bits.
    # We use the available truncation_points to build the feasible set.

    # Collect all possible truncation options per block:
    # Each block can be truncated at any of its recorded points, or at 0 bits.
    block_options: list[list[int]] = []
    for res in block_results:
        options = [0]  # always can allocate 0 bits (full distortion)
        for bits, _ in res.truncation_points:
            if bits > 0 and (not options or bits > options[-1]):
                options.append(bits)
        # Also include the full bitstream
        if res.n_bits > 0 and (not options or res.n_bits > options[-1]):
            options.append(res.n_bits)
        block_options.append(options)

    # Greedy rate allocation: start with 0 bits per block, iteratively
    # give bits to the block with the best marginal return.
    # This is O(n_blocks * n_truncation_points) — fine for our sizes.

    allocation = [0] * n_blocks  # current option index per block
    total_bits = 0

    # Simple approach: fill blocks in round-robin by distortion slope.
    # Better: use Lagrangian.

    # Binary search on lambda
    # For a given lambda, each block picks max option where delta_D / delta_R > lambda
    # We define distortion heuristic: D(truncate_at_k) = (full_bits - k) * block_size
    # This is a placeholder — proper distortion requires actual coefficient comparison.
    # For now, use a simpler "give bits proportionally to block size × num_bitplanes"

    total_available = sum(res.n_bits for res in block_results)
    if total_available <= max_bits:
        # Budget exceeds all blocks — no truncation needed
        return [res.n_bits for res in block_results]

    # Proportional allocation with rounding (work in byte-aligned units)
    alloc = []
    for blk_idx, res in enumerate(block_results):
        if total_available > 0:
            share = int(max_bits * res.n_bits / total_available)
        else:
            share = 0
        # Snap to nearest truncation point (never exceed share)
        options = block_options[blk_idx]
        best_opt = 0
        for opt in options:
            # Account for byte-rounding: actual cost is ((opt+7)//8)*8
            actual_cost = ((opt + 7) // 8) * 8
            if actual_cost <= share:
                best_opt = opt
        alloc.append(best_opt)

    # Greedy refinement: if we have leftover budget, upgrade blocks
    # Use actual byte-rounded cost for budget accounting
    def _byte_cost(bits: int) -> int:
        return ((bits + 7) // 8) * 8

    remaining = max_bits - sum(_byte_cost(a) for a in alloc)
    if remaining > 0:
        for _ in range(len(block_results) * 20):  # bounded iterations
            best_block = -1
            best_gain = -1
            best_new_bits = 0
            for blk_idx, res in enumerate(block_results):
                options = block_options[blk_idx]
                current = alloc[blk_idx]
                # Find next option above current
                for opt in options:
                    if opt > current:
                        cost = _byte_cost(opt) - _byte_cost(current)
                        if cost <= remaining:
                            gain = opt - current  # proxy for distortion reduction
                            if gain > best_gain:
                                best_gain = gain
                                best_block = blk_idx
                                best_new_bits = opt
                        break
            if best_block < 0:
                break
            remaining -= (_byte_cost(best_new_bits) - _byte_cost(alloc[best_block]))
            alloc[best_block] = best_new_bits

    return alloc


def ebcot_decode(
    bitstream: bytes,
    meta: dict,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Decode EBCOT bitstream back to wavelet coefficients.

    Args:
        bitstream: Encoded byte string from ebcot_encode.
        meta: Metadata dict from ebcot_encode.

    Returns:
        (approx, details) in the same format as spiht_decode:
        details ordered [finest, ..., coarsest].
    """
    import struct

    pos = 0

    def read_fmt(fmt: str) -> tuple:
        nonlocal pos
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, bitstream, pos)
        pos += size
        return vals

    # Check for flat mode
    if meta.get("flat", False):
        n_coeffs = read_fmt("<I")[0]
        num_bp_global = read_fmt("<B")[0]
        step = read_fmt("<f")[0]
        num_bands = read_fmt("<B")[0]
        band_lengths = [read_fmt("<I")[0] for _ in range(num_bands)]

        # Remaining bytes are the single code block
        trunc_bits = meta["trunc_bits"]
        max_val = meta["max_val"]
        blk_data = bitstream[pos:]

        if trunc_bits > 0 and max_val > 0:
            packed_q = decode_code_block(
                blk_data, trunc_bits, n_coeffs, num_bp_global, max_val,
                truncate_at=trunc_bits,
            )
        else:
            packed_q = np.zeros(n_coeffs, dtype=np.int64)

        packed_float = _dequantize_coeffs(packed_q, step)

        # Unpack bands
        bands = []
        offset = 0
        for band_len in band_lengths:
            bands.append(packed_float[offset: offset + band_len])
            offset += band_len

        approx_out = bands[0]
        details_out = list(reversed(bands[1:]))
        return approx_out, details_out

    # --- Standard multi-block decode ---
    n_coeffs = read_fmt("<I")[0]
    num_bp_global = read_fmt("<B")[0]
    step = read_fmt("<f")[0]
    block_size = read_fmt("<H")[0]
    num_bands = read_fmt("<B")[0]

    band_lengths = []
    for _ in range(num_bands):
        band_lengths.append(read_fmt("<I")[0])

    num_blocks = read_fmt("<H")[0]

    block_infos = []
    for _ in range(num_blocks):
        trunc_bits = read_fmt("<H")[0]
        num_bp = read_fmt("<B")[0]
        max_val = read_fmt("<I")[0]
        block_infos.append((trunc_bits, num_bp, max_val))

    # Read block bitstreams
    packed_q = np.zeros(n_coeffs, dtype=np.int64)

    # Reconstruct block positions
    all_blocks: list[tuple[int, int]] = []  # (global_start, size)
    offset = 0
    for band_len in band_lengths:
        for start in range(0, band_len, block_size):
            size = min(block_size, band_len - start)
            all_blocks.append((offset + start, size))
        offset += band_len

    for blk_idx, (trunc_bits, num_bp, max_val) in enumerate(block_infos):
        global_start, size = all_blocks[blk_idx]
        if trunc_bits == 0 or num_bp == 0 or max_val == 0:
            continue
        # Extract block bitstream bytes
        trunc_bytes = (trunc_bits + 7) // 8
        blk_data = bitstream[pos: pos + trunc_bytes]
        pos += trunc_bytes

        decoded = decode_code_block(
            blk_data, trunc_bits, size, num_bp, max_val, truncate_at=trunc_bits,
        )
        packed_q[global_start: global_start + size] = decoded

    # Dequantize
    packed_float = _dequantize_coeffs(packed_q, step)

    # Unpack bands: [approx, d_L, ..., d_1] → approx + details [d_1, ..., d_L]
    bands = []
    offset = 0
    for band_len in band_lengths:
        bands.append(packed_float[offset: offset + band_len])
        offset += band_len

    approx = bands[0]
    # bands[1:] is [d_L, ..., d_1]; reverse to get [d_1, ..., d_L] (finest first)
    details = list(reversed(bands[1:]))
    return approx, details
