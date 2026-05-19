"""1-D SPIHT encoder/decoder for wavelet-based signal compression.

Implements the Set Partitioning in Hierarchical Trees algorithm
(Said & Pearlman, 1996) adapted for 1-D signals. The algorithm encodes
wavelet coefficients progressively by significance (bit-plane order,
MSB first), achieving near-optimal R-D performance.

The bitstream is truncatable: stopping at any point gives the best
possible reconstruction for that bit budget. This makes it ideal for
rate-distortion comparisons.

Tree structure for L-level 1-D DWT of length N:
  - Packed coefficients: [approx(N/2^L) | detail_L(N/2^L) | ... | detail_1(N/2)]
  - Root nodes: indices 0..N/2^L - 1 (approx band)
  - Parent at index i in band l has children at 2i, 2i+1 in band l-1
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BitWriter:
    """Simple fixed-size bit writer."""

    capacity_bits: int

    def __post_init__(self):
        byte_len = (self.capacity_bits + 7) // 8
        self.buffer = bytearray(byte_len)
        self.bit_pos = 0

    def write_bit(self, bit: int) -> None:
        if self.bit_pos >= self.capacity_bits:
            raise RuntimeError("BitWriter capacity exceeded")
        byte_idx = self.bit_pos // 8
        shift = 7 - (self.bit_pos % 8)
        if bit & 1:
            self.buffer[byte_idx] |= 1 << shift
        self.bit_pos += 1

    def write_bits(self, value: int, num_bits: int) -> None:
        for i in range(num_bits - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def to_bytes(self) -> bytes:
        return bytes(self.buffer[: (self.bit_pos + 7) // 8])


@dataclass
class BitReader:
    """Simple fixed-size bit reader."""

    data: bytes
    total_bits: int

    def __post_init__(self):
        self.bit_pos = 0

    def read_bit(self) -> int:
        if self.bit_pos >= self.total_bits:
            raise RuntimeError("BitReader out of data")
        byte_idx = self.bit_pos // 8
        shift = 7 - (self.bit_pos % 8)
        bit = (self.data[byte_idx] >> shift) & 1
        self.bit_pos += 1
        return bit

    def read_bits(self, num_bits: int) -> int:
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | self.read_bit()
        return value


# ---------------------------------------------------------------------------
# Binary arithmetic (range) coder with adaptive per-context probabilities.
#
# Fixed 32-bit state, two integer counters per context — chosen for direct
# portability to embedded C (no malloc, no floats). Counters are halved when
# their sum exceeds `_AC_COUNT_CAP` so the coder adapts to non-stationary
# signal statistics (e.g. between bit-planes).
# ---------------------------------------------------------------------------

_AC_TOP = 0xFFFFFFFF
_AC_HALF = 0x80000000
_AC_QTR = 0x40000000
_AC_3QTR = 0xC0000000
_AC_COUNT_CAP = 4096
_AC_NUM_CONTEXTS = 6  # ctx ids defined below

# Context IDs
CTX_LIP_SIG = 0  # significance of an LIP entry
CTX_LIS_A_SIG = 1  # descendant-tree significance (LIS type A)
CTX_LIS_B_SIG = 2  # grandchild-tree significance (LIS type B)
CTX_CHILD_SIG = 3  # significance of a child under a significant LIS-A
CTX_SIGN = 4  # sign bit (after significance)
CTX_REFINE = 5  # refinement-pass bit


class ArithEncoder:
    """Adaptive binary arithmetic encoder backed by a `BitWriter`.

    Budget is enforced on emitted output bits, not on symbol count.
    """

    def __init__(self, capacity_bits: int, n_contexts: int = _AC_NUM_CONTEXTS):
        # Small overshoot capacity to allow renormalization spill before truncation.
        self.writer = BitWriter(capacity_bits=capacity_bits + 64)
        self.capacity_bits = capacity_bits
        self.low = 0
        self.high = _AC_TOP
        self.pending = 0
        self.ctx0 = [1] * n_contexts
        self.ctx1 = [1] * n_contexts

    @property
    def bits_out(self) -> int:
        # Effective output bits = emitted + pending (worst-case still-to-emit).
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
            self.ctx1[ctx] = c1 + 1
        else:
            self.high = split
            self.ctx0[ctx] = c0 + 1
        if self.ctx0[ctx] + self.ctx1[ctx] > _AC_COUNT_CAP:
            # Halve (rounded up) to retain adaptivity without overflow.
            self.ctx0[ctx] = (self.ctx0[ctx] + 1) >> 1
            self.ctx1[ctx] = (self.ctx1[ctx] + 1) >> 1
        # Renormalize
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
        """Output the final two bits of state."""
        self.pending += 1
        if self.low < _AC_QTR:
            self._emit(0)
        else:
            self._emit(1)


class ArithDecoder:
    """Adaptive binary arithmetic decoder mirroring `ArithEncoder`."""

    def __init__(self, data: bytes, total_bits: int, n_contexts: int = _AC_NUM_CONTEXTS):
        self.reader = BitReader(data=data, total_bits=total_bits)
        self.low = 0
        self.high = _AC_TOP
        self.code = 0
        self.ctx0 = [1] * n_contexts
        self.ctx1 = [1] * n_contexts
        # Prime 32-bit code register; zero-pad past EOF.
        for _ in range(32):
            self.code = (self.code << 1) | self._read_input()

    def _read_input(self) -> int:
        if self.reader.bit_pos >= self.reader.total_bits:
            return 0
        return self.reader.read_bit()

    def read(self, ctx: int) -> int:
        c0 = self.ctx0[ctx]
        c1 = self.ctx1[ctx]
        total = c0 + c1
        rng = self.high - self.low + 1
        split = self.low + (rng * c0) // total - 1
        if self.code <= split:
            bit = 0
            self.high = split
            self.ctx0[ctx] = c0 + 1
        else:
            bit = 1
            self.low = split + 1
            self.ctx1[ctx] = c1 + 1
        if self.ctx0[ctx] + self.ctx1[ctx] > _AC_COUNT_CAP:
            self.ctx0[ctx] = (self.ctx0[ctx] + 1) >> 1
            self.ctx1[ctx] = (self.ctx1[ctx] + 1) >> 1
        # Renormalize
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
        return bit


# ---------------------------------------------------------------------------
# Sink / Source adapters: unify raw-bit and AC paths so the SPIHT main loop
# can be written once.
# ---------------------------------------------------------------------------


class _RawSink:
    """Bit sink that writes raw bits (ignores context)."""

    def __init__(self, capacity_bits: int):
        self.writer = BitWriter(capacity_bits=capacity_bits)
        self.capacity_bits = capacity_bits

    @property
    def bits_out(self) -> int:
        return self.writer.bit_pos

    def write(self, bit: int, ctx: int) -> None:
        self.writer.write_bit(bit)

    def to_bytes(self) -> bytes:
        return self.writer.to_bytes()


class _AcSink:
    """Bit sink wrapping `ArithEncoder` and enforcing the bit budget."""

    def __init__(self, capacity_bits: int):
        self.encoder = ArithEncoder(capacity_bits=capacity_bits)
        self.capacity_bits = capacity_bits

    @property
    def bits_out(self) -> int:
        return self.encoder.bits_out

    def write(self, bit: int, ctx: int) -> None:
        self.encoder.write(bit, ctx)

    def to_bytes(self) -> bytes:
        self.encoder.flush()
        return self.encoder.writer.to_bytes()


class _RawSource:
    """Bit source reading raw bits."""

    def __init__(self, data: bytes, total_bits: int):
        self.reader = BitReader(data=data, total_bits=total_bits)

    def read(self, ctx: int) -> int:
        return self.reader.read_bit()


class _AcSource:
    """Bit source wrapping `ArithDecoder`."""

    def __init__(self, data: bytes, total_bits: int):
        self.decoder = ArithDecoder(data=data, total_bits=total_bits)

    def read(self, ctx: int) -> int:
        return self.decoder.read(ctx)


class _LoggingAcSink:
    """AC sink that also records every emission for offline prior training.

    Records, for each emitted symbol:
        ctx_id:    int8     — symbol class (CTX_* constants)
        bit:       int8     — emitted bit (0/1)
        ac_p1:     float32  — probability the baseline AC assigned to bit=1
                              *before* this emission (its current ctx1/(ctx0+ctx1))
        bitplane:  int8     — current SPIHT bit-plane index (n_start - pass_idx)

    These are sufficient to compute the baseline AC's per-symbol cross-entropy
    (`-log2(p_used)` where `p_used = ac_p1 if bit else 1-ac_p1`) and to train
    a learned prior that conditions on richer features supplied externally.
    """

    def __init__(self, capacity_bits: int):
        self.encoder = ArithEncoder(capacity_bits=capacity_bits)
        self.capacity_bits = capacity_bits
        self.ctx_log: list[int] = []
        self.bit_log: list[int] = []
        self.ac_p1_log: list[float] = []
        self.bitplane_log: list[int] = []
        self._cur_bitplane: int = 0

    def set_bitplane(self, n: int) -> None:
        self._cur_bitplane = n

    @property
    def bits_out(self) -> int:
        return self.encoder.bits_out

    def write(self, bit: int, ctx: int) -> None:
        c0 = self.encoder.ctx0[ctx]
        c1 = self.encoder.ctx1[ctx]
        p1 = c1 / (c0 + c1)
        self.ctx_log.append(ctx)
        self.bit_log.append(int(bit))
        self.ac_p1_log.append(float(p1))
        self.bitplane_log.append(self._cur_bitplane)
        self.encoder.write(bit, ctx)

    def to_bytes(self) -> bytes:
        self.encoder.flush()
        return self.encoder.writer.to_bytes()

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "ctx": np.asarray(self.ctx_log, dtype=np.int8),
            "bit": np.asarray(self.bit_log, dtype=np.int8),
            "ac_p1": np.asarray(self.ac_p1_log, dtype=np.float32),
            "bitplane": np.asarray(self.bitplane_log, dtype=np.int8),
        }


# ---------------------------------------------------------------------------
# 1-D SPIHT tree helpers
# ---------------------------------------------------------------------------


def _pack_coefficients(approx: np.ndarray, details: list[np.ndarray]) -> np.ndarray:
    """Pack DWT coefficients into a flat array with tree structure.

    Layout: [approx | detail_coarsest | ... | detail_finest]
    Each band has length N/2^L for approx, N/2^L for detail_L, ..., N/2 for detail_1.

    Note: The codebase convention stores details as [finest→coarsest],
    so we reverse to get the tree-compatible [coarsest→finest] order.
    """
    bands = [approx, *list(reversed(details))]
    return np.concatenate(bands).astype(np.float64)


def _unpack_coefficients(packed: np.ndarray, approx_len: int, num_levels: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Unpack flat coefficient array back to approx + detail bands.

    Returns details in [finest→coarsest] order to match codebase convention.
    """
    approx = packed[:approx_len].astype(np.float32)
    details = []
    offset = approx_len
    band_len = approx_len  # coarsest detail same size as approx
    for _ in range(num_levels):
        details.append(packed[offset : offset + band_len].astype(np.float32))
        offset += band_len
        band_len *= 2
    # Reverse: internal [coarsest→finest] → codebase [finest→coarsest]
    details.reverse()
    return approx, details


def _band_offsets(approx_len: int, num_levels: int) -> list[int]:
    """Return starting offsets of each band in packed array."""
    offsets = [0]  # approx
    offset = approx_len
    band_len = approx_len
    for _ in range(num_levels):
        offsets.append(offset)
        offset += band_len
        band_len *= 2
    return offsets


def _children(idx: int, total_len: int, approx_len: int, num_levels: int) -> list[int]:
    """Get child indices for a node in the 1-D wavelet tree.

    Tree structure (packed order: [approx | det_coarsest | ... | det_finest]):
      - Band 0 (approx, indices 0..approx_len-1): NO children (coded directly).
      - Band 1 (coarsest detail, indices approx_len..2*approx_len-1): tree roots.
        Position p in band 1 → children at band 2 positions 2p, 2p+1.
      - Band k (k>=1, k<num_levels): children at band k+1 positions 2p, 2p+1.
      - Last band (finest detail): NO children (leaves).
    """
    offsets = _band_offsets(approx_len, num_levels)

    # Find which band this index belongs to
    band_idx = -1
    for b in range(len(offsets) - 1, -1, -1):
        if idx >= offsets[b]:
            band_idx = b
            break

    if band_idx < 0:
        return []

    # Band 0 (approx) has no children
    if band_idx == 0:
        return []

    # Last band (finest detail) has no children
    if band_idx >= num_levels:
        return []

    # Position within this band
    pos_in_band = idx - offsets[band_idx]

    # Children are in the next band (band_idx + 1)
    next_offset = offsets[band_idx + 1]
    child0 = next_offset + 2 * pos_in_band
    child1 = next_offset + 2 * pos_in_band + 1

    children = []
    if child0 < total_len:
        children.append(child0)
    if child1 < total_len:
        children.append(child1)
    return children


def _descendants(idx: int, total_len: int, approx_len: int, num_levels: int) -> list[int]:
    """Get all descendants (children, grandchildren, etc.) of a node."""
    result = []
    stack = _children(idx, total_len, approx_len, num_levels)
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(_children(node, total_len, approx_len, num_levels))
    return result


def _max_descendant(idx: int, coeffs: np.ndarray, approx_len: int, num_levels: int) -> float:
    """Max absolute value among all descendants of idx."""
    desc = _descendants(idx, len(coeffs), approx_len, num_levels)
    if not desc:
        return 0.0
    return float(np.max(np.abs(coeffs[desc])))


# ---------------------------------------------------------------------------
# SPIHT Encoder
# ---------------------------------------------------------------------------


def spiht_encode(
    approx: np.ndarray,
    details: list[np.ndarray],
    max_bits: int,
    use_ac: bool = False,
    log_emissions: bool = False,
) -> tuple[bytes, dict]:
    """SPIHT encode wavelet coefficients to a fixed bit budget.

    Args:
        approx: Approximation coefficients from DWT.
        details: List of detail bands [coarsest → finest].
        max_bits: Maximum bits for the bitstream (counts output bits, which
            equals symbol count for the raw path and coded-bit count for AC).
        use_ac: When True, route every emitted bit through an adaptive binary
            arithmetic coder with per-context probability models. The budget
            still bounds output bits, so AC trades the same number of bits for
            more SPIHT iterations (better fidelity).
        log_emissions: When True (only valid with use_ac=True), the encoder
            records every emitted symbol — ctx id, bit value, baseline-AC
            probability, and bit-plane index — into the returned metadata
            under the key ``emissions``. Used for offline training of neural
            entropy priors. Adds modest memory cost; bitstream is unchanged.

    Returns:
        (bitstream_bytes, metadata) where metadata contains info
        needed for decoding (approx_len, num_levels, max_coeff, n_bits,
        n_symbols, use_ac).
    """
    num_levels = len(details)
    approx_len = len(approx)
    coeffs = _pack_coefficients(approx, details)
    total_len = len(coeffs)

    # Initial threshold: largest power of 2 <= max(|coeffs|)
    max_coeff = float(np.max(np.abs(coeffs)))
    if max_coeff < 1e-12:
        # All zeros
        return b"", {
            "approx_len": approx_len,
            "num_levels": num_levels,
            "max_coeff": 0.0,
            "n_bits": 0,
            "n_symbols": 0,
            "n_start": 0,
            "total_len": total_len,
            "use_ac": bool(use_ac),
        }

    n_start = math.floor(math.log2(max_coeff))
    threshold = 2.0**n_start

    sink: _RawSink | _AcSink | _LoggingAcSink
    if log_emissions:
        if not use_ac:
            raise ValueError("log_emissions=True requires use_ac=True")
        sink = _LoggingAcSink(max_bits)
    elif use_ac:
        sink = _AcSink(max_bits)
    else:
        sink = _RawSink(max_bits)

    # Initialize LIP (List of Insignificant Pixels) with root nodes
    LIP = list(range(2 * approx_len))  # approx + coarsest detail
    LIS: list[tuple[int, str]] = []
    for i in range(approx_len, 2 * approx_len):
        if _children(i, total_len, approx_len, num_levels):
            LIS.append((i, "A"))
    LSP: list[int] = []

    n_symbols = 0

    def _emit(bit: int, ctx: int) -> None:
        nonlocal n_symbols
        sink.write(bit, ctx)
        n_symbols += 1
        if sink.bits_out >= max_bits:
            raise _BudgetExhausted()

    try:
        while True:
            lsp_before_sort = len(LSP)
            if log_emissions:
                # Bit-plane index = log2(threshold); record for the upcoming pass.
                sink.set_bitplane(round(math.log2(threshold)))  # type: ignore[union-attr]

            # --- Sorting pass: LIP ---
            new_lip = []
            for i in LIP:
                sig = 1 if abs(coeffs[i]) >= threshold else 0
                _emit(sig, CTX_LIP_SIG)
                if sig:
                    sign_bit = 0 if coeffs[i] >= 0 else 1
                    _emit(sign_bit, CTX_SIGN)
                    LSP.append(i)
                else:
                    new_lip.append(i)
            LIP = new_lip

            # --- Sorting pass: LIS ---
            new_lis: list[tuple[int, str]] = []
            lis_idx = 0
            while lis_idx < len(LIS):
                idx, set_type = LIS[lis_idx]
                lis_idx += 1

                if set_type == "A":
                    max_desc = _max_descendant(idx, coeffs, approx_len, num_levels)
                    sig = 1 if max_desc >= threshold else 0
                    _emit(sig, CTX_LIS_A_SIG)
                    if sig:
                        kids = _children(idx, total_len, approx_len, num_levels)
                        for k in kids:
                            child_sig = 1 if abs(coeffs[k]) >= threshold else 0
                            _emit(child_sig, CTX_CHILD_SIG)
                            if child_sig:
                                sign_bit = 0 if coeffs[k] >= 0 else 1
                                _emit(sign_bit, CTX_SIGN)
                                LSP.append(k)
                            else:
                                LIP.append(k)
                        grandkids = []
                        for k in kids:
                            grandkids.extend(_children(k, total_len, approx_len, num_levels))
                        if grandkids:
                            LIS.append((idx, "B"))
                    else:
                        new_lis.append((idx, "A"))

                elif set_type == "B":
                    kids = _children(idx, total_len, approx_len, num_levels)
                    max_gc = 0.0
                    for k in kids:
                        max_gc = max(
                            max_gc,
                            _max_descendant(k, coeffs, approx_len, num_levels),
                        )
                    sig = 1 if max_gc >= threshold else 0
                    _emit(sig, CTX_LIS_B_SIG)
                    if sig:
                        for k in kids:
                            LIS.append((k, "A"))
                    else:
                        new_lis.append((idx, "B"))

            LIS = new_lis + list(LIS[lis_idx:])

            # --- Refinement pass ---
            for i in LSP[:lsp_before_sort]:
                bit = 1 if abs(coeffs[i]) % (2 * threshold) >= threshold else 0
                _emit(bit, CTX_REFINE)

            threshold /= 2.0

    except _BudgetExhausted:
        pass

    metadata = {
        "approx_len": approx_len,
        "num_levels": num_levels,
        "max_coeff": max_coeff,
        "n_start": n_start,
        "n_bits": sink.bits_out,
        "n_symbols": n_symbols,
        "total_len": total_len,
        "use_ac": bool(use_ac),
    }
    if log_emissions:
        metadata["emissions"] = sink.to_arrays()  # type: ignore[union-attr]
    return sink.to_bytes(), metadata


class _BudgetExhausted(Exception):
    """Internal signal that bit budget is exhausted."""


# ---------------------------------------------------------------------------
# SPIHT Decoder
# ---------------------------------------------------------------------------


def spiht_decode(bitstream: bytes, metadata: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    """SPIHT decode a bitstream back to wavelet coefficients."""
    approx_len = metadata["approx_len"]
    num_levels = metadata["num_levels"]
    max_coeff = metadata["max_coeff"]
    n_start = metadata["n_start"]
    n_bits = metadata["n_bits"]
    total_len = metadata["total_len"]
    use_ac = bool(metadata.get("use_ac", False))
    n_symbols = int(metadata.get("n_symbols", n_bits))

    if max_coeff < 1e-12 or n_bits == 0:
        approx = np.zeros(approx_len, dtype=np.float32)
        details = []
        band_len = approx_len
        for _ in range(num_levels):
            details.append(np.zeros(band_len, dtype=np.float32))
            band_len *= 2
        details.reverse()
        return approx, details

    source: _RawSource | _AcSource
    if use_ac:
        # AC bytes may extend slightly past n_bits due to renormalization spill;
        # cap reads at the byte-aligned total.
        source = _AcSource(data=bitstream, total_bits=len(bitstream) * 8)
    else:
        source = _RawSource(data=bitstream, total_bits=n_bits)

    coeffs = np.zeros(total_len, dtype=np.float64)
    threshold = 2.0**n_start

    LIP = list(range(2 * approx_len))
    LIS: list[tuple[int, str]] = []
    for i in range(approx_len, 2 * approx_len):
        if _children(i, total_len, approx_len, num_levels):
            LIS.append((i, "A"))
    LSP: list[int] = []

    symbols_read = 0
    budget = n_symbols if use_ac else n_bits

    def _read(ctx: int) -> int:
        nonlocal symbols_read
        if symbols_read >= budget:
            raise _BudgetExhausted()
        bit = source.read(ctx)
        symbols_read += 1
        return bit

    try:
        while True:
            lsp_before_sort = len(LSP)

            # --- Sorting pass: LIP ---
            new_lip = []
            for i in LIP:
                sig = _read(CTX_LIP_SIG)
                if sig:
                    sign_bit = _read(CTX_SIGN)
                    coeffs[i] = -threshold * 1.5 if sign_bit else threshold * 1.5
                    LSP.append(i)
                else:
                    new_lip.append(i)
            LIP = new_lip

            # --- Sorting pass: LIS ---
            new_lis: list[tuple[int, str]] = []
            lis_idx = 0
            while lis_idx < len(LIS):
                idx, set_type = LIS[lis_idx]
                lis_idx += 1

                if set_type == "A":
                    sig = _read(CTX_LIS_A_SIG)
                    if sig:
                        kids = _children(idx, total_len, approx_len, num_levels)
                        for k in kids:
                            child_sig = _read(CTX_CHILD_SIG)
                            if child_sig:
                                sign_bit = _read(CTX_SIGN)
                                coeffs[k] = -threshold * 1.5 if sign_bit else threshold * 1.5
                                LSP.append(k)
                            else:
                                LIP.append(k)
                        grandkids = []
                        for k in kids:
                            grandkids.extend(_children(k, total_len, approx_len, num_levels))
                        if grandkids:
                            LIS.append((idx, "B"))
                    else:
                        new_lis.append((idx, "A"))

                elif set_type == "B":
                    sig = _read(CTX_LIS_B_SIG)
                    if sig:
                        kids = _children(idx, total_len, approx_len, num_levels)
                        for k in kids:
                            LIS.append((k, "A"))
                    else:
                        new_lis.append((idx, "B"))

            LIS = new_lis + list(LIS[lis_idx:])

            # --- Refinement pass ---
            for i in LSP[:lsp_before_sort]:
                bit = _read(CTX_REFINE)
                adjustment = threshold / 2
                if bit:
                    coeffs[i] += adjustment if coeffs[i] > 0 else -adjustment
                else:
                    coeffs[i] -= adjustment if coeffs[i] > 0 else -adjustment

            threshold /= 2.0

    except (_BudgetExhausted, RuntimeError):
        # Budget reached or bit source exhausted — both are normal termination.
        pass

    return _unpack_coefficients(coeffs, approx_len, num_levels)


__all__ = [
    "ArithDecoder",
    "ArithEncoder",
    "BitReader",
    "BitWriter",
    "spiht_decode",
    "spiht_encode",
]
