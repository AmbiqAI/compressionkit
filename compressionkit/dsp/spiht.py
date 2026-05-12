"""SPIHT scaffolding for wavelet-based compression.

This module provides fixed-buffer bitstream helpers and a placeholder API
for SPIHT encode/decode. Full SPIHT is non-trivial; the functions below are
structured to be extended without changing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        return bytes(self.buffer)


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


def spiht_encode(coeffs, max_bits: int | None = None) -> tuple[bytes, dict[str, int]]:
    """Placeholder SPIHT encoder. Returns bitstream and metadata."""
    raise NotImplementedError("SPIHT encode not implemented yet. Add wavelet tree traversal here.")


def spiht_decode(bitstream: bytes, metadata: dict[str, int]):
    """Placeholder SPIHT decoder. Returns reconstructed coefficients."""
    raise NotImplementedError("SPIHT decode not implemented yet. Add inverse tree traversal here.")


__all__ = [
    "BitReader",
    "BitWriter",
    "spiht_decode",
    "spiht_encode",
]
