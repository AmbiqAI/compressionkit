"""Shared types and constants for compressionkit datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LicenseTier = Literal["open", "restricted"]


@dataclass
class DatasetInfo:
    """Metadata about a dataset.

    Attributes:
        name: Short identifier (e.g. ``"ptbxl"``).
        sampling_rate: Native sample rate in Hz.
        num_leads: Number of signal leads/channels.
        description: Human-readable description.
        license: License identifier or URL.
        license_tier: ``"open"`` (sample data may be redistributed) or
            ``"restricted"`` (do not redistribute, user must obtain).
        requires_agreement: If ``True``, user must agree to terms before
            downloading (e.g. MESA from NSRR).
    """

    name: str
    sampling_rate: int
    num_leads: int
    description: str = ""
    license: str = ""
    license_tier: LicenseTier = "open"
    requires_agreement: bool = False
