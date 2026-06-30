"""Operating *lanes* — coarse tradeoff archetypes for the compression playbook.

A lane is a deliberate operating point on the noise / faithfulness /
compression-ratio / compute-placement tradeoff. There is no single best
codec for wearable physiological signals; instead the playbook offers a
small fixed set of lanes, each scored on the same metrics, so users can
read the tradeoff and pick the lane that matches their constraints.

Champions within a lane are the candidates promoted to golden runs and
HuggingFace releases.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Lane(StrEnum):
    """Coarse operating-point archetypes."""

    FAITHFUL = "faithful"
    CLEAN = "clean"
    COMPACT_LEARNED = "compact_learned"
    LONG_CONTEXT = "long_context"


@dataclass(frozen=True)
class LaneInfo:
    """Human-readable description of a lane."""

    lane: Lane
    title: str
    use_case: str
    faithfulness: str
    typical_cr: str
    compute: str


LANE_INFO: dict[Lane, LaneInfo] = {
    Lane.FAITHFUL: LaneInfo(
        lane=Lane.FAITHFUL,
        title="Faithful / near-lossless",
        use_case="Archive & diagnostic; must not lose morphology.",
        faithfulness="Preserve everything (signal + noise)",
        typical_cr="2-4x",
        compute="compact encoder, light decoder",
    ),
    Lane.CLEAN: LaneInfo(
        lane=Lane.CLEAN,
        title="Clean (denoise-then-code)",
        use_case="Monitoring; remove noise but never invent structure.",
        faithfulness="Denoise, no hallucination",
        typical_cr="4-16x",
        compute="compact encoder, light decoder",
    ),
    Lane.COMPACT_LEARNED: LaneInfo(
        lane=Lane.COMPACT_LEARNED,
        title="Compact-learned (RVQ)",
        use_case="Stream a wearable to a phone at a high ratio.",
        faithfulness="Perceptual fidelity",
        typical_cr="8-32x",
        compute="compact encoder, heavy decoder",
    ),
    Lane.LONG_CONTEXT: LaneInfo(
        lane=Lane.LONG_CONTEXT,
        title="Long-context / streaming",
        use_case="Continuous streams; exploit cross-window redundancy.",
        faithfulness="Perceptual fidelity",
        typical_cr="16x+",
        compute="compact encoder, heavy decoder on phone/cloud",
    ),
}


__all__ = ["LANE_INFO", "Lane", "LaneInfo"]
