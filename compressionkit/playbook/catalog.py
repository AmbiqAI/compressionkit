"""Method catalog — browsable *cards* over the reusable building blocks.

Each :class:`MethodCard` is a thin, declarative description of a compression
technique: which lane it serves, what tradeoff it makes, whether it ships
today, and (for self-contained codecs) how to build it. The catalog is the
funnel from "showcase of what is possible" to "winning solutions promoted to
golden + HuggingFace".

Two sources feed the catalog:

* **Explicit cards** registered here — classical DSP codecs that build without
  any trained weights, plus *planned*/*experimental* entries that advertise a
  technique even before it ships.
* **Golden hooks** — every entry in
  :data:`compressionkit.experiments.registry.GOLDEN_REGISTRY` is surfaced as a
  shipped card automatically, so the release surface and the playbook never
  drift apart.

The card intentionally does not own any algorithm logic; ``builder`` returns a
:class:`~compressionkit.evaluation.codec.Codec` assembled from the existing
``evaluation`` / ``dsp`` / ``models`` building blocks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from compressionkit.evaluation.codec import Codec
from compressionkit.playbook.lanes import Lane


class Faithfulness(StrEnum):
    """How a method treats noise in the input."""

    FAITHFUL = "faithful"
    DENOISE = "denoise"
    PERCEPTUAL = "perceptual"


class Status(StrEnum):
    """Maturity of a catalog entry."""

    SHIPPED = "shipped"
    EXPERIMENTAL = "experimental"
    PLANNED = "planned"


class Tier(StrEnum):
    """Curation tier within a lane.

    * ``BASELINE`` — a simple, well-understood reference point for the lane.
    * ``ROBUST`` — a recommended, justified method we stand behind.
    * ``EXPERIMENTAL`` — novel / in-progress; not yet a recommendation.
    """

    BASELINE = "baseline"
    ROBUST = "robust"
    EXPERIMENTAL = "experimental"


# Sort order for display (baseline first, then robust, then experimental).
TIER_ORDER: dict[Tier, int] = {Tier.BASELINE: 0, Tier.ROBUST: 1, Tier.EXPERIMENTAL: 2}


# Builder signature: keyword-only construction of a Codec for a given operating
# point. Self-contained (classical) methods provide one; trained methods leave
# it ``None`` and are loaded from a golden run directory instead.
CodecBuilder = Callable[..., Codec]


@dataclass(frozen=True)
class MethodCard:
    """Declarative description of one compression technique."""

    id: str
    display_name: str
    lane: Lane
    family: str
    faithfulness: Faithfulness
    summary: str
    status: Status
    tier: Tier = Tier.EXPERIMENTAL
    rationale: str = ""
    modality: tuple[str, ...] = ("ecg", "ppg")
    window_samples: int | None = None
    sample_rate: int | None = None
    target_crs: tuple[int, ...] = ()
    edge_notes: str = ""
    preprocess: str | None = None
    transform: str | None = None
    encoder_stage: str | None = None
    entropy: str | None = None
    builder: CodecBuilder | None = field(default=None, repr=False)
    golden_id: str | None = None

    @property
    def stages(self) -> tuple[str, str, str, str] | None:
        """The 4-stage decomposition, or ``None`` if not annotated."""
        slots = (self.preprocess, self.transform, self.encoder_stage, self.entropy)
        if all(s is None for s in slots):
            return None
        return tuple(s or "-" for s in slots)  # type: ignore[return-value]

    @property
    def runnable(self) -> bool:
        """Whether the method can be built and run without trained weights."""
        return self.builder is not None


_CATALOG: dict[str, MethodCard] = {}


def register_method(card: MethodCard) -> MethodCard:
    """Register an explicit method card. Idempotent for the same object."""
    existing = _CATALOG.get(card.id)
    if existing is not None and existing is not card:
        raise ValueError(f"Method id {card.id!r} already registered.")
    _CATALOG[card.id] = card
    return card


# ---------------------------------------------------------------------------
# Golden-registry hook
# ---------------------------------------------------------------------------

_LANE_BY_METHOD: dict[str, Lane] = {
    "spiht": Lane.FAITHFUL,
    "hybrid": Lane.CLEAN,
    "rvq": Lane.COMPACT_LEARNED,
}
_FAITHFULNESS_BY_METHOD: dict[str, Faithfulness] = {
    "spiht": Faithfulness.FAITHFUL,
    "hybrid": Faithfulness.DENOISE,
    "rvq": Faithfulness.PERCEPTUAL,
}


def _cards_from_goldens() -> list[MethodCard]:
    """Surface every registered golden experiment as a shipped card."""
    from compressionkit.experiments.registry import list_goldens

    cards: list[MethodCard] = []
    for exp in list_goldens():
        cards.append(
            MethodCard(
                id=exp.experiment_id,
                display_name=exp.run_name,
                lane=_LANE_BY_METHOD.get(exp.method, Lane.COMPACT_LEARNED),
                family=f"golden-{exp.method}",
                faithfulness=_FAITHFULNESS_BY_METHOD.get(exp.method, Faithfulness.PERCEPTUAL),
                summary=f"Golden {exp.method} {exp.compression_ratio}x for {exp.modality}.",
                status=Status.SHIPPED,
                tier=Tier.ROBUST,
                rationale="Shipped golden run with validated deploy artifacts.",
                modality=(exp.modality,),
                sample_rate=exp.sample_rate,
                target_crs=(exp.compression_ratio,),
                edge_notes="Published LiteRT INT8 deploy artifacts.",
                golden_id=exp.experiment_id,
            )
        )
    return cards


def list_methods(
    *,
    lane: Lane | None = None,
    modality: str | None = None,
    status: Status | None = None,
    tier: Tier | None = None,
    include_goldens: bool = True,
) -> list[MethodCard]:
    """Return catalog cards, optionally filtered.

    Explicit cards take precedence over golden-derived cards on id collision.
    Sorted by lane, then curation tier (baseline -> robust -> experimental).
    """
    merged: dict[str, MethodCard] = {}
    if include_goldens:
        for card in _cards_from_goldens():
            merged[card.id] = card
    merged.update(_CATALOG)  # explicit cards win

    cards = list(merged.values())
    if lane is not None:
        cards = [c for c in cards if c.lane == lane]
    if modality is not None:
        cards = [c for c in cards if modality in c.modality]
    if status is not None:
        cards = [c for c in cards if c.status == status]
    if tier is not None:
        cards = [c for c in cards if c.tier == tier]
    return sorted(cards, key=lambda c: (c.lane.value, TIER_ORDER[c.tier], c.id))


def get_method(method_id: str) -> MethodCard:
    """Look up a card by id (explicit cards first, then golden-derived)."""
    if method_id in _CATALOG:
        return _CATALOG[method_id]
    for card in _cards_from_goldens():
        if card.id == method_id:
            return card
    known = sorted(c.id for c in list_methods())
    raise KeyError(f"Unknown method {method_id!r}. Known: {known}")


# ---------------------------------------------------------------------------
# Model-size accounting (compact encoder / heavy decoder asymmetry)
# ---------------------------------------------------------------------------


def encoder_decoder_params(golden_id: str, results_root: str = "results") -> tuple[int, int] | None:
    """Return ``(encoder_params, decoder_params)`` for a golden run if available.

    Reads the separately-saved ``encoder.keras`` / ``decoder.keras`` artifacts
    so the compact-encoder / heavy-decoder asymmetry can be reported per card.
    Returns ``None`` when the run directory or artifacts are not present.
    """
    from pathlib import Path

    from compressionkit.experiments.registry import get_golden

    try:
        exp = get_golden(golden_id)
    except KeyError:
        return None
    run_dir = Path(results_root) / exp.run_name
    enc_path = run_dir / "encoder.keras"
    dec_path = run_dir / "decoder.keras"
    if not enc_path.exists() or not dec_path.exists():
        return None

    import keras

    encoder = keras.models.load_model(enc_path, compile=False)
    decoder = keras.models.load_model(dec_path, compile=False)
    return int(encoder.count_params()), int(decoder.count_params())


__all__ = [
    "TIER_ORDER",
    "CodecBuilder",
    "Faithfulness",
    "MethodCard",
    "Status",
    "Tier",
    "encoder_decoder_params",
    "get_method",
    "list_methods",
    "register_method",
]
