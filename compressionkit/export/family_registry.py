"""Single source of truth for codec-family dispatch.

Before this module existed, "what does codec family X need" was answered
independently in the runtime loader, deploy validator, model-card generator,
and the HuggingFace publisher — each with its own copy of the known-families
list, required-file list, and dispatch logic. That duplication is exactly how
the hybrid lane's ``family`` field went stale in one place (the exporter)
without the runtime loader, validator, or publisher noticing: nothing forced
them to agree.

Every consumer that needs to answer "given family X, which runtime class
loads it / which files are required / which model-card generator applies /
what's the default license" should look it up here via :data:`FAMILY_REGISTRY`
instead of re-deriving its own copy. Adding a new codec family means adding
one :class:`CodecFamilySpec` entry, not touching every consumer module.

Runtime classes are imported lazily (inside the small ``_load_*`` wrappers)
to avoid a module-import cycle: this module lives under ``compressionkit.export``
but needs to hand back ``compressionkit.runtime`` codec instances, and
``compressionkit.runtime.loader`` needs this registry — importing the runtime
classes only when a loader is actually invoked sidesteps that cycle entirely.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from compressionkit.export.artifact_contract import (
    HYBRID_HF_FILE_RENAMES,
    RVQ_HF_FILE_RENAMES,
    SPIHT_HF_FILE_RENAMES,
)
from compressionkit.export.model_card import generate_model_card, generate_spiht_model_card

if TYPE_CHECKING:
    from compressionkit.runtime.base import Codec, CodecFamily

__all__ = ["FAMILY_REGISTRY", "KNOWN_FAMILIES", "CodecFamilySpec", "get_family_spec"]


def _load_rvq(deploy_dir: Path) -> Codec:
    from compressionkit.runtime.codec import RVQCodec

    return RVQCodec(deploy_dir)


def _load_spiht(deploy_dir: Path) -> Codec:
    from compressionkit.runtime.spiht import SpihtCodec

    return SpihtCodec.from_deploy_dir(deploy_dir)


def _load_hybrid(deploy_dir: Path) -> Codec:
    from compressionkit.runtime.hybrid import HybridSpihtCodec

    return HybridSpihtCodec.from_deploy_dir(deploy_dir)


@dataclass(frozen=True)
class CodecFamilySpec:
    """Everything a downstream consumer needs to know about one codec family.

    Attributes:
        family: The manifest ``family`` value (``"rvq"``, ``"spiht"``, ``"hybrid"``).
        loader: Hydrates a codec instance from a local deploy directory.
        required_artifacts: Files required for baseline (non-strict) deploy validation.
        release_extras: Additional files required under ``strict_release``.
        hf_file_renames: ``(local_name, hf_name)`` pairs for HuggingFace staging.
        has_c_sources: Whether a ``c_sources/`` subtree should be copied verbatim
            when staging for HuggingFace (DSP families ship a portable C reference).
        has_trained_weights: Whether the package carries proprietary trained
            weights, i.e. whether ``LICENSE-MODEL-WEIGHTS.md`` should be staged
            and whether the default license should stay restrictive.
        model_card_generator: A ``generate_model_card``/``generate_spiht_model_card``
            compatible callable: ``(deploy_dir, scorecard_path, license_id) -> str``.
        default_license: SPDX license id used when the caller leaves the CLI's
            ``"other"`` sentinel unchanged.
    """

    family: CodecFamily
    loader: Callable[[Path], Codec]
    required_artifacts: tuple[str, ...]
    release_extras: tuple[str, ...]
    hf_file_renames: tuple[tuple[str, str], ...]
    has_c_sources: bool
    has_trained_weights: bool
    model_card_generator: Callable[..., str]
    default_license: str


FAMILY_REGISTRY: dict[str, CodecFamilySpec] = {
    "rvq": CodecFamilySpec(
        family="rvq",
        loader=_load_rvq,
        required_artifacts=("encoder.tflite", "codebook.npz", "codebook.h"),
        release_extras=("model_card.json", "README.md", "scorecard.json", "reference_vectors.npz", "sample_data.npz"),
        hf_file_renames=tuple(RVQ_HF_FILE_RENAMES),
        has_c_sources=False,
        has_trained_weights=True,
        model_card_generator=generate_model_card,
        default_license="other",
    ),
    "spiht": CodecFamilySpec(
        family="spiht",
        loader=_load_spiht,
        required_artifacts=("sample_stimulus.npz", "reference_vectors.npz", "spiht_app_config.h"),
        release_extras=(
            "model_card.json",
            "README.md",
            "scorecard.json",
            "reference_vectors.npz",
            "sample_stimulus.npz",
        ),
        hf_file_renames=tuple(SPIHT_HF_FILE_RENAMES),
        has_c_sources=True,
        has_trained_weights=False,
        model_card_generator=generate_spiht_model_card,
        default_license="apache-2.0",
    ),
    "hybrid": CodecFamilySpec(
        family="hybrid",
        loader=_load_hybrid,
        required_artifacts=(
            "sample_stimulus.npz",
            "reference_vectors.npz",
            "spiht_app_config.h",
            "denoiser_gain_model.keras",
            "denoiser_gain_model.tflite",
            "hybrid_manifest.json",
        ),
        release_extras=(
            "model_card.json",
            "README.md",
            "scorecard.json",
            "reference_vectors.npz",
            "sample_stimulus.npz",
            "denoiser_train_config.json",
        ),
        hf_file_renames=tuple(HYBRID_HF_FILE_RENAMES),
        has_c_sources=True,
        has_trained_weights=True,
        model_card_generator=generate_spiht_model_card,
        default_license="other",
    ),
}

KNOWN_FAMILIES: tuple[str, ...] = tuple(FAMILY_REGISTRY)


def get_family_spec(family: str) -> CodecFamilySpec:
    """Look up the :class:`CodecFamilySpec` for a manifest ``family`` value.

    Raises:
        ValueError: If ``family`` is not a registered codec family.
    """
    try:
        return FAMILY_REGISTRY[family]
    except KeyError:
        raise ValueError(f"Unknown codec family {family!r}; expected one of {KNOWN_FAMILIES!r}") from None
