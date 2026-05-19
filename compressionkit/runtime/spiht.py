"""DSP-only SPIHT codec runtime.

Wraps :class:`compressionkit.evaluation.codec.SpihtAcCodec` with the
loader convention used by RVQ goldens so a user can do::

    from compressionkit.runtime import load_codec

    codec = load_codec("Ambiq/compressionkit-ppg-spiht-4x")
    enc = codec.compress(frame)
    recon = codec.decompress(enc)

SPIHT deploy packages contain no weights — only a small manifest with
codec parameters (wavelet, levels, bit budget formula, AC flag), a
scorecard, and license-safe sample stimulus. The C sources live under
``c_sources/`` inside the deploy package for embedded integration.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from compressionkit.evaluation.codec import SpihtAcCodec
from compressionkit.runtime.base import EncodedFrame

logger = logging.getLogger(__name__)

__all__ = ["SpihtCodec", "SpihtCodecConfig"]

# Wavelet / modality names are embedded verbatim in generated C string
# literals; restrict them up-front to safe identifier-like characters.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Modality-paired wavelet defaults (mirrors AGENTS.md).
_DEFAULT_WAVELET_BY_MODALITY: dict[str, str] = {"ppg": "coif5", "ecg": "bior4.4"}

Modality = Literal["ppg", "ecg"]


class SpihtCodecConfig(BaseModel):
    """Typed codec parameters for SPIHT.

    Used both as the constructor argument for :class:`SpihtCodec` and
    as the schema for the ``codec`` section of ``deploy_manifest.json``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    modality: Modality
    sample_rate: int = Field(..., gt=0)
    frame_size: int = Field(..., gt=0)
    target_cr: float = Field(..., gt=1.0)
    wavelet: str = Field(..., min_length=1)
    levels: int = Field(default=6, gt=0, le=10)
    use_ac: bool = True
    bits_per_sample: int = Field(default=16, gt=0, le=32)

    @model_validator(mode="after")
    def _check_frame_vs_levels(self) -> SpihtCodecConfig:
        """A DWT of ``levels`` levels needs ``frame_size >= 2**levels``."""
        minimum = 1 << self.levels
        if self.frame_size < minimum:
            raise ValueError(
                f"frame_size={self.frame_size} is too small for levels={self.levels}; "
                f"need at least {minimum} samples (2**levels)"
            )
        return self

    @classmethod
    def with_defaults(
        cls,
        *,
        modality: Modality,
        sample_rate: int,
        frame_size: int,
        target_cr: float,
        wavelet: str | None = None,
        levels: int = 6,
        use_ac: bool = True,
        bits_per_sample: int = 16,
    ) -> SpihtCodecConfig:
        """Build a config with the modality-paired wavelet as default."""
        chosen_wavelet = wavelet or _DEFAULT_WAVELET_BY_MODALITY[modality]
        return cls(
            modality=modality,
            sample_rate=sample_rate,
            frame_size=frame_size,
            target_cr=target_cr,
            wavelet=chosen_wavelet,
            levels=levels,
            use_ac=use_ac,
            bits_per_sample=bits_per_sample,
        )

    def validate_c_safe(self) -> None:
        """Reject names that would corrupt the generated C header."""
        if not _SAFE_NAME_RE.match(self.wavelet):
            raise ValueError(f"wavelet name {self.wavelet!r} contains characters unsafe for C string-literal embedding")
        if not _SAFE_NAME_RE.match(self.modality):
            raise ValueError(
                f"modality name {self.modality!r} contains characters unsafe for C string-literal embedding"
            )


class SpihtCodec:
    """Thin runtime wrapper around :class:`SpihtAcCodec`.

    Adds the deploy-directory loading convention plus ``compress`` /
    ``decompress`` methods that satisfy
    :class:`compressionkit.runtime.base.Codec`.
    """

    def __init__(
        self,
        *,
        modality: Modality | None = None,
        sample_rate: int | None = None,
        frame_size: int | None = None,
        target_cr: float | None = None,
        wavelet: str | None = None,
        levels: int = 6,
        use_ac: bool = True,
        bits_per_sample: int = 16,
        name: str | None = None,
        config: SpihtCodecConfig | None = None,
        deploy_dir: Path | None = None,
        manifest: dict | None = None,
    ) -> None:
        if config is None:
            if modality is None or sample_rate is None or frame_size is None or target_cr is None:
                raise ValueError(
                    "SpihtCodec requires either a `config=SpihtCodecConfig(...)` or all of "
                    "(modality, sample_rate, frame_size, target_cr)"
                )
            config = SpihtCodecConfig.with_defaults(
                modality=modality,
                sample_rate=sample_rate,
                frame_size=frame_size,
                target_cr=target_cr,
                wavelet=wavelet,
                levels=levels,
                use_ac=use_ac,
                bits_per_sample=bits_per_sample,
            )
        config.validate_c_safe()
        self._config = config
        self._inner = SpihtAcCodec(
            name=name or f"spiht_{config.modality}_{int(config.target_cr):02d}x",
            modality=config.modality,
            sample_rate=config.sample_rate,
            frame_size=config.frame_size,
            target_cr=config.target_cr,
            wavelet=config.wavelet,
            levels=config.levels,
            use_ac=config.use_ac,
            bits_per_sample=config.bits_per_sample,
        )
        self._deploy_dir = deploy_dir
        self._manifest = manifest or {}

    @property
    def config(self) -> SpihtCodecConfig:
        """Typed codec parameters."""
        return self._config

    # ----- Codec protocol fields -----

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def modality(self) -> str:
        return self._inner.modality

    @property
    def sample_rate(self) -> int:
        return self._inner.sample_rate

    @property
    def frame_size(self) -> int:
        return self._inner.frame_size

    @property
    def target_cr(self) -> float:
        return self._inner.target_cr

    @property
    def wavelet(self) -> str:
        return self._inner.wavelet

    @property
    def levels(self) -> int:
        return self._inner.levels

    @property
    def use_ac(self) -> bool:
        return self._inner.use_ac

    @property
    def max_bits(self) -> int:
        return self._inner.max_bits

    @property
    def manifest(self) -> dict:
        return dict(self._manifest)

    @property
    def deploy_dir(self) -> Path | None:
        return self._deploy_dir

    @property
    def sample_stimulus_path(self) -> Path | None:
        if self._deploy_dir is None:
            return None
        for name in ("sample_stimulus.npz", "sample_data.npz"):
            candidate = self._deploy_dir / name
            if candidate.exists():
                return candidate
        return None

    # ----- Codec protocol methods -----

    def compress(self, frame: np.ndarray) -> EncodedFrame:
        """Encode a single ``(frame_size,)`` frame."""
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"SpihtCodec.compress expects a 1-D (frame_size,) array, got shape {arr.shape}")
        if arr.shape[0] != self._inner.frame_size:
            raise ValueError(f"SpihtCodec.compress expects {self._inner.frame_size} samples, got {arr.shape[0]}")
        inner = self._inner.encode(arr)
        return EncodedFrame(payload=inner.payload, nbits=inner.nbits, side=dict(inner.side))

    def decompress(self, encoded: EncodedFrame) -> np.ndarray:
        """Decode an opaque payload back to a ``(frame_size,)`` frame."""
        # SpihtAcCodec.decode expects its own EncodedFrame dataclass; rebuild it.
        from compressionkit.evaluation.codec import EncodedFrame as _EvalEncodedFrame

        inner = _EvalEncodedFrame(payload=encoded.payload, nbits=encoded.nbits, side=dict(encoded.side))
        return self._inner.decode(inner)

    # ----- Loader conventions -----

    @classmethod
    def from_deploy_dir(cls, deploy_dir: str | Path) -> SpihtCodec:
        """Hydrate a SPIHT codec from a deploy directory.

        Reads ``deploy_manifest.json`` and constructs the codec with
        the parameters stored under ``manifest["codec"]``. The codec
        section is validated via :class:`SpihtCodecConfig`.
        """
        deploy_dir = Path(deploy_dir)
        manifest_path = deploy_dir / "deploy_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"deploy_manifest.json not found in {deploy_dir}")
        with manifest_path.open() as f:
            manifest = json.load(f)

        if manifest.get("family") != "spiht":
            raise ValueError(f"Expected family='spiht' in manifest, got {manifest.get('family')!r}")

        codec_section = manifest.get("codec")
        if not isinstance(codec_section, dict):
            raise ValueError(f"manifest['codec'] must be a dict, got {type(codec_section).__name__}")
        # Drop unknown keys (forward-compat) before strict pydantic parse.
        known = set(SpihtCodecConfig.model_fields)
        cleaned = {k: v for k, v in codec_section.items() if k in known}
        config = SpihtCodecConfig.model_validate(cleaned)
        return cls(
            config=config,
            name=manifest.get("model_name"),
            deploy_dir=deploy_dir,
            manifest=manifest,
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> SpihtCodec:
        """Download a SPIHT codec from HuggingFace and hydrate it."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "huggingface_hub is required for from_pretrained(). Install with: uv sync --extra hf"
            ) from exc

        kwargs: dict[str, str] = {"repo_id": repo_id, "repo_type": "model"}
        if revision is not None:
            kwargs["revision"] = revision
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)

        local_dir = snapshot_download(**kwargs)
        return cls.from_deploy_dir(local_dir)
