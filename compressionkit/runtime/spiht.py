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
from pathlib import Path

import numpy as np

from compressionkit.evaluation.codec import SpihtAcCodec
from compressionkit.runtime.base import EncodedFrame

logger = logging.getLogger(__name__)

__all__ = ["SpihtCodec"]


class SpihtCodec:
    """Thin runtime wrapper around :class:`SpihtAcCodec`.

    Adds the deploy-directory loading convention plus ``compress`` /
    ``decompress`` methods that satisfy
    :class:`compressionkit.runtime.base.Codec`.
    """

    def __init__(
        self,
        *,
        modality: str,
        sample_rate: int,
        frame_size: int,
        target_cr: float,
        wavelet: str = "bior4.4",
        levels: int = 6,
        use_ac: bool = True,
        bits_per_sample: int = 16,
        name: str | None = None,
        deploy_dir: Path | None = None,
        manifest: dict | None = None,
    ) -> None:
        self._inner = SpihtAcCodec(
            name=name or f"spiht_{modality}_{int(target_cr):02d}x",
            modality=modality,
            sample_rate=sample_rate,
            frame_size=frame_size,
            target_cr=target_cr,
            wavelet=wavelet,
            levels=levels,
            use_ac=use_ac,
            bits_per_sample=bits_per_sample,
        )
        self._deploy_dir = deploy_dir
        self._manifest = manifest or {}

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
        inner = self._inner.encode(np.asarray(frame, dtype=np.float32))
        return EncodedFrame(payload=inner.payload, nbits=inner.nbits, side=dict(inner.side))

    def decompress(self, encoded: EncodedFrame) -> np.ndarray:
        """Decode an opaque payload back to a ``(frame_size,)`` frame."""
        # SpihtAcCodec.decode expects its own EncodedFrame dataclass; rebuild it.
        from compressionkit.evaluation.codec import EncodedFrame as _EvalEncodedFrame

        inner = _EvalEncodedFrame(
            payload=encoded.payload, nbits=encoded.nbits, side=dict(encoded.side)
        )
        return self._inner.decode(inner)

    # ----- Loader conventions -----

    @classmethod
    def from_deploy_dir(cls, deploy_dir: str | Path) -> SpihtCodec:
        """Hydrate a SPIHT codec from a deploy directory.

        Reads ``deploy_manifest.json`` and constructs the codec with
        the parameters stored under ``manifest["codec"]``.
        """
        deploy_dir = Path(deploy_dir)
        manifest_path = deploy_dir / "deploy_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"deploy_manifest.json not found in {deploy_dir}")
        with manifest_path.open() as f:
            manifest = json.load(f)

        if manifest.get("family") != "spiht":
            raise ValueError(
                f"Expected family='spiht' in manifest, got {manifest.get('family')!r}"
            )

        codec_cfg = manifest.get("codec", {})
        return cls(
            modality=codec_cfg["modality"],
            sample_rate=int(codec_cfg["sample_rate"]),
            frame_size=int(codec_cfg["frame_size"]),
            target_cr=float(codec_cfg["target_cr"]),
            wavelet=codec_cfg.get("wavelet", "bior4.4"),
            levels=int(codec_cfg.get("levels", 6)),
            use_ac=bool(codec_cfg.get("use_ac", True)),
            bits_per_sample=int(codec_cfg.get("bits_per_sample", 16)),
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
                "huggingface_hub is required for from_pretrained(). "
                "Install with: uv sync --extra hf"
            ) from exc

        kwargs: dict = {"repo_id": repo_id, "repo_type": "model"}
        if revision is not None:
            kwargs["revision"] = revision
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)

        local_dir = snapshot_download(**kwargs)
        return cls.from_deploy_dir(local_dir)
