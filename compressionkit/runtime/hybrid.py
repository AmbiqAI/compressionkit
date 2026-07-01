"""Hybrid codec runtimes.

The v1 hybrid lane is a learned wavelet-gain preprocessor followed by the
standard SPIHT runtime. The deploy package carries both the SPIHT operating
point and a ``hybrid_manifest.json`` that identifies the denoiser artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from compressionkit.pipeline.learned_stages import load_wavelet_gain_preprocessor
from compressionkit.runtime.base import EncodedFrame
from compressionkit.runtime.spiht import SpihtCodec, SpihtCodecConfig


class HybridSpihtCodec:
    """Learned-denoiser + SPIHT runtime loaded from a deploy directory."""

    def __init__(
        self,
        *,
        spiht: SpihtCodec,
        denoiser: Any,
        manifest: dict[str, Any],
        hybrid_manifest: dict[str, Any],
        deploy_dir: Path,
    ) -> None:
        self._spiht = spiht
        self._denoiser = denoiser
        self._manifest = manifest
        self._hybrid_manifest = hybrid_manifest
        self._deploy_dir = deploy_dir

    @classmethod
    def from_deploy_dir(cls, deploy_dir: str | Path) -> HybridSpihtCodec:
        """Hydrate the v1 learned wavelet-gain SPIHT hybrid package."""
        deploy_dir = Path(deploy_dir)
        manifest_path = deploy_dir / "deploy_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"deploy_manifest.json not found in {deploy_dir}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("family") != "hybrid":
            raise ValueError(f"Expected family='hybrid' in manifest, got {manifest.get('family')!r}")

        hybrid_manifest_path = deploy_dir / "hybrid_manifest.json"
        if not hybrid_manifest_path.exists():
            raise FileNotFoundError(f"hybrid_manifest.json not found in {deploy_dir}")
        hybrid_manifest = json.loads(hybrid_manifest_path.read_text())
        stages = hybrid_manifest.get("stages")
        if not isinstance(stages, list):
            raise ValueError("hybrid_manifest['stages'] must be a list")
        denoise_stage = next(
            (stage for stage in stages if isinstance(stage, dict) and stage.get("stage") == "denoise"), None
        )
        if not isinstance(denoise_stage, dict):
            raise ValueError("hybrid_manifest is missing a denoise stage")

        spec_name = manifest.get("spec", "codec_spec.json")
        spec_path = deploy_dir / str(spec_name)
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        codec_section = spec.get("codec") if isinstance(spec, dict) else None
        if not isinstance(codec_section, dict):
            codec_section = manifest.get("codec")
        if not isinstance(codec_section, dict):
            raise ValueError("hybrid deploy package is missing SPIHT codec parameters")
        known = set(SpihtCodecConfig.model_fields)
        config = SpihtCodecConfig.model_validate({k: v for k, v in codec_section.items() if k in known})

        denoiser_artifact = denoise_stage.get("artifact")
        if not isinstance(denoiser_artifact, str):
            raise ValueError("hybrid denoise stage is missing its artifact path")
        denoiser = load_wavelet_gain_preprocessor(
            deploy_dir / denoiser_artifact,
            frame_size=int(denoise_stage.get("frame_size") or config.frame_size),
            wavelet=str(denoise_stage.get("wavelet") or config.wavelet),
            levels=int(denoise_stage.get("levels") or config.levels),
        )
        spiht = SpihtCodec(
            config=config,
            name=str(manifest.get("model_name") or "hybrid_spiht"),
            deploy_dir=deploy_dir,
            manifest=manifest,
        )
        return cls(
            spiht=spiht,
            denoiser=denoiser,
            manifest=manifest,
            hybrid_manifest=hybrid_manifest,
            deploy_dir=deploy_dir,
        )

    @property
    def name(self) -> str:
        return str(self._manifest.get("model_name") or self._spiht.name)

    @property
    def modality(self) -> str:
        return self._spiht.modality

    @property
    def sample_rate(self) -> int:
        return self._spiht.sample_rate

    @property
    def frame_size(self) -> int:
        return self._spiht.frame_size

    @property
    def target_cr(self) -> float:
        return self._spiht.target_cr

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    @property
    def hybrid_manifest(self) -> dict[str, Any]:
        return dict(self._hybrid_manifest)

    @property
    def deploy_dir(self) -> Path:
        return self._deploy_dir

    def compress(self, frame: np.ndarray) -> EncodedFrame:
        """Denoise a frame, then encode it with the SPIHT backend."""
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"HybridSpihtCodec.compress expects a 1-D frame, got shape {arr.shape}")
        if arr.shape[0] != self.frame_size:
            raise ValueError(f"HybridSpihtCodec.compress expects {self.frame_size} samples, got {arr.shape[0]}")
        denoised, _ = self._denoiser.forward(arr)
        encoded = self._spiht.compress(denoised)
        side = dict(encoded.side)
        side["pre_denoised"] = True
        side["hybrid_pipeline"] = "wavelet_gain_spiht"
        return EncodedFrame(payload=encoded.payload, nbits=encoded.nbits, side=side)

    def decompress(self, encoded: EncodedFrame) -> np.ndarray:
        """Decode the SPIHT payload produced by :meth:`compress`."""
        return self._spiht.decompress(encoded)


__all__ = ["HybridSpihtCodec"]
