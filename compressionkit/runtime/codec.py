"""Lightweight RVQ codec for inference using LiteRT + numpy.

This module requires only ``numpy`` and ``ai-edge-litert`` (or
``tflite-runtime``).  No Keras, TensorFlow, or training dependencies
are needed.

Example::

    from compressionkit.runtime import RVQCodec

    codec = RVQCodec("path/to/deploy/")
    indices = codec.encode(signal)   # (1, 1, T, 1) float32 → (1, 1, T', levels) int
    recon   = codec.decode(indices)  # → (1, 1, T, 1) float32
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Try ai-edge-litert first, then tflite-runtime, then tf.lite
_Interpreter = None

with contextlib.suppress(ImportError):
    from ai_edge_litert.interpreter import Interpreter as _Interpreter  # type: ignore[assignment]
if _Interpreter is None:
    with contextlib.suppress(ImportError):
        from tflite_runtime.interpreter import Interpreter as _Interpreter  # type: ignore[assignment]
if _Interpreter is None:
    with contextlib.suppress(ImportError):
        from tensorflow.lite.python.interpreter import Interpreter as _Interpreter  # type: ignore[assignment]

if _Interpreter is None:
    raise ImportError("No TFLite runtime found. Install one of: ai-edge-litert, tflite-runtime, or tensorflow.")


def _ensure_symlink(directory: Path, hf_name: str, local_name: str) -> None:
    """Create a symlink from local_name → hf_name if hf_name exists but local_name doesn't."""
    hf_path = directory / hf_name
    local_path = directory / local_name
    if hf_path.exists() and not local_path.exists():
        local_path.symlink_to(hf_name)  # relative symlink: both files are in the same directory


class RVQCodec:
    """Lightweight RVQ autoencoder codec using LiteRT for inference.

    Loads a deployment package (``deploy_manifest.json``, ``.tflite``
    models, ``codebook.npz``) and provides ``encode`` / ``decode``
    methods that run entirely on LiteRT + numpy.

    Args:
        deploy_dir: Path to directory containing deployment artifacts.

    Example::

        codec = RVQCodec("results/ppg_rvq_64hz_04x_golden/deploy")

        # Or load from HuggingFace Hub
        codec = RVQCodec.from_pretrained("Ambiq/compressionkit-ppg-4x")

        # Encode: float32 signal → RVQ indices
        signal = np.random.randn(1, 1, 320, 1).astype(np.float32)
        indices = codec.encode(signal)

        # Decode: RVQ indices → reconstructed signal
        recon = codec.decode(indices)
    """

    def __init__(self, deploy_dir: str | Path) -> None:
        self._deploy_dir = Path(deploy_dir)
        manifest_path = self._deploy_dir / "deploy_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Deploy manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            self._manifest = json.load(f)

        self._spec = self._load_codec_spec()
        runtime_spec = self._spec or self._manifest

        # Load encoder
        enc_path = self._deploy_dir / runtime_spec["encoder"]["tflite"]
        self._encoder = _Interpreter(model_path=str(enc_path))
        self._encoder.allocate_tensors()
        self._enc_input = self._encoder.get_input_details()[0]
        self._enc_output = self._encoder.get_output_details()[0]

        # Load decoder — prefer float32 TFLite, fall back to INT8, then skip
        self._decoder = None
        self._dec_input = None
        self._dec_output = None
        dec_info = runtime_spec.get("decoder", {})
        dec_f32 = dec_info.get("float32_tflite")
        dec_int8 = dec_info.get("int8_tflite") or dec_info.get("tflite")

        # Also check for legacy "keras"-only manifests that ship a decoder.tflite
        dec_candidates = []
        if dec_f32:
            dec_candidates.append(dec_f32)
        if dec_int8:
            dec_candidates.append(dec_int8)
        # Fallback: look for common filenames
        for fallback in ("decoder_float32.tflite", "decoder.tflite"):
            if fallback not in dec_candidates:
                dec_candidates.append(fallback)

        for dec_name in dec_candidates:
            dec_path = self._deploy_dir / dec_name
            if dec_path.exists():
                self._decoder = _Interpreter(model_path=str(dec_path))
                self._decoder.allocate_tensors()
                self._dec_input = self._decoder.get_input_details()[0]
                self._dec_output = self._decoder.get_output_details()[0]
                logger.info("Loaded decoder: %s", dec_name)
                break

        # Load codebook
        cb_path = self._deploy_dir / runtime_spec["codebook"]["npz"]
        cb_data = np.load(cb_path)
        self._codebooks = [cb_data[k] for k in sorted(cb_data.files)]
        self._num_levels = len(self._codebooks)
        self._num_embeddings = self._codebooks[0].shape[0]
        self._embedding_dim = self._codebooks[0].shape[1]

        logger.info(
            "RVQCodec loaded: %s (levels=%d, K=%d, D=%d)",
            self._manifest.get("model_name", "unknown"),
            self._num_levels,
            self._num_embeddings,
            self._embedding_dim,
        )

    def _load_codec_spec(self) -> dict:
        spec_name = self._manifest.get("spec", "codec_spec.json")
        if isinstance(spec_name, str):
            spec_path = self._deploy_dir / spec_name
            if spec_path.exists():
                with spec_path.open() as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    return loaded
        return {}

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, cache_dir: str | Path | None = None
    ) -> RVQCodec:
        """Load a codec from a HuggingFace Hub model repository.

        Downloads the deployment artifacts and creates an ``RVQCodec``
        instance pointing at the cached snapshot directory.

        Requires ``huggingface_hub`` (install with ``uv sync --extra hf``).

        Args:
            repo_id: HuggingFace repo ID, e.g.
                ``"Ambiq/compressionkit-ppg-4x"``.
            revision: Optional git revision (branch, tag, or commit hash).
            cache_dir: Optional local cache directory for downloaded files.

        Returns:
            An ``RVQCodec`` loaded from the downloaded artifacts.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for from_pretrained(). Install with: uv sync --extra hf"
            ) from exc

        kwargs: dict = {"repo_id": repo_id, "repo_type": "model"}
        if revision is not None:
            kwargs["revision"] = revision
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)

        local_dir = snapshot_download(**kwargs)
        logger.info("Downloaded %s → %s", repo_id, local_dir)

        # HF repos store the manifest as config.json; the codec expects
        # deploy_manifest.json.  Create a symlink if needed.
        local_path = Path(local_dir)
        manifest_path = local_path / "deploy_manifest.json"
        config_path = local_path / "config.json"
        if not manifest_path.exists() and config_path.exists():
            manifest_path.symlink_to(config_path)

        # HF repos rename encoder.tflite → encoder_int8.tflite.
        # Symlink the expected name if the manifest references it.
        _ensure_symlink(local_path, "encoder_int8.tflite", "encoder.tflite")
        _ensure_symlink(local_path, "decoder_int8.tflite", "decoder.tflite")
        _ensure_symlink(local_path, "sample_stimulus.npz", "sample_data.npz")

        return cls(local_dir)

    @property
    def manifest(self) -> dict:
        """Return the deployment manifest dictionary."""
        return self._manifest

    @property
    def spec(self) -> dict:
        """Return the canonical runtime hydration spec when present."""
        return self._spec

    @property
    def num_levels(self) -> int:
        """Number of RVQ quantization levels."""
        return self._num_levels

    @property
    def num_embeddings(self) -> int:
        """Number of codebook entries per level (K)."""
        return self._num_embeddings

    @property
    def embedding_dim(self) -> int:
        """Codebook embedding dimension (D)."""
        return self._embedding_dim

    @property
    def has_decoder(self) -> bool:
        """Whether a decoder TFLite model is available."""
        return self._decoder is not None

    def _quantize(self, data: np.ndarray, details: dict) -> np.ndarray:
        """Quantize float32 data to INT8 using TFLite quantization params."""
        qparams = details.get("quantization_parameters", {})
        scales = qparams.get("scales", np.array([1.0]))
        zero_points = qparams.get("zero_points", np.array([0]))
        if details["dtype"] == np.int8:
            quantized = np.round(data / scales[0] + zero_points[0])
            return np.clip(quantized, -128, 127).astype(np.int8)
        return data.astype(details["dtype"])

    def _dequantize(self, data: np.ndarray, details: dict) -> np.ndarray:
        """Dequantize INT8 data back to float32."""
        qparams = details.get("quantization_parameters", {})
        scales = qparams.get("scales", np.array([1.0]))
        zero_points = qparams.get("zero_points", np.array([0]))
        if details["dtype"] == np.int8:
            return ((data.astype(np.float32) - zero_points[0]) * scales[0]).astype(np.float32)
        return data.astype(np.float32)

    def encode_latent(self, signal: np.ndarray) -> np.ndarray:
        """Run the encoder to get continuous latent vectors.

        Args:
            signal: Input signal array matching encoder input shape.
                For a typical model: ``(1, 1, T, 1)`` float32.

        Returns:
            Latent array in float32, shape matching encoder output
            (e.g. ``(1, 1, T', D)``).
        """
        inp = self._quantize(signal, self._enc_input)
        self._encoder.set_tensor(self._enc_input["index"], inp)
        self._encoder.invoke()
        raw_out = self._encoder.get_tensor(self._enc_output["index"])
        return self._dequantize(raw_out, self._enc_output)

    def quantize_latent(self, latent: np.ndarray) -> np.ndarray:
        """Quantize continuous latents to RVQ indices via codebook lookup.

        Performs residual vector quantization: at each level, find the
        nearest codebook entry, record its index, and subtract the
        selected embedding from the residual.

        Args:
            latent: Continuous latent array, shape ``(..., D)`` where
                D is the codebook embedding dimension.

        Returns:
            Index array of shape ``(*latent.shape[:-1], num_levels)``
            with integer dtype.
        """
        spatial_shape = latent.shape[:-1]
        flat = latent.reshape(-1, self._embedding_dim)

        indices = np.zeros((flat.shape[0], self._num_levels), dtype=np.int32)
        residual = flat.copy()

        for level in range(self._num_levels):
            cb = self._codebooks[level]  # (K, D)
            # Nearest-neighbor: argmin ||residual - cb||^2
            # = argmin(||r||^2 - 2*r@cb.T + ||cb||^2)
            dots = residual @ cb.T  # (N, K)
            cb_norms = np.sum(cb**2, axis=1, keepdims=True).T  # (1, K)
            dists = -2 * dots + cb_norms  # ignore ||r||^2 (constant per query)
            indices[:, level] = np.argmin(dists, axis=1)
            selected = cb[indices[:, level]]  # (N, D)
            residual = residual - selected

        return indices.reshape(*spatial_shape, self._num_levels)

    def dequantize_indices(self, indices: np.ndarray) -> np.ndarray:
        """Reconstruct continuous latents from RVQ indices.

        Sums the codebook embeddings across all levels to reconstruct
        the quantized latent vector.

        Args:
            indices: Index array of shape ``(..., num_levels)`` with
                integer dtype.

        Returns:
            Reconstructed latent array of shape ``(..., D)``.
        """
        spatial_shape = indices.shape[:-1]
        flat = indices.reshape(-1, self._num_levels)

        reconstructed = np.zeros((flat.shape[0], self._embedding_dim), dtype=np.float32)
        for level in range(self._num_levels):
            cb = self._codebooks[level]
            reconstructed += cb[flat[:, level]]

        return reconstructed.reshape(*spatial_shape, self._embedding_dim)

    def decode_latent(self, latent: np.ndarray) -> np.ndarray:
        """Run the decoder TFLite model on a latent array.

        Args:
            latent: Latent array matching decoder input shape (float32).

        Returns:
            Reconstructed signal array in float32.

        Raises:
            RuntimeError: If no decoder TFLite model is available.
        """
        if self._decoder is None:
            raise RuntimeError(
                "No decoder TFLite available. Use dequantize_indices() "
                "for codebook-only reconstruction, or export a decoder "
                "TFLite via export_for_deployment()."
            )
        inp = self._quantize(latent, self._dec_input)
        self._decoder.set_tensor(self._dec_input["index"], inp)
        self._decoder.invoke()
        raw_out = self._decoder.get_tensor(self._dec_output["index"])
        return self._dequantize(raw_out, self._dec_output)

    def encode(self, signal: np.ndarray) -> np.ndarray:
        """Encode a signal to RVQ indices.

        Runs the encoder TFLite model and then quantizes the latent
        output via codebook lookup.

        Args:
            signal: Input signal, shape matching encoder input
                (e.g. ``(1, 1, 320, 1)`` float32).

        Returns:
            RVQ index array, shape ``(*latent_spatial, num_levels)``.
        """
        latent = self.encode_latent(signal)
        return self.quantize_latent(latent)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Decode RVQ indices back to a signal.

        Reconstructs the latent vector from codebook indices and runs
        the decoder TFLite model.

        Args:
            indices: RVQ index array from ``encode()``.

        Returns:
            Reconstructed signal in float32.
        """
        latent = self.dequantize_indices(indices)
        return self.decode_latent(latent)

    # ------------------------------------------------------------------
    # Codec protocol shims
    # ------------------------------------------------------------------
    # These let ``RVQCodec`` satisfy
    # :class:`compressionkit.runtime.base.Codec` without changing the
    # existing ``encode``/``decode`` semantics that downstream code and
    # tests already rely on.

    @property
    def name(self) -> str:
        """Codec name (from manifest, falls back to ``"rvq"``)."""
        return str(self._manifest.get("model_name", "rvq"))

    @property
    def modality(self) -> str:
        """``"ppg"`` or ``"ecg"`` if recorded in the model card."""
        card = self._manifest.get("model_card", {})
        return str(card.get("modality", "unknown"))

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz (from model card; ``0`` if unknown)."""
        card = self._manifest.get("model_card", {})
        return int(card.get("sample_rate", 0) or 0)

    @property
    def frame_size(self) -> int:
        """Number of samples per input frame (from encoder shape)."""
        shape = self._manifest.get("encoder", {}).get("input_shape", [])
        # Encoder input is typically (N, 1, T, 1); pull the time axis if present.
        if len(shape) >= 3 and shape[-2] is not None:
            return int(shape[-2])
        return 0

    @property
    def target_cr(self) -> float:
        """Target compression ratio from the model card."""
        card = self._manifest.get("model_card", {})
        return float(card.get("compression_ratio", 0.0) or 0.0)

    def compress(self, frame):
        """Encode a frame for the uniform :class:`Codec` protocol.

        Returns an :class:`~compressionkit.runtime.base.EncodedFrame`
        whose payload carries the RVQ index array.
        """
        from compressionkit.runtime.base import EncodedFrame

        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, 1, -1, 1)
        elif arr.ndim != 4:
            raise ValueError(f"RVQCodec.compress expects a (T,) or (N,1,T,1) frame, got shape {arr.shape}")
        indices = self.encode(arr)
        # Theoretical upper-bound: every index uniformly drawn from the full
        # codebook. True entropy is lower; use this for worst-case CR reporting.
        nbits = int(indices.size) * int(np.ceil(np.log2(max(self._num_embeddings, 2))))
        return EncodedFrame(
            payload=indices,
            nbits=nbits,
            side={"indices_shape": tuple(indices.shape)},
        )

    def decompress(self, encoded):
        """Decode an :class:`EncodedFrame` produced by :meth:`compress`."""
        recon = self.decode(np.asarray(encoded.payload))
        return np.asarray(recon, dtype=np.float32).reshape(-1)
