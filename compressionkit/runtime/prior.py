"""Entropy prior runtime for RVQ token sequences.

Loads a prior TFLite model and computes per-token log-probabilities
for a sequence of RVQ indices.  Used by :class:`TwoStageCodec` to
achieve compression beyond the uniform-codebook baseline.

This module requires only ``numpy`` and a LiteRT interpreter.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from compressionkit.runtime.codec import _Interpreter

logger = logging.getLogger(__name__)


class EntropyPrior:
    """Lightweight entropy prior using a causal CNN TFLite model.

    The prior predicts next-token logits for a flattened sequence of
    RVQ indices.  Given indices of shape ``(B, T', num_levels)``, the
    tokens are flattened to ``(B, T' * num_levels)`` and fed through
    the causal model, which outputs ``(B, seq_len, vocab_size)``
    logits.

    Args:
        prior_tflite: Path to the prior ``.tflite`` model.
        vocab_size: Number of codebook entries (K). Defaults to 256.
        context_length: Maximum context window for the prior.
            If *None*, inferred from the TFLite input shape.

    Example::

        prior = EntropyPrior("prior.tflite")
        log_probs = prior.predict_log_probs(indices)
        bits_per_token = prior.bits_per_token(indices)
    """

    def __init__(
        self,
        prior_tflite: str | Path,
        vocab_size: int = 256,
        context_length: int | None = None,
    ) -> None:
        self._path = Path(prior_tflite)
        if not self._path.exists():
            raise FileNotFoundError(f"Prior TFLite not found: {self._path}")

        self._vocab_size = vocab_size
        self._interpreter = _Interpreter(model_path=str(self._path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()[0]
        self._output_details = self._interpreter.get_output_details()[0]

        # Infer context length from input shape
        input_shape = self._input_details["shape"]  # (1, context_length)
        self._context_length = context_length or int(input_shape[-1])

        logger.info(
            "EntropyPrior loaded: %s (ctx=%d, vocab=%d)",
            self._path.name, self._context_length, self._vocab_size,
        )

    @property
    def context_length(self) -> int:
        """Maximum context window length."""
        return self._context_length

    @property
    def vocab_size(self) -> int:
        """Codebook vocabulary size (K)."""
        return self._vocab_size

    def _flatten_indices(self, indices: np.ndarray) -> np.ndarray:
        """Flatten RVQ indices to a 1-D token sequence.

        Args:
            indices: Shape ``(B, T', num_levels)`` or ``(B, 1, T', num_levels)``.

        Returns:
            Flattened tokens of shape ``(B, T' * num_levels)``, with levels
            interleaved per time step: ``[t0_l0, t0_l1, t1_l0, t1_l1, ...]``.
        """
        if indices.ndim == 4:
            # (B, 1, T', L) → (B, T', L)
            indices = indices[:, 0]
        batch = indices.shape[0]
        return indices.reshape(batch, -1).astype(np.int32)

    def predict_logits(self, tokens: np.ndarray) -> np.ndarray:
        """Run the prior model on a flattened token sequence.

        Args:
            tokens: Token array ``(B, seq_len)`` with int32 dtype.
                ``seq_len`` must be ≤ ``context_length``.

        Returns:
            Logits array ``(B, seq_len, vocab_size)`` float32.
        """
        batch, seq_len = tokens.shape
        if seq_len > self._context_length:
            raise ValueError(
                f"Token sequence length {seq_len} exceeds "
                f"context_length {self._context_length}"
            )

        # Pad to context_length if needed
        if seq_len < self._context_length:
            pad_width = self._context_length - seq_len
            tokens = np.pad(tokens, ((0, 0), (0, pad_width)), constant_values=0)

        # Quantize input if needed
        inp = tokens.astype(self._input_details["dtype"])
        self._interpreter.set_tensor(self._input_details["index"], inp)
        self._interpreter.invoke()
        logits = self._interpreter.get_tensor(self._output_details["index"])

        # Dequantize output if INT8
        if self._output_details["dtype"] == np.int8:
            qp = self._output_details.get("quantization_parameters", {})
            scales = qp.get("scales", np.array([1.0]))
            zps = qp.get("zero_points", np.array([0]))
            logits = (logits.astype(np.float32) - zps[0]) * scales[0]

        # Trim to original sequence length
        return logits[:, :seq_len, :].astype(np.float32)

    def predict_log_probs(self, indices: np.ndarray) -> np.ndarray:
        """Compute per-token log-probabilities for RVQ indices.

        Uses the causal prior: for token at position ``t``, the
        prediction is based on tokens ``0..t-1``.  The log-prob at
        position 0 uses a uniform prior.

        Args:
            indices: RVQ indices ``(B, T', num_levels)`` or
                ``(B, 1, T', num_levels)``.

        Returns:
            Log-probabilities ``(B, seq_len)`` where ``seq_len = T' * num_levels``.
        """
        tokens = self._flatten_indices(indices)
        logits = self.predict_logits(tokens)

        # Shift: logits at position t predict token at t+1
        # For position 0, use uniform prior
        log_probs = np.full(tokens.shape, -np.log(self._vocab_size), dtype=np.float32)

        if tokens.shape[1] > 1:
            # logits[:, :-1, :] predict tokens[:, 1:]
            shifted_logits = logits[:, :-1, :]
            shifted_tokens = tokens[:, 1:]

            # Log-softmax
            max_logits = np.max(shifted_logits, axis=-1, keepdims=True)
            exp_logits = np.exp(shifted_logits - max_logits)
            log_sum_exp = np.log(np.sum(exp_logits, axis=-1, keepdims=True)) + max_logits
            all_log_probs = shifted_logits - log_sum_exp

            # Gather log-prob for the actual token
            batch_idx = np.arange(tokens.shape[0])[:, None]
            seq_idx = np.arange(shifted_tokens.shape[1])[None, :]
            log_probs[:, 1:] = all_log_probs[batch_idx, seq_idx, shifted_tokens]

        return log_probs

    def bits_per_token(self, indices: np.ndarray) -> float:
        """Compute mean bits-per-token for the given RVQ indices.

        Args:
            indices: RVQ indices ``(B, T', num_levels)``.

        Returns:
            Mean bits per token (lower is better).
        """
        log_probs = self.predict_log_probs(indices)
        return float(-np.mean(log_probs) / np.log(2))
