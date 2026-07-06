"""Benchmark harness for entropy-coding algorithms across validation sets and noise conditions.

Complements :mod:`compressionkit.runtime.entropy_algorithms`: that module
defines *what* an entropy algorithm is (anything satisfying
:class:`compressionkit.pipeline.stages.EntropyCoder`); this module defines
*how to test one* — on a plain token stream, or swept across the same
SNR/noise-bank conditions used to train the codecs themselves (via
:mod:`compressionkit.evaluation.empirical_regime`, the shared primitives
already used by the RVQ-vs-SPIHT robustness sweeps).

Typical usage — compare an AI prior against classical baselines on clean
tokens, then sweep both across SNR::

    from compressionkit.runtime.entropy_algorithms import LearnedEntropyCoder, StaticHistogramEntropyCoder
    from compressionkit.evaluation.entropy_benchmark import evaluate_entropy_coder, run_noise_sweep

    result = evaluate_entropy_coder(coder, clean_tokens, vocab_size=256)

    results = run_noise_sweep(
        encode_to_tokens=lambda signal: my_codec.encode(signal),  # -> flat int token array
        make_coder=lambda: LearnedEntropyCoder(prior=my_prior, backend="rans"),
        clean_windows=clean_windows,          # (n, window_len) float32
        noise_bank=noise_bank,                # 1-D or ragged noise segments
        snr_db_list=DEFAULT_SNR_DB,            # from compressionkit.evaluation.empirical_regime
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from compressionkit.evaluation.empirical_regime import add_empirical_noise, snr_label
from compressionkit.pipeline.stages import EntropyCoder

__all__ = [
    "EntropyBenchmarkResult",
    "evaluate_entropy_coder",
    "run_noise_sweep",
]


@dataclass
class EntropyBenchmarkResult:
    """One row of an entropy-coding scorecard."""

    coder_name: str
    condition: str
    """E.g. ``"clean"``, ``"12dB"``, or a caller-supplied validation-set tag."""
    num_tokens: int
    vocab_size: int
    bytes_used: int
    bits_per_token: float
    uniform_bits_per_token: float
    cr_uplift_vs_uniform: float
    roundtrip_ok: bool | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "coder_name": self.coder_name,
            "condition": self.condition,
            "num_tokens": self.num_tokens,
            "vocab_size": self.vocab_size,
            "bytes_used": self.bytes_used,
            "bits_per_token": self.bits_per_token,
            "uniform_bits_per_token": self.uniform_bits_per_token,
            "cr_uplift_vs_uniform": self.cr_uplift_vs_uniform,
            "roundtrip_ok": self.roundtrip_ok,
        }
        d.update(self.extra)
        return d


def evaluate_entropy_coder(
    coder: EntropyCoder,
    tokens: np.ndarray,
    *,
    vocab_size: int,
    condition: str = "default",
    verify_roundtrip: bool = True,
) -> EntropyBenchmarkResult:
    """Encode ``tokens`` with ``coder`` and report bpt / roundtrip correctness.

    Works with *any* :class:`EntropyCoder`-conforming object — an AI prior
    (:class:`~compressionkit.runtime.entropy_algorithms.LearnedEntropyCoder`),
    a classical baseline, or ``RawEntropy``/``DeflateEntropy``/``LzmaEntropy``
    from :mod:`compressionkit.pipeline.dsp_stages` — so results are directly
    comparable across algorithms.
    """
    tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
    bitstream, nbits = coder.encode(tokens)
    num_tokens = int(tokens.size)
    bpt = nbits / max(num_tokens, 1)
    uniform_bpt = float(np.log2(vocab_size))

    roundtrip_ok: bool | None = None
    if verify_roundtrip:
        decoded = coder.decode(bitstream, num_tokens)
        roundtrip_ok = bool(np.array_equal(tokens, np.asarray(decoded).reshape(-1)))

    return EntropyBenchmarkResult(
        coder_name=getattr(coder, "name", coder.__class__.__name__),
        condition=condition,
        num_tokens=num_tokens,
        vocab_size=vocab_size,
        bytes_used=len(bitstream),
        bits_per_token=bpt,
        uniform_bits_per_token=uniform_bpt,
        cr_uplift_vs_uniform=uniform_bpt / bpt if bpt > 0 else float("inf"),
        roundtrip_ok=roundtrip_ok,
    )


def run_noise_sweep(
    *,
    encode_to_tokens: Callable[[np.ndarray], np.ndarray],
    make_coder: Callable[[], EntropyCoder],
    clean_windows: np.ndarray,
    vocab_size: int,
    noise_bank: np.ndarray | None = None,
    snr_db_list: list[float | None] | None = None,
    seed: int = 0,
    verify_roundtrip: bool = True,
) -> list[EntropyBenchmarkResult]:
    """Evaluate an entropy coder across the same SNR ladder used to train/eval codecs.

    For each SNR level (``None`` = pristine/clean), injects real empirical
    noise via :func:`compressionkit.evaluation.empirical_regime.add_empirical_noise`
    (the same primitive used by the RVQ-vs-SPIHT robustness sweeps), runs the
    signal through ``encode_to_tokens`` (typically a frozen, already-trained
    codec's encoder — RVQ, SPIHT, Hybrid, whatever), and scores the resulting
    token/symbol stream with a fresh coder instance from ``make_coder``.

    Args:
        encode_to_tokens: Maps one ``(window_len,)`` (or batched) signal
            array to a flat int token/symbol array. Bring your own codec
            adapter here — this harness doesn't hardcode RVQ.
        make_coder: Factory returning a fresh :class:`EntropyCoder` instance
            per condition (important for stateful/fitted coders like
            :class:`~compressionkit.runtime.entropy_algorithms.StaticHistogramEntropyCoder`
            that should be re-fit — or reused unchanged — per caller's choice).
        clean_windows: ``(n_windows, window_len)`` float32 pristine signal.
        vocab_size: Token alphabet size (for the uniform-coding baseline).
        noise_bank: Real noise/residual segments; required unless
            ``snr_db_list`` is ``[None]`` (clean-only).
        snr_db_list: SNR levels in dB; ``None`` denotes clean. Defaults to
            ``compressionkit.evaluation.empirical_regime.DEFAULT_SNR_DB``.
        seed: Base seed for noise injection (offset per SNR level for
            reproducible-but-distinct draws).

    Returns:
        One :class:`EntropyBenchmarkResult` per SNR level.
    """
    if snr_db_list is None:
        from compressionkit.evaluation.empirical_regime import DEFAULT_SNR_DB

        snr_db_list = DEFAULT_SNR_DB

    results: list[EntropyBenchmarkResult] = []
    for i, snr_db in enumerate(snr_db_list):
        if snr_db is None:
            windows = clean_windows
        else:
            if noise_bank is None:
                raise ValueError(f"noise_bank is required for snr_db={snr_db} (only None/clean can skip it)")
            windows = add_empirical_noise(clean_windows, noise_bank, snr_db, seed=seed + i)

        tokens = np.concatenate([np.asarray(encode_to_tokens(w)).reshape(-1) for w in windows])
        coder = make_coder()
        result = evaluate_entropy_coder(
            coder,
            tokens,
            vocab_size=vocab_size,
            condition=snr_label(snr_db),
            verify_roundtrip=verify_roundtrip,
        )
        results.append(result)
    return results
