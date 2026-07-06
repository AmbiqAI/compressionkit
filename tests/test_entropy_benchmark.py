"""Tests for the entropy-coding benchmark harness (compressionkit.evaluation.entropy_benchmark)."""

from __future__ import annotations

import numpy as np

from compressionkit.evaluation.entropy_benchmark import evaluate_entropy_coder, run_noise_sweep
from compressionkit.runtime.entropy_algorithms import LearnedEntropyCoder, StaticHistogramEntropyCoder


class _MockPrior:
    def __init__(self, vocab_size: int, context_length: int = 32) -> None:
        self.vocab_size = vocab_size
        self.context_length = context_length

    def predict_next_probs(self, context_tokens: np.ndarray) -> np.ndarray:
        batch = context_tokens.shape[0]
        return np.full((batch, self.vocab_size), 1.0 / self.vocab_size, dtype=np.float32)


class TestEvaluateEntropyCoder:
    def test_uniform_prior_matches_uniform_baseline(self):
        vocab = 16
        rng = np.random.default_rng(0)
        tokens = rng.integers(0, vocab, size=1000).astype(np.int32)
        coder = LearnedEntropyCoder(prior=_MockPrior(vocab), backend="rans")

        result = evaluate_entropy_coder(coder, tokens, vocab_size=vocab)

        assert result.num_tokens == 1000
        assert result.roundtrip_ok is True
        assert result.uniform_bits_per_token == 4.0
        # Uniform-probability coder should land close to the uniform baseline
        # (small residual gap from rANS's fixed state-flush overhead).
        assert abs(result.bits_per_token - 4.0) < 0.2
        assert 0.8 < result.cr_uplift_vs_uniform < 1.2

    def test_skewed_static_histogram_beats_uniform(self):
        vocab = 64
        rng = np.random.default_rng(3)
        raw = np.array([0.7**k for k in range(vocab)])
        p = raw / raw.sum()
        tokens = rng.choice(vocab, size=500, p=p).astype(np.int32)
        coder = StaticHistogramEntropyCoder.fit(tokens, vocab_size=vocab)

        result = evaluate_entropy_coder(coder, tokens, vocab_size=vocab, condition="clean")

        assert result.condition == "clean"
        assert result.roundtrip_ok is True
        assert result.cr_uplift_vs_uniform > 1.5  # meaningfully better than uniform


class TestRunNoiseSweep:
    def test_sweep_across_snr_with_synthetic_noise(self):
        vocab = 16
        rng = np.random.default_rng(4)
        window_len = 64
        n_windows = 20
        clean_windows = rng.normal(size=(n_windows, window_len)).astype(np.float32)
        noise_bank = np.stack([rng.normal(size=window_len).astype(np.float32) for _ in range(10)])

        def encode_to_tokens(window: np.ndarray) -> np.ndarray:
            # Trivial deterministic "codec": bucket amplitude into vocab bins.
            scaled = np.clip((window + 4.0) / 8.0, 0.0, 1.0)
            return (scaled * (vocab - 1)).astype(np.int32)

        def make_coder() -> LearnedEntropyCoder:
            return LearnedEntropyCoder(prior=_MockPrior(vocab), backend="rans")

        results = run_noise_sweep(
            encode_to_tokens=encode_to_tokens,
            make_coder=make_coder,
            clean_windows=clean_windows,
            vocab_size=vocab,
            noise_bank=noise_bank,
            snr_db_list=[None, 12.0, 0.0],
            seed=0,
        )

        assert [r.condition for r in results] == ["clean", "12dB", "0dB"]
        for r in results:
            assert r.roundtrip_ok is True
            assert r.num_tokens == n_windows * window_len

    def test_clean_only_does_not_require_noise_bank(self):
        vocab = 8
        clean_windows = np.zeros((5, 32), dtype=np.float32)

        def encode_to_tokens(window: np.ndarray) -> np.ndarray:
            return np.zeros(window.shape[0], dtype=np.int32)

        results = run_noise_sweep(
            encode_to_tokens=encode_to_tokens,
            make_coder=lambda: LearnedEntropyCoder(prior=_MockPrior(vocab), backend="rans"),
            clean_windows=clean_windows,
            vocab_size=vocab,
            snr_db_list=[None],
        )
        assert len(results) == 1
        assert results[0].condition == "clean"
