---
icon: lucide/binary
---

# Entropy-Coding Research Framework

This page applies the general [Experiment Architecture](experiment-architecture.md)
(Blocks → Ready-made experiments → Golden releases) specifically to the
entropy-coding stage of the codec pipeline (`preprocess -> transform ->
encoder -> entropy`). It exists so the research workflow for trying new
entropy coders/priors is documented once, in the repo, instead of being
re-explained in every session.

## Why this layer exists

The entropy stage is a **separate, swappable, lossless second stage** on top
of an already-trained, frozen codec (RVQ, SPIHT, Hybrid). It never touches
reconstruction quality — only how compactly the codec's own output symbols
are packed into bits. That makes it a low-risk, high-value place to
experiment: a new entropy algorithm can be benchmarked against the existing
default using the *same* frozen codec and *same* validation data, with zero
risk to shipped quality metrics.

## Blocks

| Block | Module | Purpose |
|---|---|---|
| `EntropyCoder` protocol | `compressionkit.pipeline.stages` | The narrow interface (`encode(symbols)->(bytes,nbits)`, `decode(bitstream,n)->symbols`) every entropy algorithm implements — classical or learned. |
| `RawEntropy`, `DeflateEntropy`, `LzmaEntropy` | `compressionkit.pipeline.dsp_stages` | Classical baselines, already conform to `EntropyCoder`. |
| `LearnedEntropyCoder` | `compressionkit.runtime.entropy_algorithms` | Wraps any `PriorLike` model (vocab_size, context_length, predict_next_probs) + a bit-packing backend (`"arithmetic"` or `"rans"`). Conforms to `EntropyCoder`. |
| `StaticHistogramEntropyCoder`, `MarkovEntropyCoder` | `compressionkit.runtime.entropy_algorithms` | Non-AI classical baselines (order-0 / order-1), same backends as the learned coder, so comparisons isolate the *probability source*, not the bit-packer. |
| `build_wavenet_prior`, `build_gru_prior`, `build_cnn_prior`, `build_dscnn_prior`, `build_hybrid_prior`, `build_cnngru_prior` | `compressionkit.generative.causal_priors` | Causal token-prior architectures. `build_wavenet_prior` (default `num_layers=6`) is the empirically-strongest architecture found so far — prefer it unless a deployment constraint calls for something smaller. |
| `_arithmetic_encode`/`_rans_encode` + decode counterparts | `compressionkit.runtime.two_stage` | The actual bit-packing implementations backing `LearnedEntropyCoder`. |
| `EntropyBenchmarkResult`, `evaluate_entropy_coder`, `run_noise_sweep` | `compressionkit.evaluation.entropy_benchmark` | Scorecard-style evaluation for *any* `EntropyCoder`, including sweeping across the same SNR/noise-bank conditions used to train the parent codec (reuses `compressionkit.evaluation.empirical_regime`). |
| `build_datasets` (per modality) | `compressionkit.trainers.{ecg_rvq,ppg_rvq}` | The parent codec's own dataset + augmentation pipeline. **Reuse this for prior training data** (see Known Lessons below) instead of hand-rolling noise injection. |

## Ready-made experiments

- `scripts/measure_rvq_entropy.py` — research sweep script: architecture comparison, hyperparameter sweeps, entropy measurement. Not registry-integrated; a readable end-to-end flow for one-off experiments. Supports `--lr-schedule cosine` and `--steps-per-epoch` for decoupling epoch granularity from full-dataset-size (mirrors the parent codec's own `steps_per_epoch`/`epochs` convention).
- `compressionkit.trainers.rvq_prior.train_prior_from_config` (+ `compressionkit.configs.rvq_prior.RvqPriorConfig`) — the more "production-shaped" trainer: resolves the parent codec from the golden registry, writes `<parent_run_dir>/prior/prior.weights.h5`, and can export `prior_int8.tflite` + a manifest into the parent's `deploy/` directory. **As of writing this trainer is mid-upgrade** — see Known Lessons.

## Golden promotion path

Mirrors [Adding a Codec Family](adding-a-codec-family.md): an entropy prior
becomes "golden" by producing and validating artifacts, not by being
rewritten around a mandatory base class.

1. Prove the new prior/coder beats the current default using
   `evaluate_entropy_coder`/`run_noise_sweep` on the **same** parent codec's
   real validation data (and, ideally, the same noise conditions it was
   trained under).
2. Train via `train_prior_from_config` with `export_tflite=True` to produce
   `prior_int8.tflite` + `prior_manifest.json` in the parent's `deploy/`.
3. Validate the artifact the same way any other deploy artifact is validated
   (`compressionkit.export.validate.validate_deploy_package`).
4. Register/update the golden entry so the two-stage codec (parent + prior)
   is reproducible from config.

## Known lessons (read before re-deriving these)

- **Single-level bug**: `train_prior_from_config` historically trained on
  `tokens[..., 0]` only — silently discarding all but RVQ level 0. Every
  shipped golden uses `num_levels=2`. Fix: interleave *all* levels
  (level-major intra-step order), matching `measure_rvq_entropy.py`'s
  `_interleave_levels` convention, before windowing.
- **Missing LR schedule**: the prior trained at a flat LR with no decay,
  which plateaus well before it should. The parent codec always anneals via
  `CosineDecayRestarts` anchored to its *own* total step budget
  (`first_decay_steps` == `steps_per_epoch * epochs`, which incidentally
  means the codec's own schedule never actually restarts either — same
  config pattern, same caveat). Mirror this for the prior.
- **"Epoch" is not a comparable unit across configs.** The parent codec
  decouples "epoch" from dataset size (`steps_per_epoch=200` is a fixed,
  small logging/checkpoint chunk, not a full data pass). A naive
  `model.fit(x, y, epochs=N)` call makes one "epoch" a *full pass* over
  however many windows exist — a completely different unit. Always compare
  **total steps** (`steps_per_epoch * epochs`), not epoch counts, when
  judging whether two training runs are comparable.
- **Reuse the codec's own augmented training pipeline, don't hand-roll
  noise injection.** `compressionkit.trainers.ecg_rvq.build_datasets(cfg,
  preprocessor, augmenter)` transparently reuses the existing pre-built
  TFRecord cache (`data.cache.cache_root`) via `force_rebuild=False` — no
  wasted re-extraction cycles — and guarantees the prior sees *exactly* the
  windowing/crop/normalization/augmentation distribution the codec itself
  trained on. Confirmed empirically: ECG's Gaussian-noise augmenter changes
  ~23% of RVQ token assignments vs. clean encoding of the same files — a
  real, meaningful effect, not a no-op.
- **PPG's golden configs currently ship with augmentation disabled**
  (`gaussian_noise=[0.0,0.0]`) — no synthetic noise injection is needed for
  the PPG prior; real wearable data already carries natural noise, so more
  unique training examples (not synthetic augmentation) is the right lever.
- **Validate on genuinely held-out files, not a random window split of the
  same extracted pool.** A random 90/10 split of *windows* (not files) can
  leak near-duplicate overlapping context between "train" and "val" from
  the same underlying recording. Extract validation from separate files.
- **Entropy-prior compression uplift is not free at every operating point.**
  It's largest at low compression ratios (codec leaves more redundancy
  behind) and shrinks under input noise (noise both raises token-stream
  entropy and specifically undermines the prior's context-dependent
  advantage). Don't assume a fixed uplift number transfers across CR tiers
  or SNR conditions — measure it via `run_noise_sweep` for the actual
  operating regime you care about.
