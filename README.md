# compressionKIT

**Edge-grade compression for physiological signals (PPG, ECG).** Three flavors,
one runtime contract:

- **DSP-only** — wavelet + SPIHT + arithmetic coding. No trained weights,
  Apache-licensed C99 reference, ideal for the most constrained MCUs at
  modest CRs.
- **AI-only** — Residual Vector Quantization (RVQ) autoencoders, INT8
  TFLite, the deepest CRs at the best quality.
- **DSP+AI hybrid** — DSP front-end + small neural prior, sweet spot for
  low-CR / low-energy deployments.

[![Docs](https://img.shields.io/badge/docs-ambiqai.github.io-blue)](https://ambiqai.github.io/compressionkit/)
[![HuggingFace](https://img.shields.io/badge/HF-Ambiq%2Fcompressionkit--*-yellow)](https://huggingface.co/Ambiq)
[![License](https://img.shields.io/badge/license-Ambiq%20Silicon%20Only-green)](LICENSE-MODEL-WEIGHTS.md)

Every release ships a registry of [golden experiments](https://ambiqai.github.io/compressionkit/experiments/),
each reproducible with a single command and published to HuggingFace.

> **Pre-v1.** APIs and configs may shift on major versions until v1.

---

## Quickstart

```bash
uv pip install compressionkit                # runtime
# or, working from this repo:
uv sync
```

Load any golden codec from HuggingFace and round-trip a sample frame — the
loader dispatches on the deploy manifest's `family` field, so the same code
works for DSP-only SPIHT, AI-only RVQ, and hybrid packages:

```python
from compressionkit.runtime import load_codec
import numpy as np

codec = load_codec("Ambiq/compressionkit-ppg-4x")          # AI-only RVQ
# codec = load_codec("Ambiq/compressionkit-ppg-spiht-4x")  # DSP-only SPIHT

frame = ...  # (frame_size,) float32
enc = codec.compress(frame)
recon = codec.decompress(enc)
```

The legacy RVQ-native API is still available for code that wants raw indices:

```python
from compressionkit.runtime import RVQCodec
codec = RVQCodec.from_pretrained("Ambiq/compressionkit-ppg-4x")
indices = codec.encode(signal)
recon = codec.decode(indices)
```

See the [HuggingFace testing guide](https://ambiqai.github.io/compressionkit/huggingface/) for
two-stage (codec + entropy prior) usage.

---

## DSP-only (SPIHT) quickstart

DSP-only SPIHT codecs ship as **weightless** deploy packages: a deploy
manifest, codec parameters, license-safe synthetic stimulus, reference
encode/decode vectors, and the portable C99 reference under
`c_sources/` with a generated `spiht_app_config.h`.

```bash
# Build the weightless deploy package locally (no dataset required).
uv run compressionkit golden run ppg-spiht-4x

# Optional: publish to HuggingFace (Apache-2.0, no proprietary weights).
uv run compressionkit golden run ppg-spiht-4x --publish
```

Python:

```python
from compressionkit.runtime import load_codec

codec = load_codec("Ambiq/compressionkit-ppg-spiht-4x")
enc = codec.compress(frame)   # frame: (frame_size,) float32
recon = codec.decompress(enc)
```

C (embedded integration):

```c
#include "spiht_app_config.h"

float frame[APP_SPIHT_FRAME_SIZE];
uint8_t bitstream[APP_SPIHT_MAX_BYTES];
/* ... fill frame from sensor ... */
size_t nbits = spiht_encode_frame(&enc, bitstream, APP_SPIHT_MAX_BITS);
```

---

## Reproduce a golden experiment

The lifecycle runner dispatches on the experiment's `method` field:

- `method == "rvq"`: dataset pre-flight → training → evaluation → export → optional publish.
- `method == "spiht"`: build codec → write weightless deploy package → optional publish.

```bash
# Single experiment.
uv run compressionkit golden run ppg-rvq-4x
uv run compressionkit golden run ppg-spiht-4x

# Whole modality.
uv run compressionkit golden run-all --modality ppg

# Train + publish to HuggingFace (requires HF_TOKEN).
uv run compressionkit golden run ppg-rvq-4x --publish
```

Outputs land under `results/<run_name>/`, with the publishable deploy package in
`results/<run_name>/deploy/`.

---

## Model Zoo

### PPG · MESA · 64 Hz

| Experiment | CR | PRD (%) | Cosine | HR MAE (bpm) | HuggingFace |
|------------|----|---------|--------|--------------|-------------|
| [`ppg-rvq-2x`](https://ambiqai.github.io/compressionkit/experiments/ppg-rvq-2x/)  | 2×  | 2.99 | 0.9921 | 0.10 | [`Ambiq/compressionkit-ppg-2x`](https://huggingface.co/Ambiq/compressionkit-ppg-2x) |
| [`ppg-rvq-4x`](https://ambiqai.github.io/compressionkit/experiments/ppg-rvq-4x/)  | 4×  | 3.96 | 0.9887 | 0.09 | [`Ambiq/compressionkit-ppg-4x`](https://huggingface.co/Ambiq/compressionkit-ppg-4x) |
| [`ppg-rvq-8x`](https://ambiqai.github.io/compressionkit/experiments/ppg-rvq-8x/)  | 8×  | 6.95 | 0.9688 | 0.19 | [`Ambiq/compressionkit-ppg-8x`](https://huggingface.co/Ambiq/compressionkit-ppg-8x) |
| [`ppg-rvq-16x`](https://ambiqai.github.io/compressionkit/experiments/ppg-rvq-16x/) | 16× | 8.60 | 0.9688 | 0.20 | [`Ambiq/compressionkit-ppg-16x`](https://huggingface.co/Ambiq/compressionkit-ppg-16x) |
| [`ppg-rvq-32x`](https://ambiqai.github.io/compressionkit/experiments/ppg-rvq-32x/) | 32× | 11.26 | 0.9606 | 0.30 | [`Ambiq/compressionkit-ppg-32x`](https://huggingface.co/Ambiq/compressionkit-ppg-32x) |

### ECG · PTB-XL · 256 Hz (Lead II)

| Experiment | CR | PRD (%) | Cosine | HuggingFace |
|------------|----|---------|--------|-------------|
| [`ecg-rvq-2x`](https://ambiqai.github.io/compressionkit/experiments/ecg-rvq-2x/)  | 2×  | 2.59  | 0.9997 | [`Ambiq/compressionkit-ecg-2x`](https://huggingface.co/Ambiq/compressionkit-ecg-2x) |
| [`ecg-rvq-4x`](https://ambiqai.github.io/compressionkit/experiments/ecg-rvq-4x/)  | 4×  | 3.68  | 0.9993 | [`Ambiq/compressionkit-ecg-4x`](https://huggingface.co/Ambiq/compressionkit-ecg-4x) |
| [`ecg-rvq-8x`](https://ambiqai.github.io/compressionkit/experiments/ecg-rvq-8x/)  | 8×  | 6.66  | 0.9978 | [`Ambiq/compressionkit-ecg-8x`](https://huggingface.co/Ambiq/compressionkit-ecg-8x) |
| [`ecg-rvq-16x`](https://ambiqai.github.io/compressionkit/experiments/ecg-rvq-16x/) | 16× | 11.02 | 0.9938 | [`Ambiq/compressionkit-ecg-16x`](https://huggingface.co/Ambiq/compressionkit-ecg-16x) |
| [`ecg-rvq-32x`](https://ambiqai.github.io/compressionkit/experiments/ecg-rvq-32x/) | 32× | 15.53 | 0.9878 | [`Ambiq/compressionkit-ecg-32x`](https://huggingface.co/Ambiq/compressionkit-ecg-32x) |
| [`ecg-rvq-64x`](https://ambiqai.github.io/compressionkit/experiments/ecg-rvq-64x/) | 64× | —     | —      | [`Ambiq/compressionkit-ecg-64x`](https://huggingface.co/Ambiq/compressionkit-ecg-64x) |

### Two-stage (codec + entropy prior)

Selected tiers ship a small causal-transformer entropy prior that lossless-codes the codec's
token stream for additional CR uplift. The prior bundles alongside the codec in the same
HuggingFace repo (`prior_int8.tflite`).

- [`ppg-rvq-4x-prior`](https://ambiqai.github.io/compressionkit/experiments/ppg-rvq-4x-prior/) ·
  [`ppg-rvq-8x-prior`](https://ambiqai.github.io/compressionkit/experiments/ppg-rvq-8x-prior/)
- [`ecg-rvq-4x-prior`](https://ambiqai.github.io/compressionkit/experiments/ecg-rvq-4x-prior/) ·
  [`ecg-rvq-8x-prior`](https://ambiqai.github.io/compressionkit/experiments/ecg-rvq-8x-prior/)

### DSP-only (SPIHT)

Weightless codec packages built from wavelet (`coif5` for PPG, `bior4.4` for
ECG, both L=6) + SPIHT + arithmetic coding. No trained weights, Apache-2.0
licensed, distributed with portable C99 source for direct MCU integration.

- `ppg-spiht-2x` · `ppg-spiht-4x` · `ppg-spiht-8x` → `Ambiq/compressionkit-ppg-spiht-{2,4,8}x`
- `ecg-spiht-2x` · `ecg-spiht-4x` · `ecg-spiht-8x` → `Ambiq/compressionkit-ecg-spiht-{2,4,8}x`

---

## When to use which tier

| Use case | Recommendation |
|----------|----------------|
| Real-time streaming over BLE | 2×–4× (best fidelity, lowest decoder cost) |
| On-device store-and-forward  | 8×–16× (CR-fidelity sweet spot) |
| Long-term archival / cold storage | 32×–64× or two-stage variants |

Full evaluation methodology lives in the
[evaluation reference](https://ambiqai.github.io/compressionkit/api/evaluation/).

---

## Repository Layout

```
compressionkit/         # Importable Python package
├── cli/                # Console entry points (compressionkit, train-*-rvq, golden subcommand)
├── configs/            # Pydantic run-config schemas
├── datasets/           # MESA (PPG), PTB-XL (ECG), dataset contract
├── evaluation/         # Metrics, overlap-add eval, scorecards
├── experiments/        # Golden registry + lifecycle runner
├── export/             # TFLite, C-header, codebook, deploy packager
├── layers/             # VQ / RVQ / EMA-RVQ Keras layers
├── models/             # RVQ autoencoder + entropy prior
├── recipes/            # Golden training recipes
└── runtime/            # Inference runtime (RVQCodec, EntropyPrior, TwoStageCodec)

configs/                # Versioned YAML configs for every golden experiment
scripts/                # Utilities (HF publish, doc rendering, etc.)
docs/                   # Zensical documentation site
tests/                  # pytest suite
```

---

## Development

```bash
uv run pytest -q
uv run ruff check
uv run ruff format compressionkit tests scripts
uv run --group docs zensical build
```

The dev container ships Python 3.12, CUDA, `uv`, and all tooling preinstalled.

---

## License

Apache 2.0. Copyright © 2026 Ambiq AI.
