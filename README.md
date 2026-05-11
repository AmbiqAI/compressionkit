# CompressionKIT

**AI-powered compression toolkit for physiological signals targeting edge and wearable devices.**

CompressionKIT provides end-to-end pipelines for training, evaluating, and deploying neural compression codecs for continuous physiological signals (PPG, ECG). Models are designed for on-device inference on Ambiq-class MCUs via INT8 LiteRT, with server-side decoding via Keras.

📖 **Full documentation**: <https://ambiqai.github.io/compressionkit/>

---

## Features

- **PPG & ECG support** — 2× – 32× compression across both signal types with golden reference configs.
- **Edge-ready export** — INT8 quantized encoder (LiteRT + C header), RVQ codebook (C header + NumPy), decoder (`.keras`).
- **YAML-driven training** — modular configs for datasets, models, losses, evaluation, and export.
- **Physiologically-aware evaluation** — MSE, PRD, cosine similarity, plus HR/HRV preservation for PPG.
- **Modular architecture** — clean separation of datasets, preprocessing, models, evaluation, and export.

---

## Installation

```bash
# Requires Python 3.12 and uv (https://docs.astral.sh/uv/).
uv sync              # install runtime + dev dependencies
uv sync --group docs # optionally include doc site deps
```

Activate the environment with `source .venv/bin/activate`, or prefix any command with `uv run`.

---

## Quick Start

Train a golden model from a YAML config:

```bash
# PPG, 8× compression, MESA dataset
uv run train-ppg-rvq --config configs/ppg_rvq_64hz_08x_golden.yaml

# ECG, 8× compression, PTB-XL dataset
uv run train-ecg-rvq --config configs/ecg_rvq_256hz_08x_golden.yaml
```

Artifacts (INT8 TFLite, C header, codebook, Keras decoder, metrics JSON, plots) land under `results/<run_name>/`.

### Reproduce the full golden sweep

```bash
# Train all 5 PPG tiers (2× / 4× / 8× / 16× / 32×)
bash experiments/run_ppg_golden.sh

# Train all 5 ECG tiers
bash experiments/run_ecg_golden.sh

# Collect metrics into results/golden_summary.{csv,json}
# and regenerate docs/assets/plots/*.png
bash scripts/refresh_golden_plots.sh
```

All scripts are safe to run from any working directory — they `cd` to the repo root internally.

---

## Repository Layout

```
compressionkit/           # Python package (importable)
├── cli/                 # Console entry points (train-ppg-rvq, train-ecg-rvq)
├── configs/             # Pydantic run-config schemas
├── datasets/            # MESA (PPG) + PTB-XL (ECG) dataset classes, NSRR downloader
├── dsp/                 # Classical DSP primitives (STFT, DWT, SPIHT, wavelet codec)
├── evaluation/          # Metrics, overlap-add eval, quality tiers, artifact writers
├── export/              # TFLite / C-header / codebook / deployment pipeline
├── layers/              # VQ, RVQ, EMA-RVQ, Gumbel-Softmax Keras layers
├── models/              # RVQ autoencoder builders
├── preprocessing/       # Signal augmentation + conditioning
├── recipes/             # Golden end-to-end training recipes (PPG, ECG)
├── trainers/            # Reusable training building blocks + common helpers
├── callbacks/           # Custom Keras callbacks
└── logging/             # W&B integration

configs/                  # Versioned YAML run configs (golden + experiments)
recipes/                  # Active / experimental recipes (copy-and-edit playground)
scripts/                  # Package utilities (result collection, plotting, eval)
tests/                    # Pytest suite (smoke + unit tests)
experiments/              # Local sweep runners + one-off scripts (gitignored)
docs/                     # MkDocs Material / Zensical documentation
```

### Recipes: golden vs. active

- **Golden recipes** (`compressionkit/recipes/`) ship with the package and are what the CLI / console
  scripts invoke. Read one top-to-bottom to see the full training flow, step by step.
- **Active recipes** (`recipes/`) are source-controlled experimental scripts — copy a golden recipe,
  tweak it, iterate. Promote back into the package if it becomes broadly useful.

---

## Model Zoo

### PPG models (v1.0) — MESA, 64 Hz

| Model | Config | CR | PRD (%) | Cosine | HR MAE (bpm) |
|-------|--------|----|---------|--------|--------------|
| `ppg-rvq-02x` | [`ppg_rvq_64hz_02x_golden.yaml`](configs/ppg_rvq_64hz_02x_golden.yaml) | 1.78× | 3.32 | 0.9934 | 0.10 |
| `ppg-rvq-04x` | [`ppg_rvq_64hz_04x_golden.yaml`](configs/ppg_rvq_64hz_04x_golden.yaml) | 3.56× | 3.70 | 0.9908 | 0.11 |
| `ppg-rvq-08x` | [`ppg_rvq_64hz_08x_golden.yaml`](configs/ppg_rvq_64hz_08x_golden.yaml) | 7.11× | 6.17 | 0.9816 | 0.16 |
| `ppg-rvq-16x` | [`ppg_rvq_64hz_16x_golden.yaml`](configs/ppg_rvq_64hz_16x_golden.yaml) | 14.22× | 8.22 | 0.9693 | 0.22 |
| `ppg-rvq-32x` | [`ppg_rvq_64hz_32x_golden.yaml`](configs/ppg_rvq_64hz_32x_golden.yaml) | 28.44× | 11.19 | 0.9608 | 0.22 |

### ECG models (v1.0) — PTB-XL, 256 Hz

| Model | Config | CR | PRD (%) | Cosine |
|-------|--------|----|---------|--------|
| `ecg-rvq-02x` | [`ecg_rvq_256hz_02x_golden.yaml`](configs/ecg_rvq_256hz_02x_golden.yaml) | 1.78× | 2.60 | 0.9997 |
| `ecg-rvq-04x` | [`ecg_rvq_256hz_04x_golden.yaml`](configs/ecg_rvq_256hz_04x_golden.yaml) | 3.56× | 3.34 | 0.9994 |
| `ecg-rvq-08x` | [`ecg_rvq_256hz_08x_golden.yaml`](configs/ecg_rvq_256hz_08x_golden.yaml) | 7.11× | 6.21 | 0.9981 |
| `ecg-rvq-16x` | [`ecg_rvq_256hz_16x_golden.yaml`](configs/ecg_rvq_256hz_16x_golden.yaml) | 14.22× | 10.39 | 0.9945 |
| `ecg-rvq-32x` | [`ecg_rvq_256hz_32x_golden.yaml`](configs/ecg_rvq_256hz_32x_golden.yaml) | 28.44× | 14.39 | 0.9895 |

---

## Architecture

The core model is a **Residual Vector Quantized (RVQ) autoencoder**:

```
Input signal → Encoder (strided Conv2D) → RVQ Codebook → Decoder (UpSampling2D) → Reconstructed signal
                        │                      │
                INT8 TFLite + C header   C header + NumPy
                   (on-device)              (on-device)
```

Compression ratio is determined by the encoder downsampling factor and the number of RVQ levels. See the [methods docs](https://ambiqai.github.io/compressionkit/methods/rvq/) for full details.

---

## Datasets

| Dataset | Signal | Rate | Access | Notes |
|---------|--------|------|--------|-------|
| **MESA** | PPG | 256 Hz → 64 Hz | Restricted | Apply at [sleepdata.org/datasets/mesa](https://sleepdata.org/datasets/mesa). Set `NSRR_TOKEN=...` to download programmatically. |
| **PTB-XL** | ECG | 500 Hz → 256 Hz | Open (CC BY 4.0) | Auto-downloaded on first use. |

---

## Development

```bash
# Run the test suite
uv run pytest

# Lint + format
uv run ruff check .
uv run ruff format .

# Build the docs
uv run --group docs zensical build
```

The dev container (`.devcontainer/`) comes with Python 3.12, CUDA, `uv`, and all tooling preinstalled.

---

## Extending

1. **Add a new compression ratio** — copy an existing YAML in `configs/`, tweak `downsample_factor` / `num_levels`, and run `train-ppg-rvq --config ...`.
2. **Add a new signal type** — add a `Dataset` class under `compressionkit/datasets/`, a preprocessing module, reusable building blocks in `compressionkit/trainers/`, and a recipe in `compressionkit/recipes/` whose `main` is registered via `[project.scripts]` in `pyproject.toml`.
3. **Add a new layer or model variant** — drop a Keras layer into `compressionkit/layers/` and a model builder into `compressionkit/models/`; wire it via the relevant run config.

---

## License

Copyright © 2026 Ambiq AI. All rights reserved.
