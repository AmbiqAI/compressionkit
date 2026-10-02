# compressionKIT

**Neural codecs for wearable and physiological signals.**

compressionKIT helps teams compress continuous sensor waveforms for edge and wearable systems: PPG, ECG, IMU, and other time-series signals where raw data movement dominates memory, radio energy, and cloud cost.

The toolkit is built around reusable codec blocks rather than one fixed architecture. A codec can combine learned encoders, quantizers, entropy models, DSP stages, runtime loaders, deploy exporters, and validation scorecards. The current v1 release artifacts focus on PPG and ECG, but the package is intended to be extended to new wearable modalities and codec families.

[![Docs](https://img.shields.io/badge/docs-ambiqai.github.io-blue)](https://ambiqai.github.io/compressionkit/)
[![HuggingFace](https://img.shields.io/badge/HF-Ambiq%2Fcompressionkit--*-yellow)](https://huggingface.co/Ambiq)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

> **Pre-v1.** APIs, configs, and release conventions may shift on major versions until v1.

---

## What compressionKIT provides

- **Codec building blocks** for wearable time-series compression: models, quantizers, losses, preprocessing, evaluation, scorecards, export, and runtime loading.
- **Edge deploy artifacts** such as LiteRT/TFLite models, C headers, codebooks, manifests, checksums, reference vectors, and HuggingFace bundles.
- **Release-grade validation** at the artifact boundary, so a codec can be checked without re-running training.
- **Physiology-aware evaluation** for PPG and ECG, with room to add modality-specific metrics for IMU and future signals.
- **Progressive workflows**: try a published codec first, validate an existing deploy package next, then reproduce or extend experiments when needed.

compressionKIT is not meant to force every experiment into one framework. Start with the blocks you need, then opt into the golden release workflow only when a result should become a supported artifact.

---

## Quickstart

Install the runtime and HuggingFace helper:

```bash
uv pip install "compressionkit[hf]"

# Or, working from this repository:
uv sync --extra hf
```

Load a published v1 codec and round-trip a frame. No training dataset is required for this path.

```python
import numpy as np
from compressionkit.runtime import load_codec

codec = load_codec("Ambiq/compressionkit-ppg-4x-v1.0")

frame = np.zeros(codec.frame_size, dtype=np.float32)  # replace with your signal frame
encoded = codec.compress(frame)
reconstructed = codec.decompress(encoded)

print(codec.modality, codec.sample_rate, encoded.nbits, reconstructed.shape)
```

For RVQ-specific workflows that need raw token indices:

```python
from compressionkit.runtime import RVQCodec

codec = RVQCodec.from_pretrained("Ambiq/compressionkit-ecg-4x-v1.0")
indices = codec.encode(signal)
reconstructed = codec.decode(indices)
```

Prefer notebooks? Start with:

- [`examples/01_quickstart_golden_codec.ipynb`](examples/01_quickstart_golden_codec.ipynb)
- [`examples/02_evaluate_on_your_data.ipynb`](examples/02_evaluate_on_your_data.ipynb)

---

## Current v1 release surface

The v1 golden packages are the supported examples of the broader toolkit:

- **Published neural codec bundles** for PPG and ECG at multiple compression ratios.
- **Local reproducible DSP and hybrid baselines** used for comparison, packaging, and deployment experiments.
- **Deploy manifests and validation artifacts** that make packages self-describing and testable.
- **Scorecards** that report signal fidelity, physiological metric preservation, and robustness behavior.

The currently shipped methods include RVQ neural autoencoders, SPIHT-style DSP packages, and hybrid experiments. They are not the only intended options; they are the first release-grade lanes built on the common codec and artifact contract.

For details, see:

- [Golden experiments](https://ambiqai.github.io/compressionkit/experiments/) for reproducible release runs.
- [Model zoo](https://ambiqai.github.io/compressionkit/models/) for published PPG and ECG bundles.
- [Experiment architecture](https://ambiqai.github.io/compressionkit/experiment-architecture/) for the block-first extension model.
- [Validation scorecard](https://ambiqai.github.io/compressionkit/validation-scorecard/) for metric rationale.
- [Deployment guide](https://ambiqai.github.io/compressionkit/deployment/) for package contents and runtime integration.

---

## Reproducibility ladder

Choose the lightest path that answers your question:

| Goal | Command or API | Dataset required? |
|------|----------------|-------------------|
| Try a published codec | `load_codec("Ambiq/compressionkit-ppg-4x-v1.0")` | No |
| Validate a local deploy package | `uv run compressionkit golden validate-deploy results/.../deploy` | No |
| Stage a HuggingFace package | `uv run compressionkit golden run <id> --skip-train --publish --dry-run` | No, if `deploy/` exists |
| Reproduce a golden run | `uv run compressionkit golden run <id>` | Yes |
| Extend an experiment | Copy a config or script, change one decision, export, validate | Usually |
| Promote to golden | Add registry entry, frozen config, scorecard, docs, and publication target | Yes |

Training and real-data evaluation read datasets from `./datasets` by default. Override the root with `COMPRESSIONKIT_DATASETS_DIR`. See the [dataset setup guide](https://ambiqai.github.io/compressionkit/datasets/) for layout and licensing.

---

## Extend compressionKIT

A custom codec does not need to start in the golden registry. Typical extension work looks like this:

1. Reuse dataset, preprocessing, model, loss, export, or validation blocks directly.
2. Copy the closest config or ready-made recipe.
3. Change one decision at a time: modality, frame size, encoder, quantizer, entropy model, loss, scorecard, or deployment target.
4. Export a deploy package.
5. Validate the artifact contract.
6. Promote the result to a golden only after the artifact, scorecard, and docs are stable.

This keeps simple paths simple while leaving room for new neural codecs, new sensor modalities, and new deployment targets.

---

## Repository layout

```text
compressionkit/         # Importable Python package
├── configs/            # Pydantic config schemas and path helpers
├── datasets/           # Dataset contracts and loaders
├── dsp/                # DSP primitives and reference codecs
├── evaluation/         # Metrics, robustness checks, scorecards
├── experiments/        # Golden registry and lifecycle CLI
├── export/             # Deploy packaging, validation, model cards
├── layers/             # Keras layers for codec models
├── models/             # Neural codec architectures
├── pipeline/           # Composable codec stages
├── recipes/            # Ready-made training flows
├── runtime/            # Runtime loaders and codec interfaces
└── synthetic/          # Synthetic signal and artifact generation

configs/                # Versioned YAML configs
astro-site/             # Astro documentation site
examples/               # Notebook quickstarts
scripts/                # Release, evaluation, and plotting utilities
tests/                  # pytest suite
```

---

## Development

```bash
uv sync --group dev --extra hf
uv run ruff check compressionkit/ tests/ scripts/
uv run ruff format compressionkit/ tests/ scripts/
uv run pytest tests/ -ra
npm --prefix astro-site ci
npm --prefix astro-site run build
```

---

## License

Source code is licensed under the [BSD 3-Clause License](LICENSE). Model weights, codebooks, and published deployment artifacts may carry additional Ambiq model-weight terms; see [LICENSE-MODEL-WEIGHTS.md](LICENSE-MODEL-WEIGHTS.md).
