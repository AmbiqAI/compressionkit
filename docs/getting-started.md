---
icon: lucide/play
---

# Getting Started

Start with the smallest path that answers your question. You do not need a
dataset to load a published codec, inspect its package, or run a synthetic
round-trip.

## 1. Install

For a source checkout:

```bash
git clone https://github.com/AmbiqAI/compressionkit.git
cd compressionkit
uv sync --extra hf
```

For a package install:

```bash
uv pip install "compressionkit[hf]"
```

The `hf` extra is only needed when downloading HuggingFace bundles. Local deploy
packages can be loaded after they are already on disk.

## 2. Try a published codec

Published v1 RVQ bundles are available for PPG and ECG:

| Signal | HuggingFace repos | Frame |
|--------|-------------------|-------|
| PPG | `Ambiq/compressionkit-ppg-{2,4,8,16,32}x` | 5 s at 64 Hz |
| ECG | `Ambiq/compressionkit-ecg-{2,4,8,16,32,64}x` | 2 s at 256 Hz |

```python
import numpy as np

from compressionkit.runtime import load_codec

codec = load_codec("Ambiq/compressionkit-ppg-4x-v1.0")

t = np.arange(codec.frame_size, dtype=np.float32) / codec.sample_rate
frame = 0.6 * np.sin(2.0 * np.pi * 1.2 * t)

encoded = codec.compress(frame.astype(np.float32))
reconstructed = codec.decompress(encoded)

print(codec.modality, codec.compression_ratio, reconstructed.shape)
```

For a fuller HuggingFace example using the bundled sample stimulus, see
[Load & test a HuggingFace model](huggingface.md).

## 3. Choose your next step

| Goal | Go here | Dataset required? |
|------|---------|-------------------|
| See quality and tradeoffs | [Model zoo](models/index.md), [PPG models](models/ppg.md), [ECG models](models/ecg.md) | No |
| Understand SPIHT / RVQ / hybrid behavior | [Methods](methods/index.md), [Validation Scorecard](validation-scorecard.md) | No |
| Run notebooks or synthetic examples | [Example Notebooks](examples.md) | No |
| Evaluate on your own waveform | [Dataset Setup · bring your own data](datasets.md#bring-your-own-data) | Your signal only |
| Validate a deploy package | [Deployment Guide](deployment.md#recommended-validation) | No |
| Reproduce a golden run | [Golden Experiments](experiments/index.md) | Yes |
| Build a new supported artifact | [Experiment Architecture](experiment-architecture.md), [V1 Release Contract](release-contract.md) | Usually |

## 4. When datasets matter

Datasets are only needed when you train, reproduce published metrics, or evaluate
on real recordings. The current release uses:

| Signal | Release dataset surface | Notes |
|--------|-------------------------|-------|
| PPG | `ppg-unified-strict-sanitize-v1` | Open-source PPG cache built from BIDMC, BUT PPG, PPG-DaLiA, and WESAD |
| ECG | `ptb-xl` | PTB-XL Lead II windows resampled to 256 Hz |

See [Dataset Setup](datasets.md) for cache layout, licensing, and synthetic
generators.

## 5. Validate before deployment

Before integrating a codec into a product workflow, check the package boundary:

```bash
uv run compressionkit golden validate-deploy results/ppg_rvq_64hz_08x_golden/deploy --max-vectors 1
```

For runtime packaging, HuggingFace naming, and split encoder/decoder deployment,
see [Deployment Guide](deployment.md).
