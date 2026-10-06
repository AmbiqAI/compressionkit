---
title: "Getting Started"
description: "Getting Started: compressionKIT concepts, configuration and usage."
---


Start with the smallest path that answers your question. You do not need a
dataset to load a published codec, inspect its package, or run a synthetic
round-trip.

## 1. Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/). Run the commands below from a terminal.

For a source checkout:

```bash
git clone https://github.com/AmbiqAI/compressionkit.git
cd compressionkit
uv sync --python 3.12 --extra hf
```

Use the source checkout above while package publication is being finalized. The distribution name in project metadata is `compression-kit`; the Python import is `compressionkit`.

The `hf` extra is only needed when downloading HuggingFace bundles. Local deploy
packages can be loaded after they are already on disk.

## 2. Try a published codec

Published v1 RVQ bundles are available for PPG and ECG:

| Signal | HuggingFace repos | Frame |
|--------|-------------------|-------|
| PPG | `Ambiq/compressionkit-ppg-{2,4,8,16,32}x-v1.1` | 5 s at 64 Hz |
| ECG | `Ambiq/compressionkit-ecg-{2,4,8,16,32,64}x-v1.1` | 2 s at 256 Hz |

```python
import numpy as np

from compressionkit.runtime import load_codec

codec = load_codec("Ambiq/compressionkit-ppg-4x-v1.1")

t = np.arange(codec.frame_size, dtype=np.float32) / codec.sample_rate
frame = 0.6 * np.sin(2.0 * np.pi * 1.2 * t)

encoded = codec.compress(frame.astype(np.float32))
reconstructed = codec.decompress(encoded)

print(codec.modality, codec.target_cr, reconstructed.shape)
```

Save the example as `try_codec.py` in the checkout and run `uv run python try_codec.py`. It prints the signal type, ratio, and reconstruction shape; this is a loading check, not a quality evaluation.

For a fuller HuggingFace example using the bundled sample stimulus, see
[Load & test a HuggingFace model](/compressionkit/huggingface/).

## 3. Choose your next step

| Goal | Go here | Dataset required? |
|------|---------|-------------------|
| See quality and tradeoffs | [Model zoo](/compressionkit/models/), [PPG models](/compressionkit/models/ppg/), [ECG models](/compressionkit/models/ecg/) | No |
| Understand SPIHT / RVQ / hybrid behavior | [Methods](/compressionkit/methods/), [Validation Scorecard](/compressionkit/validation-scorecard/) | No |
| Run notebooks or synthetic examples | [Example Notebooks](/compressionkit/examples/) | No |
| Evaluate on your own waveform | [Dataset Setup · bring your own data](/compressionkit/datasets/#bring-your-own-data) | Your signal only |
| Validate a deploy package | [Deployment Guide](/compressionkit/deployment/#recommended-validation) | No |
| Reproduce a golden run | [Golden Experiments](/compressionkit/experiments/) | Yes |
| Build a new supported artifact | [Experiment Architecture](/compressionkit/experiment-architecture/), [V1 Release Contract](/compressionkit/release-contract/) | Usually |

## 4. When datasets matter

Datasets are only needed when you train, reproduce published metrics, or evaluate
on real recordings. The current release uses:

| Signal | Release dataset surface | Notes |
|--------|-------------------------|-------|
| PPG | `ppg-unified-strict-sanitize-v1` | Open-source PPG cache built from BIDMC, BUT PPG, PPG-DaLiA, and WESAD |
| ECG | `ptb-xl` | PTB-XL Lead II windows resampled to 256 Hz |

See [Dataset Setup](/compressionkit/datasets/) for cache layout, licensing, and synthetic
generators.

## 5. Validate before deployment

Before integrating a codec into a product workflow, check the package boundary:

```bash
uv run compressionkit golden validate-deploy results/ppg_rvq_64hz_08x_golden/deploy --max-vectors 1
```

For runtime packaging, HuggingFace naming, and split encoder/decoder deployment,
see [Deployment Guide](/compressionkit/deployment/).
