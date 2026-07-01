---
icon: lucide/radio
---
#

[![](./assets/compressionkit-light.png#only-light)](.)
[![](./assets/compressionkit-dark.png#only-dark)](.)

**Neural codecs for wearable and physiological signals on edge devices.**

compressionKIT helps teams compress continuous sensor waveforms before they become a memory, radio, battery, or cloud-ingestion problem. It is designed for wearable and edge signals such as **PPG**, **ECG**, **IMU**, and future time-series modalities where preserving waveform utility matters.

The package provides reusable codec blocks, deploy packaging, runtime loading, and validation scorecards. The v1 release artifacts focus on PPG and ECG, but the architecture is intentionally broader than any single method or modality.

<div class="ck-hero-links" markdown="1">

- [Get started](getting-started.md)
- [Try HuggingFace bundles](huggingface.md)
- [View model zoo](models/index.md)
- [Understand experiments](experiments/index.md)

</div>

---

## What compressionKIT is

compressionKIT is a toolkit for building, evaluating, packaging, and deploying compression codecs for wearable signals.

It includes:

- neural codec models and quantization components
- DSP and hybrid codec stages where they are useful
- preprocessing and augmentation blocks
- physiological and signal-fidelity scorecards
- deploy package generation with manifests, checksums, reference vectors, and C headers
- runtime loaders for local and HuggingFace artifacts
- golden release automation for supported PPG and ECG operating points

It is not a general physiological inference framework. It does not try to own arrhythmia classification, sleep staging, activity recognition, or downstream clinical decisions. It focuses on the codec layer: reducing data movement while preserving the signal content downstream systems need.

!!! note "Current release scope"
    The current v1 release surface publishes neural codec bundles for PPG and ECG and includes reproducible DSP/hybrid comparison lanes. Those methods are examples of the codec framework, not the boundary of what compressionKIT can support.

---

## Why it matters

<div class="grid cards" markdown="1">

-   :material-watch-variant:{ .lg .middle } **Wearable-first compression**

    ---

    Reduce flash, PSRAM, BLE, cellular, and cloud-storage pressure by compressing the waveform at the sensor edge.

-   :material-vector-polyline:{ .lg .middle } **Signal utility preserved**

    ---

    Scorecards report waveform fidelity and physiology-aware metrics such as PPG heart-rate preservation and ECG morphology behavior.

-   :material-chip:{ .lg .middle } **Deployable artifacts**

    ---

    Export LiteRT/TFLite models, C headers, codebooks, reference vectors, and package manifests for edge and host runtimes.

-   :material-puzzle:{ .lg .middle } **Extensible codec surface**

    ---

    Start from reusable blocks. Add new modalities, losses, quantizers, entropy models, or deployment targets without joining a mandatory experiment framework.

</div>

---

## High-level getting started

Start with the lightest path that answers your question.

| Goal | Best starting point | Dataset required? |
|------|---------------------|-------------------|
| Try a published codec | [HuggingFace guide](huggingface.md) or [example notebooks](examples.md) | No |
| Evaluate on your own signal | [Dataset setup · bring your own data](datasets.md#bring-your-own-data) | Your signal only |
| Inspect supported release artifacts | [Model zoo](models/index.md) and [golden experiments](experiments/index.md) | No |
| Validate a deploy package | [Deployment guide](deployment.md) | No |
| Reproduce golden results | [Golden experiments](experiments/index.md) | Yes |
| Extend or create a new codec | [Experiment architecture](experiment-architecture.md) | Usually |

Minimal runtime example:

```python
import numpy as np
from compressionkit.runtime import load_codec

codec = load_codec("Ambiq/compressionkit-ppg-4x")
frame = np.zeros(codec.frame_size, dtype=np.float32)

encoded = codec.compress(frame)
reconstructed = codec.decompress(encoded)
```

Install locally:

```bash
uv pip install "compressionkit[hf]"

# Or, from a source checkout:
uv sync --extra hf
```

---

## Current v1 artifacts

The first release-grade packages exercise the common artifact contract across PPG and ECG:

| Area | Current v1 status | Where to go |
|------|-------------------|-------------|
| PPG neural codecs | Published HuggingFace bundles from 2x to 32x | [PPG model zoo](models/ppg.md) |
| ECG neural codecs | Published HuggingFace bundles from 2x to 64x | [ECG model zoo](models/ecg.md) |
| DSP and hybrid lanes | Registered and locally reproducible; publication pending | [Experiments](experiments/index.md) |
| Release contract | Manifests, specs, checksums, reference vectors, scorecards | [V1 release contract](release-contract.md) |
| Runtime/deployment | Local and HuggingFace loading plus deploy validation | [Deployment guide](deployment.md) |

For method details, see [Methods](methods/index.md). For the philosophy behind blocks, ready-made experiments, and goldens, see [Experiment Architecture](experiment-architecture.md).

---

## Where to go next

<div class="grid cards" markdown>

-   :material-play-circle:{ .lg .middle } **Getting Started**

    ---

    Install the package, run the notebooks, and learn the shortest path to a codec round-trip.

    [:octicons-arrow-right-24: Start here](getting-started.md)

-   :material-cloud-download:{ .lg .middle } **HuggingFace Bundles**

    ---

    Download a published codec, run it on a sample frame, and understand bundle contents.

    [:octicons-arrow-right-24: Load a bundle](huggingface.md)

-   :material-view-grid:{ .lg .middle } **Model Zoo**

    ---

    Browse current PPG and ECG release artifacts, metrics, and package links.

    [:octicons-arrow-right-24: Browse models](models/index.md)

-   :material-flask-outline:{ .lg .middle } **Golden Experiments**

    ---

    Reproduce, validate, stage, or extend release-grade experiments.

    [:octicons-arrow-right-24: View goldens](experiments/index.md)

-   :material-blocks:{ .lg .middle } **Experiment Architecture**

    ---

    Learn how reusable blocks, recipes, deploy artifacts, and golden releases fit together.

    [:octicons-arrow-right-24: Read architecture](experiment-architecture.md)

-   :material-badge-check:{ .lg .middle } **Validation Scorecard**

    ---

    Understand noise handling, artifact behavior, physiological metrics, and release checks.

    [:octicons-arrow-right-24: Read scorecard](validation-scorecard.md)

</div>