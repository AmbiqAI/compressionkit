---
title: "Example Notebooks"
description: "Browse PPG and ECG notebook examples for codec round trips and evaluation."
---


Runnable Jupyter notebooks live in the [`examples/`](https://github.com/AmbiqAI/compressionkit/tree/main/examples)
folder. They are designed to pair with the [golden releases](/compressionkit/experiments/):
load a published codec from HuggingFace and try it with **no dataset
required**.

## Browse the examples

The guides display saved notebook cells and outputs. They are not a record of a fresh execution in your environment. Each guide includes its notebook download.

| Example | What you will do |
| --- | --- |
| [PPG codec quickstart](/compressionkit/guides/01_quickstart_golden_codec/) | Load a codec, round-trip a frame, and inspect reconstruction quality. |
| [Evaluate your recordings](/compressionkit/guides/02_evaluate_on_your_data/) | Evaluate your own data or synthetic signals and inspect aggregate errors. |
| [ECG codec quickstart](/compressionkit/guides/03_quickstart_golden_codec_ecg/) | Load an ECG codec and inspect its reconstructed frames. |

## Running them

```bash
# From a clone of the repo.
uv sync --python 3.12 --extra hf
uv run jupyter notebook examples/
```

Each notebook exposes a single `CODEC_SOURCE` knob at the top:

```python
# A published golden codec (downloads from HuggingFace) ...
CODEC_SOURCE = "Ambiq/compressionkit-ppg-4x-v1.0"
# ... or a local deploy package you built yourself (runs offline):
# CODEC_SOURCE = "results/ppg_rvq_64hz_04x_golden/deploy"
```

Point it at any published RVQ tier in the [Model Zoo](/compressionkit/models/), PPG or
ECG, and the rest of the notebook adapts to the codec's modality, sample rate,
and frame size automatically. Use a local deploy package when evaluating a
reproducible SPIHT, hybrid, or custom run.

## What you don't need

- **No dataset.** Notebook 1 uses the reference frames shipped inside the deploy
  package; notebook 2 can synthesize signals on the fly.
- **No training for the published RVQ path.** Local comparison lanes only need a
  deploy package that has already been built.

When you are ready to evaluate at population scale or reproduce a golden run from
scratch, see [Dataset Setup](/compressionkit/datasets/) and the
[CLI reference](/compressionkit/cli/).

## Notebook environment setup

Use Python 3.12. For local execution, follow [Getting started](/compressionkit/getting-started/#1-install) and launch the notebook from the repository root.

For a hosted notebook, verify the runtime's Python version before installing. Add a setup cell before the notebook's existing imports:

```python title="Notebook setup cell"
import sys
assert sys.version_info[:2] == (3, 12), "Select a Python 3.12 runtime"
%pip install "compression-kit[hf] @ git+https://github.com/AmbiqAI/compressionkit.git"
```

Restart the kernel after installation if prompted, then run the notebook from the first cell. Access to the source repository and network access to download model bundles are required. Install into the same kernel that runs the notebook. Dataset-dependent examples also need accessible files and paths adjusted for the hosted working directory.

The Colab action opens the source; it does not install dependencies, mount datasets or validate hosted runtime compatibility. Saved outputs shown here are not evidence of a successful hosted run.
