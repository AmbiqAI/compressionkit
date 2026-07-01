---
icon: lucide/notebook
---

# Example Notebooks

Runnable Jupyter notebooks live in the [`examples/`](https://github.com/AmbiqAI/compressionkit/tree/main/examples)
folder. They are designed to pair with the [golden releases](experiments/index.md):
load a published codec from HuggingFace and try it in minutes \u2014 **no dataset
required**.

| Notebook | What it shows |
|----------|---------------|
| [`01_quickstart_golden_codec.ipynb`](https://github.com/AmbiqAI/compressionkit/blob/main/examples/01_quickstart_golden_codec.ipynb) | Load a golden codec, round-trip a frame, measure true CR and fidelity, and plot reconstructions using the license-safe reference frames bundled in every deploy package. |
| [`02_evaluate_on_your_data.ipynb`](https://github.com/AmbiqAI/compressionkit/blob/main/examples/02_evaluate_on_your_data.ipynb) | Evaluate a codec on **your own recordings** or the built-in synthetic generators, with per-frame metrics, an aggregate scorecard, and distribution plots. |

## Running them

```bash
# From a clone of the repo (installs Jupyter via the docs/dev extras).
uv sync
uv run jupyter lab examples/
```

Each notebook exposes a single `CODEC_SOURCE` knob at the top:

```python
# A published golden codec (downloads from HuggingFace) ...
CODEC_SOURCE = "Ambiq/compressionkit-ppg-4x"
# ... or a local deploy package you built yourself (runs offline):
# CODEC_SOURCE = "results/ppg_rvq_64hz_04x_golden/deploy"
```

Point it at any published RVQ tier in the [Model Zoo](models/index.md), PPG or
ECG, and the rest of the notebook adapts to the codec's modality, sample rate,
and frame size automatically. Use a local deploy package when evaluating a
reproducible SPIHT, hybrid, or custom run.

## What you don't need

- **No dataset.** Notebook 1 uses the reference frames shipped inside the deploy
  package; notebook 2 can synthesize signals on the fly.
- **No training for the published RVQ path.** Local comparison lanes only need a
  deploy package that has already been built.

When you are ready to evaluate at population scale or reproduce a golden run from
scratch, see [Dataset Setup](datasets.md) and the
[CLI reference](cli.md).
