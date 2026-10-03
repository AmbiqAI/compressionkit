---
title: "Load a Hugging Face bundle"
description: "Download a codec bundle, run a sample frame, or load a local entropy-prior package."
---


Published v1 RVQ bundles live at `Ambiq/compressionkit-{modality}-{cr}x-{version}` (e.g. `-v1.0`).
This page shows the minimum code to download one and run the encoder + decoder on a sample frame.
SPIHT and hybrid package links are listed on their [experiment pages](/compressionkit/experiments/). The RVQ example below uses the RVQ-specific loader.

## 1. Install

From the [source checkout](/compressionkit/getting-started/#1-install):

```bash
uv sync --python 3.12 --extra hf
```

The `hf` extra adds `huggingface_hub` for downloading bundles. Inference can run offline once a bundle is on disk — the extra is only needed
for the `snapshot_download` / `from_pretrained` calls below.

## 2. Single-stage codec

Single-stage repos contain `encoder_int8.tflite`, `decoder_int8.tflite`, `codebook.npz`, and
`sample_stimulus.npz`. [`RVQCodec.from_pretrained`](/compressionkit/reference/api/compressionkit/runtime/codec/#compressionkit.runtime.codec.RVQCodec.from_pretrained) downloads the bundle and wires
up the LiteRT interpreters. Use the full source-checkout installation above: package imports also require Keras and evaluation dependencies.

```python
from huggingface_hub import snapshot_download
from compressionkit.runtime import RVQCodec
import numpy as np

repo = "Ambiq/compressionkit-ppg-4x-v1.0"
codec = RVQCodec.from_pretrained(repo)

# Sanity check on the bundled license-safe sample (same cached files).
# Published bundles ship `inputs`, `targets`, and `reconstructions` arrays.
deploy_dir = snapshot_download(repo)
signal = np.load(f"{deploy_dir}/sample_stimulus.npz")["inputs"][:1]
indices = codec.encode(signal)
recon = codec.decode(indices)
print("shape:", recon.shape)
```


:::note[Note]
Use `RVQCodec.from_pretrained(repo_id)` rather than `RVQCodec(local_dir)` on a raw
`snapshot_download` directory. HuggingFace bundles store the manifest as `config.json`
(not `deploy_manifest.json`) and rename the TFLite files to `*_int8.tflite`;
`from_pretrained` reconciles those names so the constructor can find them.

:::

<span id="3-two-stage-codec-codec-entropy-prior"></span>

## 3. Two-stage codec (codec + entropy prior)


:::note[Use a package containing a prior]
The optional entropy-prior stage (`prior_int8.tflite` + `prior_manifest.json`) is
reproducible from the `*-prior` golden registry entries
(`compressionkit golden list` → `ppg-rvq-4x-prior`, `ppg-rvq-8x-prior`,
`ecg-rvq-4x-prior`, `ecg-rvq-8x-prior`) and runs from locally built deploy
packages. The snippet below uses that local-package path, so
reproduce one first, e.g. `uv run compressionkit golden run ppg-rvq-8x-prior`.

:::

Two-stage deploy packages add `prior_int8.tflite` and `prior_manifest.json`. Use
[`TwoStageCodec`](/compressionkit/reference/api/compressionkit/runtime/two_stage/#compressionkit.runtime.two_stage.TwoStageCodec) to estimate entropy-prior bitrate uplift on top
of the codec's downsample ratio.

```python
from compressionkit.runtime import RVQCodec
from compressionkit.runtime.prior import EntropyPrior
from compressionkit.runtime.two_stage import TwoStageCodec
import numpy as np

deploy_dir = "results/ppg_rvq_64hz_08x_golden/deploy"  # locally built (golden run)
codec = RVQCodec(deploy_dir)
prior = EntropyPrior(f"{deploy_dir}/prior_int8.tflite")
two_stage = TwoStageCodec(codec, prior)

sample = np.load(f"{deploy_dir}/sample_data.npz")["inputs"][:1]
indices = codec.encode(sample)
rates = two_stage.estimate_bitrate(indices)
print(f"CR uplift estimate: {rates['cr_uplift']:.2f}x")
```

## 4. Where to look next

- [Experiments index](/compressionkit/experiments/) — every golden + its reproduction command.
- [Deployment guide](/compressionkit/deployment/) — moving the same artifacts onto an Ambiq-class MCU.
- [Model Zoo](/compressionkit/models/) — full quality metrics for each tier.
