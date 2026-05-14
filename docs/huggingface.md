---
icon: lucide/cloud-download
---

# Load & test a HuggingFace model in 5 minutes

Every golden experiment publishes its deploy package to `Ambiq/compressionkit-{modality}-{cr}x`.
This page shows the minimum code to download one, run the encoder + decoder on a sample frame, and
(for two-stage entries) compress with the bundled entropy prior.

## 1. Install

```bash
uv pip install compressionkit
# or, from this repo
uv sync
```

## 2. Single-stage codec

Single-stage repos contain `encoder_int8.tflite`, `decoder_int8.tflite`, `codebook.npz`, and
`sample_stimulus.npz`. The [`RVQCodec`](api/models.md) runtime needs only NumPy and a LiteRT
interpreter.

```python
from huggingface_hub import snapshot_download
from compressionkit.runtime import RVQCodec
import numpy as np

deploy_dir = snapshot_download("Ambiq/compressionkit-ppg-4x")
codec = RVQCodec(deploy_dir)

# Sanity check on the bundled license-safe sample.
signal = np.load(f"{deploy_dir}/sample_stimulus.npz")["stimulus"][:1]
indices = codec.encode(signal)
recon = codec.decode(indices)
print("shape:", recon.shape)
```

!!! tip
    `RVQCodec.from_pretrained("Ambiq/compressionkit-ppg-4x")` is a one-line shortcut that
    wraps the `snapshot_download` call above.

## 3. Two-stage codec (codec + entropy prior)

Two-stage repos add `prior_int8.tflite` and `prior_manifest.json`. Use
[`TwoStageCodec`](api/models.md#two-stage-codec) to entropy-code the RVQ indices for an extra CR
uplift on top of the codec's downsample ratio.

```python
from huggingface_hub import snapshot_download
from compressionkit.runtime import RVQCodec
from compressionkit.runtime.prior import EntropyPrior
from compressionkit.runtime.two_stage import TwoStageCodec
import numpy as np

deploy_dir = snapshot_download("Ambiq/compressionkit-ecg-4x")  # ships the prior too
codec = RVQCodec(deploy_dir)
prior = EntropyPrior(f"{deploy_dir}/prior_int8.tflite")
two_stage = TwoStageCodec(codec, prior)

sample = np.load(f"{deploy_dir}/sample_stimulus.npz")["stimulus"]
result = two_stage.compress(sample)
print(f"CR uplift from prior: {result.cr_uplift:.2f}x  ({len(result.bitstream)} bytes)")
recon = two_stage.decompress(result)
```

## 4. Where to look next

- [Experiments index](experiments/index.md) — every golden + its reproduction command.
- [Deployment guide](deployment.md) — moving the same artifacts onto an Ambiq-class MCU.
- [Model Zoo](models/index.md) — full quality metrics for each tier.
