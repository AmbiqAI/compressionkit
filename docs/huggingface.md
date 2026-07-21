---
icon: lucide/cloud-download
---

# Load & test a HuggingFace model in 5 minutes

Published v1 RVQ bundles live at `Ambiq/compressionkit-{modality}-{cr}x-{version}` (e.g. `-v1.0`).
This page shows the minimum code to download one and run the encoder + decoder on a sample frame.
Entropy-prior packages are reproducible from the golden registry, but the v1
HuggingFace bundle surface is limited to single-stage RVQ codecs.

## 1. Install

```bash
uv pip install "compressionkit[hf]"
# or, from this repo
uv sync --extra hf
```

The `hf` extra adds `huggingface_hub` for downloading bundles. The core runtime
(LiteRT + NumPy) runs offline once a bundle is on disk — the extra is only needed
for the `snapshot_download` / `from_pretrained` calls below.

## 2. Single-stage codec

Single-stage repos contain `encoder_int8.tflite`, `encoder_float32.tflite`, `decoder_int8.tflite`,
`codebook.npz`, `sample_stimulus.npz`, and `demo_recordings.npz`. `encoder_float32.tflite` is intended for browser and
host LiteRT integrations that need float32 I/O. [`RVQCodec.from_pretrained`](api/models.md) downloads the bundle and wires
up the LiteRT interpreters — it needs only NumPy and a LiteRT runtime.

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

`demo_recordings.npz` holds ten real, quality-gated continuous examples at the model rate (`signals` has shape `(10, samples)`). Read `demo_recordings_manifest.json` with it: the manifest records source provenance, ODC-By attribution, signal offsets, and the quality measurements used for selection. Apply the model's usual framing and normalization before inference.

For each 64 Hz PPG 320-sample frame or 256 Hz ECG 512-sample frame, use per-frame layer normalization before either encoder variant:

```python
mean = frame.mean()
scale = np.sqrt(np.mean((frame - mean) ** 2) + 1e-3)
encoder_input = (frame - mean) / scale
```

For display in raw units, undo it after decoding with `decoded * scale + mean`. INT8 RVQ releases produced under the current release policy include `quantization_report.json`, which records parity against `encoder_float32.tflite` on a 2,048-frame real-preprocessed holdout distinct from the 4,096 frames used for LiteRT calibration.

To refresh all local RVQ bundles before publishing an asset update:

```bash
scripts/devcontainer.sh exec -- uv run python scripts/attach_rvq_demo_recordings.py --modality all
```

!!! note
    Use `RVQCodec.from_pretrained(repo_id)` rather than `RVQCodec(local_dir)` on a raw
    `snapshot_download` directory. HuggingFace bundles store the manifest as `config.json`
    (not `deploy_manifest.json`) and rename the TFLite files to `*_int8.tflite`;
    `from_pretrained` reconciles those names so the constructor can find them.

## 3. Two-stage codec (codec + entropy prior)

!!! info "Outside the v1 HuggingFace bundle surface"
    The optional entropy-prior stage (`prior_int8.tflite` + `prior_manifest.json`) is
    reproducible from the `*-prior` golden registry entries
    (`compressionkit golden list` → `ppg-rvq-4x-prior`, `ppg-rvq-8x-prior`,
    `ecg-rvq-4x-prior`, `ecg-rvq-8x-prior`) and runs from locally built deploy
    packages. The snippet below uses that local-package path, so
    reproduce one first, e.g. `uv run compressionkit golden run ppg-rvq-8x-prior`.

Two-stage deploy packages add `prior_int8.tflite` and `prior_manifest.json`. Use
[`TwoStageCodec`](api/models.md#two-stage-codec) to estimate entropy-prior bitrate uplift on top
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

- [Experiments index](experiments/index.md) — every golden + its reproduction command.
- [Deployment guide](deployment.md) — moving the same artifacts onto an Ambiq-class MCU.
- [Model Zoo](models/index.md) — full quality metrics for each tier.
