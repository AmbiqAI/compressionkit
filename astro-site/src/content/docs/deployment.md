---
title: "Deployment Guide"
description: "Export and validate codec packages for integration."
---


This guide covers the deployment package emitted by `compressionkit.export.deploy.export_for_deployment()`, the lightweight runtime APIs, HuggingFace loading, and the optional two-stage entropy-coding path.

For the release-grade package requirements that span AI, DSP, HuggingFace publication, and docs generation, see the [v1 release contract](/compressionkit/release-contract/).

## Start with a host round-trip

Use `load_codec()` to select the runtime from package metadata. Start with a downloaded package and a representative frame before splitting the pipeline across devices. See the [quickstart](/compressionkit/getting-started/#2-try-a-published-codec) for installation and the common interface.

## From sensor to reconstruction

| Step | Your integration owns | Package/runtime supplies |
| --- | --- | --- |
| Prepare a frame | Sensor acquisition, sample-rate conversion, frame boundaries and units | Expected sample rate, frame size and codec configuration |
| Encode | Target runtime integration and buffer management | Python reference codec; exported artifacts according to the manifest |
| Store or transmit | Packet format, codec identity/version, sequencing and error handling | Family-specific payload and decoding side information |
| Reconstruct | Matching package selection and handling missing frames | Decoder for the matching encoded representation |
| Validate | Application acceptance criteria and target measurements | Saved reference data and evaluation tools |

### What crosses the transport boundary

`compress()` returns an `EncodedFrame` with `payload`, `nbits` and `side`. Preserve the information needed by `decompress()`:

- **RVQ:** the payload is a token-index array. Its NumPy storage size is not its packed token bitrate. An integration must define how indices are packed and restore their shape.
- **SPIHT:** the payload is a coded bitstream with decoding side information.
- **Hybrid:** the SPIHT payload represents the preprocessed signal; use the matching hybrid package and metadata.

This Python object is not a network protocol. Define a stable envelope with the codec/package version, frame identity, payload length and required side information. Count that envelope and transport overhead when estimating bandwidth savings.

### What an embedded export does not establish

A model file or C header is an integration input. It does not establish a complete firmware codec, a working sensor-to-radio path, or measured latency and memory use on your board. Check the actual package contents, implement any missing codec/transport pieces, and compare target results against the reference runtime before treating the integration as validated.

## Deployment Workflow

The standard RVQ deployment flow has three pieces:

1. Train a model and export the `deploy/` directory under `results/<run_name>/deploy/`.
2. Integrate the encoder and RVQ codebook into your runtime to turn frames into token indices. Match the exported tensor types and quantization settings.
3. Decode either locally, on a host, or in the cloud depending on your product architecture.

Use `load_codec()` for the common interface across families. The lower-level examples below are specific to RVQ.

## Deploy Package Contents

`export_for_deployment()` writes a manifest plus the artifacts needed for single-stage inference. Depending on export options, some files are always present and some are optional.

| File | Required | Purpose | Typical target |
|------|----------|---------|----------------|
| `deploy_manifest.json` | Yes | Declares file names, tensor shapes, quantization mode, and codebook metadata. | Runtime metadata |
| `encoder.tflite` | Yes | INT8 LiteRT encoder used to produce continuous latents from input frames. | MCU / edge device |
| `encoder.h` | Yes | C header for the quantized encoder blob. | MCU firmware |
| `encoder.keras` | Yes | Reference encoder kept in Keras format. | Server / offline tools |
| `decoder.keras` | Yes | Reference decoder kept in Keras format. | Server / offline tools |
| `decoder_float32.tflite` | Optional, exported by default | Float32 LiteRT decoder for host-side reconstruction without Keras. | x86 / ARM Linux |
| `decoder.tflite` | Optional | INT8 LiteRT decoder for full on-device reconstruction. | MCU / edge device |
| `decoder.h` | Optional | C header for the INT8 decoder blob. | MCU firmware |
| `codebook.npz` | Yes | NumPy archive containing RVQ codebook tables. | Python runtime |
| `codebook.h` | Yes | C header containing RVQ codebook tables. | MCU firmware |
| `sample_data.npz` | Optional | Synthetic or evaluation sample inputs / targets / reconstructions. | Validation / demos |
| `model_card.json` | Optional | Metadata used when publishing to HuggingFace. | Release tooling |

The manifest is the source of truth. The runtime reads it first, then resolves the encoder, optional decoder, and codebook files from the names listed there.

## Local Runtime Quickstart

Use the source-checkout installation from [Getting started](/compressionkit/getting-started/#1-install). LiteRT executes the exported models, but importing the package also loads modules with Keras and other development dependencies; this is not a separately packaged minimal runtime.

```python
import numpy as np

from compressionkit.runtime import RVQCodec


def synthetic_ppg_frame(frame_size: int = 320, sample_rate: int = 64) -> np.ndarray:
    t = np.arange(frame_size, dtype=np.float32) / np.float32(sample_rate)
    waveform = (
        0.55 * np.sin(2.0 * np.pi * 1.2 * t)
        + 0.18 * np.sin(2.0 * np.pi * 2.4 * t + 0.3)
        + 0.03 * np.sin(2.0 * np.pi * 0.15 * t)
    )
    return waveform.reshape(1, 1, frame_size, 1).astype(np.float32)


codec = RVQCodec("results/ppg_rvq_64hz_04x_golden/deploy")
signal = synthetic_ppg_frame()

indices = codec.encode(signal)
reconstruction = codec.decode(indices)

print(indices.shape)
print(reconstruction.shape)
print(codec.num_levels, codec.num_embeddings, codec.embedding_dim)
```

Use the lower-level methods when you want to separate the pipeline into explicit steps:

```python
latent = codec.encode_latent(signal)
indices = codec.quantize_latent(latent)
quantized_latent = codec.dequantize_indices(indices)

if codec.has_decoder:
    reconstruction = codec.decode_latent(quantized_latent)
```

This is useful when the encoder runs on-device and the decoder runs elsewhere.

## HuggingFace Quickstart

Published model repos follow the convention `Ambiq/compressionkit-{modality}-{cr}x-{version}`, for example `Ambiq/compressionkit-ppg-4x-v1.0` or `Ambiq/compressionkit-ecg-8x-v1.0`.

Install the optional Hub dependency:

```bash
uv sync --extra hf
```

Then load a codec directly from the Hub:

```python
import numpy as np

from compressionkit.runtime import RVQCodec


def synthetic_ecg_frame(frame_size: int = 512, sample_rate: int = 256) -> np.ndarray:
    t = np.arange(frame_size, dtype=np.float32) / np.float32(sample_rate)
    waveform = (
        0.75 * np.sin(2.0 * np.pi * 1.1 * t)
        + 0.12 * np.sin(2.0 * np.pi * 9.0 * t)
        + 0.02 * np.sin(2.0 * np.pi * 0.3 * t)
    )
    return waveform.reshape(1, 1, frame_size, 1).astype(np.float32)


codec = RVQCodec.from_pretrained("Ambiq/compressionkit-ecg-4x-v1.0")
signal = synthetic_ecg_frame()

indices = codec.encode(signal)
reconstruction = codec.decode(indices)
```

The Hub staging step renames a few artifacts for distribution, but `RVQCodec.from_pretrained()` handles those differences for you. In particular, it maps:

- `config.json` to `deploy_manifest.json`
- `encoder_int8.tflite` to `encoder.tflite`
- `decoder_int8.tflite` to `decoder.tflite`
- `sample_stimulus.npz` to `sample_data.npz`

## Two-Stage Compression

The two-stage path is for advanced users who want additional bitrate reduction beyond uniform RVQ token coding.

Stage 1 uses the standard RVQ codec to produce token indices. Stage 2 runs a causal entropy prior over those tokens and arithmetic-codes them into a compressed bitstream.


:::note[Note]
The example below is illustrative and uses a locally built prior package.
Reproduce a `*-prior` experiment before using this local path. Check the package manifest for the matching prior files.

:::

```python
import numpy as np

from compressionkit.runtime import RVQCodec
from compressionkit.runtime.prior import EntropyPrior
from compressionkit.runtime.two_stage import TwoStageCodec


def synthetic_ppg_frame(frame_size: int = 320, sample_rate: int = 64) -> np.ndarray:
    t = np.arange(frame_size, dtype=np.float32) / np.float32(sample_rate)
    waveform = 0.6 * np.sin(2.0 * np.pi * 1.25 * t) + 0.1 * np.sin(2.0 * np.pi * 2.5 * t)
    return waveform.reshape(1, 1, frame_size, 1).astype(np.float32)


run_dir = "results/ppg_rvq_64hz_04x_golden"
codec = RVQCodec(f"{run_dir}/deploy")
prior = EntropyPrior(f"{run_dir}/deploy/prior_int8.tflite")
two_stage = TwoStageCodec(codec, prior)

signal = synthetic_ppg_frame()
result = two_stage.compress(signal)
reconstruction = two_stage.decompress(result)

print(result.bits_per_token_uniform)
print(result.bits_per_token_actual)
print(result.cr_uplift)
```

If you already have RVQ indices, you can skip the encoder and operate directly on tokens:

```python
indices = codec.encode(signal)
estimate = two_stage.estimate_bitrate(indices)
compressed = two_stage.compress_indices(indices)
restored_indices = two_stage.decompress_indices(compressed)
```

Use the two-stage path when transport or storage cost is the limiting factor and you can afford the extra prior model. Use the single-stage codec when simplicity, fixed compute, or embedded deployment dominates.

SPIHT and hybrid packages use family-specific runtime loaders. Use [load_codec](/compressionkit/reference/api/compressionkit/runtime/loader/#compressionkit.runtime.loader.load_codec) to dispatch from package metadata, and consult the [experiment page](/compressionkit/experiments/) for the matching bundle target.

## Platform Considerations

<span id="mcu-edge-device-path"></span>

### MCU / edge-device path

- Keep `encoder.tflite` and `codebook.h` on-device.
- Add `decoder.tflite` and `decoder.h` only if the product must reconstruct on the same device.
- Prefer fixed frame shapes that match the exported encoder input exactly.
- Validate end-to-end quantization behavior with `sample_data.npz` before integrating with firmware.

### x86 or ARM Linux host path

- LiteRT can execute the exported encoder and decoder; install the package dependencies required by the Python imports.
- `decoder_float32.tflite` supports host reconstruction through a LiteRT interpreter rather than loading the Keras decoder model. This does not remove Keras from the package import dependencies.
- HuggingFace loading is most convenient in notebooks, evaluation services, and server-side decode pipelines.

### Split deployment path

- A common production split is encoder plus codebook on the wearable and decoder off-device.
- In that arrangement, transmit RVQ indices or the two-stage bitstream instead of raw windows.
- The runtime helpers (`encode_latent`, `quantize_latent`, `dequantize_indices`, `decode_latent`) map cleanly onto that boundary.

## Recommended Validation

Before shipping a deploy package, verify these basics:

1. Load the `deploy/` directory with `RVQCodec` and confirm `codec.manifest` matches the expected frame size and latent shape.
2. Run `encode()` and `decode()` on a synthetic frame to confirm artifact completeness.
3. If you publish to HuggingFace, verify `RVQCodec.from_pretrained()` on a clean environment.
4. If you ship a two-stage path, validate both `compress()` and `decompress()` with the exact prior artifact you plan to release.

## Related Pages

- See [PPG models](/compressionkit/models/ppg/) and [ECG models](/compressionkit/models/ecg/) for model-specific deployment notes.
- See [Export API](/compressionkit/api/export/) for the low-level export API reference.
- See [Signal compression demo](/compressionkit/demo/ppg-codec/) for a product-facing example of the codec in a demo workflow.
