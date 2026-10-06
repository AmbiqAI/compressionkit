---
title: "PPG Models (v1.0)"
description: "PPG codec configurations, evaluation results and supporting resources."
---


PPG release artifacts and comparison lanes at 64 Hz. The published HuggingFace
bundles are RVQ neural codecs; the experiment registry also includes SPIHT DSP
and hybrid AI+DSP configurations with package links.


:::note[Per-experiment reproduction pages]
Each entry below has a dedicated [experiment page](/compressionkit/experiments/) with the
one-command lifecycle (`compressionkit golden run <id>`), dataset details, and deploy
artifact layout: [`ppg-rvq-2x`](/compressionkit/experiments/ppg-rvq-2x/) ·
[`ppg-rvq-4x`](/compressionkit/experiments/ppg-rvq-4x/) ·
[`ppg-rvq-8x`](/compressionkit/experiments/ppg-rvq-8x/) ·
[`ppg-rvq-16x`](/compressionkit/experiments/ppg-rvq-16x/) ·
[`ppg-rvq-32x`](/compressionkit/experiments/ppg-rvq-32x/).
Two-stage variants: [`ppg-rvq-4x-prior`](/compressionkit/experiments/ppg-rvq-4x-prior/) ·
[`ppg-rvq-8x-prior`](/compressionkit/experiments/ppg-rvq-8x-prior/).

:::

## Published RVQ results

The table below summarizes the published RVQ bundles. These are the v1 packages
available through HuggingFace and the runtime loader.

These validation results and the noise-aware customer scorecards use different evaluation views. Keep the metric definitions and recording windows with the numbers when comparing results.

| Model | Config | CR | PRD (%) | MSE | Cosine |
| --- | --- | --- | --- | --- | --- |
| `ppg-rvq-02x` | [`ppg_rvq_64hz_02x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_02x_golden.yaml) | 2.00x | 0.80 | 0.000063 | 0.99997 |
| `ppg-rvq-04x` | [`ppg_rvq_64hz_04x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_04x_golden.yaml) | 4.00x | 1.29 | 0.000165 | 0.99992 |
| `ppg-rvq-08x` | [`ppg_rvq_64hz_08x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_08x_golden.yaml) | 8.00x | 2.76 | 0.000756 | 0.99963 |
| `ppg-rvq-16x` | [`ppg_rvq_64hz_16x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_16x_golden.yaml) | 16.00x | 5.90 | 0.003457 | 0.99831 |
| `ppg-rvq-32x` | [`ppg_rvq_64hz_32x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_32x_golden.yaml) | 32.00x | 12.62 | 0.015843 | 0.99220 |


| Model | HR MAE (bpm) | SDNN MAE (ms) |
| --- | --- | --- |
| `ppg-rvq-02x` | 0.06 | 9.6 |
| `ppg-rvq-04x` | 0.05 | 12.9 |
| `ppg-rvq-08x` | 0.10 | 32.4 |
| `ppg-rvq-16x` | 0.19 | 58.2 |
| `ppg-rvq-32x` | 0.36 | 110.2 |


All metrics are on the validation set. HR/HRV metrics are from long-recording overlap-add evaluation (60 s windows, 50% hop).

![PPG PRD vs compression ratio](/compressionkit/assets/plots/ppg_prd_light.png#only-light)
![PPG PRD vs compression ratio](/compressionkit/assets/plots/ppg_prd_dark.png#only-dark)

<details>
<summary>Mean squared error plots</summary>

![PPG MSE vs compression ratio](/compressionkit/assets/plots/ppg_mse_light.png#only-light)
![PPG MSE vs compression ratio](/compressionkit/assets/plots/ppg_mse_dark.png#only-dark)

</details>

<details>
<summary>Cosine similarity plots</summary>

![PPG Cosine Similarity vs compression ratio](/compressionkit/assets/plots/ppg_cos_light.png#only-light)
![PPG Cosine Similarity vs compression ratio](/compressionkit/assets/plots/ppg_cos_dark.png#only-dark)

</details>

## Codec tradeoffs under noise and artifacts

PPG has three release-facing comparison lanes:

| Lane | Role | Current publication status |
|------|------|----------------------------|
| **SPIHT** | DSP faithfulness baseline; preserves clean/noisy inputs with minimal learned behavior | Hugging Face package links on the experiment pages |
| **RVQ** | Published neural codec bundles; compact learned representation with physiological scorecards | Published for 2x, 4x, 8x, 16x, and 32x |
| **Hybrid** | Learned denoising front end with a SPIHT backend; designed for wearable-noise regimes | Hugging Face package links on the experiment pages |

The robustness sweep scores each lane against a filtered clean-truth proxy after
injecting empirical noise or additive artifacts. Lower PRD is better. This is a
different question from faithful reconstruction of a clean validation frame: it
asks which codec best preserves the recoverable pulse waveform when the input is
already corrupted.

| Condition family | Current PPG crossover pattern |
|------------------|-------------------------------|
| **Clean input** | SPIHT wins at 2x-8x, while RVQ wins at 16x-32x in the current robustness fixture. |
| **Native / empirical SNR ladder** | Hybrid wins most injected-noise cells from native through -12 dB. At 32x, RVQ remains strongest for native, 12 dB, and 8 dB before hybrid takes over at heavier noise. |
| **Motion artifacts** | RVQ is the current winner across the measured CR ladder. |
| **Baseline wander** | Hybrid is the current winner across the measured CR ladder. |

![PPG robustness winners across SNR](/compressionkit/assets/plots/ppg_robustness_winners_snr.png)

![PPG robustness winners across artifacts](/compressionkit/assets/plots/ppg_robustness_winners_artifacts.png)

The practical read is that SPIHT is a strong low-noise baseline, RVQ is the
published neural operating surface, and hybrid approaches become important when
the input looks more like a wrist-worn signal with empirical noise or drift. See
[PPG CR vs fidelity](/compressionkit/methods/cr_vs_fidelity_ppg/) and
[Validation Scorecard](/compressionkit/validation-scorecard/) for the detailed scorecard
definitions and noise-aware metrics.

## Model Details

### Architecture

All PPG models share the same architecture:

- **Encoder**: Strided Conv2D blocks (stride 2 per stage) with depthwise-separable convolutions, followed by a 1x1 projection to `embedding_dim=16`
- **RVQ bottleneck**: EMA-updated codebooks with 256 entries, 8 bits/index
- **Decoder**: UpSampling2D + Conv2D mirror of the encoder
- **Training**: EMA decay 0.99, base filters 48, filter multiplier 1.25, 200 epochs

### Compression Ratio Breakdown

| Model | Encoder Stages | Downsample | Latent Positions | RVQ Levels | Bits/Frame |
| --- | --- | --- | --- | --- | --- |
| 02x | 1 | 2x | 160 | 2 | 2560 |
| 04x | 2 | 4x | 80 | 2 | 1280 |
| 08x | 3 | 8x | 40 | 2 | 640 |
| 16x | 4 | 16x | 20 | 2 | 320 |
| 32x | 4 | 16x | 20 | 1 | 160 |


| Model | CR |
| --- | --- |
| 02x | 2.00x |
| 04x | 4.00x |
| 08x | 8.00x |
| 16x | 16.00x |
| 32x | 32.00x |


Frame size = 320 samples (5 s at 64 Hz). Raw frame = 5120 bits (16-bit). Codebook size K = 256 (8 bits/index).

<details>
<summary>Latent sample rate plots</summary>

![PPG effective latent sample rate](/compressionkit/assets/plots/ppg_effective_rate_light.png#only-light)
![PPG effective latent sample rate](/compressionkit/assets/plots/ppg_effective_rate_dark.png#only-dark)

</details>

### Loss Function

- Primary: MSE
- Auxiliary: Derivative loss (weight 0.1) — preserves waveform first-derivative (morphology)

### HR/HRV Preservation

Long-recording evaluation uses overlap-add reconstruction on 60 s continuous segments, then computes physiological metrics with physioKit:

<details>
<summary>Heart-rate error plots</summary>

![PPG heart rate error](/compressionkit/assets/plots/ppg_hr_light.png#only-light)
![PPG heart rate error](/compressionkit/assets/plots/ppg_hr_dark.png#only-dark)

</details>

<details>
<summary>SDNN error plots</summary>

![PPG SDNN error](/compressionkit/assets/plots/ppg_sdnn_light.png#only-light)
![PPG SDNN error](/compressionkit/assets/plots/ppg_sdnn_dark.png#only-dark)

</details>

<details>
<summary>RMSSD error plots</summary>

![PPG RMSSD error](/compressionkit/assets/plots/ppg_rmssd_light.png#only-light)
![PPG RMSSD error](/compressionkit/assets/plots/ppg_rmssd_dark.png#only-dark)

</details>

| Model | HR MAE (bpm) | HR Median AE | HR Bias | SDNN MAE (ms) | RMSSD MAE (ms) |
|-------|-------------|-------------|---------|---------------|----------------|
| 02x | 0.06 | 0.000 | +0.046 | 9.6 | 13.8 |
| 04x | 0.05 | 0.000 | +0.021 | 12.9 | 18.7 |
| 08x | 0.10 | 0.018 | +0.041 | 32.4 | 48.5 |
| 16x | 0.19 | 0.032 | +0.016 | 58.2 | 90.8 |
| 32x | 0.36 | 0.076 | +0.180 | 110.2 | 171.3 |

## Dataset: Open unified PPG v1

The published v1 PPG goldens use `ppg-unified-strict-sanitize-v1`: a unified
mixture of BIDMC, BUT PPG, PPG-DaLiA, and WESAD. The release-facing PPG goldens
are MESA-free so users can reproduce the published artifacts without restricted
NSRR data.

MESA remains a supported restricted source for internal or custom experiments,
but it is not used in the published v1 PPG goldens. See [Dataset Setup](/compressionkit/datasets/)
for source-specific licensing and cache-building commands.

### Build the cache

```bash
# Build the open sources used by the v1 PPG goldens.
uv run python scripts/build_ppg_cache.py \
    --sources bidmc butppg ppg_dalia wesad
```

## Training

```bash
# Reproduce a specific published operating point.
uv run compressionkit golden run ppg-rvq-8x

# Reproduce all PPG RVQ goldens.
uv run compressionkit golden run-all --modality ppg --method rvq
```

### Output Structure

```
results/ppg_rvq_64hz_08x_golden/
├── best_model.weights.h5       # Best checkpoint
├── config.json                 # Frozen config
├── summary.json                # Training metrics
├── long_recording_eval.json    # HR/HRV preservation metrics
├── training_history_*.csv      # Epoch-by-epoch metrics
├── tensorboard/                # TensorBoard logs
├── plots/                      # Reconstruction plots
└── deploy/                     # Deployment artifacts
    ├── encoder.tflite          # INT8 quantized encoder
    ├── encoder.h               # C header for encoder
    ├── encoder.keras           # Keras encoder model
    ├── decoder.keras           # Keras decoder model
    ├── decoder_float32.tflite  # Float32 LiteRT decoder for host-side decode
    ├── decoder.tflite          # Optional INT8 decoder for on-device decode
    ├── decoder.h               # Optional C header for INT8 decoder
    ├── codebook.npz            # RVQ codebook weights
    ├── codebook.h              # C header for codebook
    ├── sample_data.npz         # 50 validation samples
    ├── model_card.json         # Metadata used for publishing
    └── deploy_manifest.json    # Artifact manifest
```

## Deployment

For the full runtime guide, see [Deployment Guide](/compressionkit/deployment/).

### Runtime Profile

- Frame shape: `(1, 1, 320, 1)` float32
- Sample rate: `64 Hz`
- Default HuggingFace repo pattern: `Ambiq/compressionkit-ppg-{cr}x-v1.0`

### Quickstart

```python
import numpy as np

from compressionkit.runtime import RVQCodec

t = np.arange(320, dtype=np.float32) / 64.0
signal = (0.6 * np.sin(2.0 * np.pi * 1.2 * t) + 0.1 * np.sin(2.0 * np.pi * 2.4 * t)).reshape(1, 1, 320, 1)

codec = RVQCodec.from_pretrained("Ambiq/compressionkit-ppg-4x-v1.0")
indices = codec.encode(signal.astype(np.float32))
reconstruction = codec.decode(indices)
```

### Deployment Notes

- Use `encoder.tflite` plus `codebook.h` for the lowest-footprint embedded path.
- Use `decoder_float32.tflite` when reconstruction runs on a host or cloud service.
- Add the two-stage prior only when bitrate is more constrained than compute or memory.

## Extending

To create a new PPG model variant:

1. Copy an existing golden config: `cp configs/ppg_rvq_64hz_08x_golden.yaml configs/ppg_rvq_64hz_08x_v2.yaml`
2. Modify parameters (e.g. `base_filters`, `num_levels`, `learning_rate`)
3. Train: `uv run train-ppg-rvq --config configs/ppg_rvq_64hz_08x_v2.yaml`
4. Compare results against the golden baseline
