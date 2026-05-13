---
icon: lucide/activity
---

# PPG Models (v1.0)

Golden reference models for PPG compression, trained on the MESA dataset at 64 Hz.

!!! tip "Per-experiment reproduction pages"
    Each entry below has a dedicated [experiment page](../experiments/index.md) with the
    one-command lifecycle (`compressionkit golden run <id>`), dataset details, and deploy
    artifact layout: [`ppg-rvq-2x`](../experiments/ppg-rvq-2x.md) ·
    [`ppg-rvq-4x`](../experiments/ppg-rvq-4x.md) ·
    [`ppg-rvq-8x`](../experiments/ppg-rvq-8x.md) ·
    [`ppg-rvq-16x`](../experiments/ppg-rvq-16x.md) ·
    [`ppg-rvq-32x`](../experiments/ppg-rvq-32x.md).
    Two-stage variants: [`ppg-rvq-4x-prior`](../experiments/ppg-rvq-4x-prior.md) ·
    [`ppg-rvq-8x-prior`](../experiments/ppg-rvq-8x-prior.md).

## Results Summary

| Model | Config | CR | PRD (%) | MSE | Cosine | HR MAE (bpm) | SDNN MAE (ms) |
|-------|--------|----|---------|-----|--------|--------------|---------------|
| `ppg-rvq-02x` | [`ppg_rvq_64hz_02x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_02x_golden.yaml) | 2.00x | 2.99 | 0.000666 | 0.9921 | 0.10 | 32.8 |
| `ppg-rvq-04x` | [`ppg_rvq_64hz_04x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_04x_golden.yaml) | 4.00x | 3.96 | 0.001167 | 0.9887 | 0.09 | 27.2 |
| `ppg-rvq-08x` | [`ppg_rvq_64hz_08x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_08x_golden.yaml) | 8.00x | 6.95 | 0.003601 | 0.9688 | 0.19 | 43.3 |
| `ppg-rvq-16x` | [`ppg_rvq_64hz_16x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_16x_golden.yaml) | 16.00x | 8.60 | 0.005492 | 0.9688 | 0.20 | 51.2 |
| `ppg-rvq-32x` | [`ppg_rvq_64hz_32x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_32x_golden.yaml) | 32.00x | 11.26 | 0.009432 | 0.9606 | 0.30 | 63.4 |

All metrics are on the validation set. HR/HRV metrics are from long-recording overlap-add evaluation (60 s windows, 50% hop).

![PPG PRD vs compression ratio](../assets/plots/ppg_prd_light.png#only-light)
![PPG PRD vs compression ratio](../assets/plots/ppg_prd_dark.png#only-dark)

![PPG MSE vs compression ratio](../assets/plots/ppg_mse_light.png#only-light)
![PPG MSE vs compression ratio](../assets/plots/ppg_mse_dark.png#only-dark)

![PPG Cosine Similarity vs compression ratio](../assets/plots/ppg_cos_light.png#only-light)
![PPG Cosine Similarity vs compression ratio](../assets/plots/ppg_cos_dark.png#only-dark)

## Model Details

### Architecture

All PPG models share the same architecture:

- **Encoder**: Strided Conv2D blocks (stride 2 per stage) with depthwise-separable convolutions, followed by a 1x1 projection to `embedding_dim=16`
- **RVQ bottleneck**: EMA-updated codebooks with 256 entries, 8 bits/index
- **Decoder**: UpSampling2D + Conv2D mirror of the encoder
- **Training**: EMA decay 0.99, base filters 48, filter multiplier 1.25, 200 epochs

### Compression Ratio Breakdown

| Model | Encoder Stages | Downsample | Latent Positions | RVQ Levels | Bits/Frame | CR |
|-------|---------------|------------|-----------------|------------|------------|-----|
| 02x | 1 | 2x | 160 | 2 | 2560 | 2.00x |
| 04x | 2 | 4x | 80 | 2 | 1280 | 4.00x |
| 08x | 3 | 8x | 40 | 2 | 640 | 8.00x |
| 16x | 4 | 16x | 20 | 2 | 320 | 16.00x |
| 32x | 4 | 16x | 20 | 1 | 160 | 32.00x |

Frame size = 320 samples (5 s at 64 Hz). Raw frame = 5120 bits (16-bit). Codebook size K = 256 (8 bits/index).

![PPG effective latent sample rate](../assets/plots/ppg_effective_rate_light.png#only-light)
![PPG effective latent sample rate](../assets/plots/ppg_effective_rate_dark.png#only-dark)

### Loss Function

- Primary: MSE
- Auxiliary: Derivative loss (weight 0.1) — preserves waveform first-derivative (morphology)

### HR/HRV Preservation

Long-recording evaluation uses overlap-add reconstruction on 60 s continuous segments, then computes physiological metrics with physioKit:

![PPG heart rate error](../assets/plots/ppg_hr_light.png#only-light)
![PPG heart rate error](../assets/plots/ppg_hr_dark.png#only-dark)

![PPG SDNN error](../assets/plots/ppg_sdnn_light.png#only-light)
![PPG SDNN error](../assets/plots/ppg_sdnn_dark.png#only-dark)

![PPG RMSSD error](../assets/plots/ppg_rmssd_light.png#only-light)
![PPG RMSSD error](../assets/plots/ppg_rmssd_dark.png#only-dark)

| Model | HR MAE (bpm) | HR Median AE | HR Bias | SDNN MAE (ms) | RMSSD MAE (ms) |
|-------|-------------|-------------|---------|---------------|----------------|
| 02x | 0.10 | 0.000 | +0.053 | 32.8 | 46.6 |
| 04x | 0.09 | 0.015 | +0.016 | 27.2 | 41.0 |
| 08x | 0.19 | 0.018 | +0.035 | 43.3 | 63.6 |
| 16x | 0.20 | 0.026 | −0.062 | 51.2 | 77.2 |
| 32x | 0.30 | 0.045 | −0.043 | 63.4 | 99.6 |

## Dataset: MESA

The [Multi-Ethnic Study of Atherosclerosis (MESA)](https://sleepdata.org/datasets/mesa) contains overnight PPG recordings at 256 Hz from ~1,900 subjects. **This is a restricted-access dataset.**

### Getting Access

1. Create an account at [sleepdata.org](https://sleepdata.org)
2. Apply for access to the [MESA dataset](https://sleepdata.org/datasets/mesa)
3. Wait for approval (may take several days)
4. Once approved, get your API token from your NSRR profile page

### Downloading

```bash
# Set your NSRR token
export NSRR_TOKEN="your-token-here"
```

```python
from compressionkit.datasets.mesa import MesaDataset

ds = MesaDataset(path="./datasets/mesa")
ds.download()  # uses NSRR_TOKEN env var
```

Or download manually from the NSRR website and place EDF files under `datasets/mesa-commercial-use/`.

## Training

```bash
# Train a specific compression ratio
python -m compressionkit.recipes.train_ppg_rvq --config configs/ppg_rvq_64hz_08x_golden.yaml

# Train all five golden configs
bash run_ppg_golden.sh
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

For the full runtime guide, see [Deployment Guide](../deployment.md).

### Runtime Profile

- Frame shape: `(1, 1, 320, 1)` float32
- Sample rate: `64 Hz`
- Default HuggingFace repo pattern: `AmbiqAI/compressionkit-ppg-{cr}x`

### Quickstart

```python
import numpy as np

from compressionkit.runtime import RVQCodec

t = np.arange(320, dtype=np.float32) / 64.0
signal = (0.6 * np.sin(2.0 * np.pi * 1.2 * t) + 0.1 * np.sin(2.0 * np.pi * 2.4 * t)).reshape(1, 1, 320, 1)

codec = RVQCodec.from_pretrained("AmbiqAI/compressionkit-ppg-4x")
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
3. Train: `python -m compressionkit.recipes.train_ppg_rvq --config configs/ppg_rvq_64hz_08x_v2.yaml`
4. Compare results against the golden baseline
