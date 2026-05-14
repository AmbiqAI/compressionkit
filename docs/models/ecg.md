---
icon: lucide/heart-pulse
---

# ECG Models (v1.0)

Golden reference models for ECG compression, trained on the PTB-XL dataset at 256 Hz (Lead II).

!!! tip "Per-experiment reproduction pages"
    Each entry has a dedicated [experiment page](../experiments/index.md):
    [`ecg-rvq-2x`](../experiments/ecg-rvq-2x.md) ·
    [`ecg-rvq-4x`](../experiments/ecg-rvq-4x.md) ·
    [`ecg-rvq-8x`](../experiments/ecg-rvq-8x.md) ·
    [`ecg-rvq-16x`](../experiments/ecg-rvq-16x.md) ·
    [`ecg-rvq-32x`](../experiments/ecg-rvq-32x.md) ·
    [`ecg-rvq-64x`](../experiments/ecg-rvq-64x.md).
    Two-stage variants: [`ecg-rvq-4x-prior`](../experiments/ecg-rvq-4x-prior.md) ·
    [`ecg-rvq-8x-prior`](../experiments/ecg-rvq-8x-prior.md).

## Results Summary

| Model | Config | CR | PRD (%) | MSE | Cosine |
|-------|--------|----|---------|-----|--------|
| `ecg-rvq-02x` | [`ecg_rvq_256hz_02x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ecg_rvq_256hz_02x_golden.yaml) | 2.00× | 2.59 | 0.000623 | 0.9997 |
| `ecg-rvq-04x` | [`ecg_rvq_256hz_04x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ecg_rvq_256hz_04x_golden.yaml) | 4.00× | 3.68 | 0.001259 | 0.9993 |
| `ecg-rvq-08x` | [`ecg_rvq_256hz_08x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ecg_rvq_256hz_08x_golden.yaml) | 8.00× | 6.66 | 0.004123 | 0.9978 |
| `ecg-rvq-16x` | [`ecg_rvq_256hz_16x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ecg_rvq_256hz_16x_golden.yaml) | 16.00× | 11.02 | 0.011278 | 0.9938 |
| `ecg-rvq-32x` | [`ecg_rvq_256hz_32x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ecg_rvq_256hz_32x_golden.yaml) | 32.00× | 15.53 | 0.022405 | 0.9878 |

All metrics are on the validation set.

![ECG PRD vs compression ratio](../assets/plots/ecg_prd_light.png#only-light)
![ECG PRD vs compression ratio](../assets/plots/ecg_prd_dark.png#only-dark)

![ECG MSE vs compression ratio](../assets/plots/ecg_mse_light.png#only-light)
![ECG MSE vs compression ratio](../assets/plots/ecg_mse_dark.png#only-dark)

![ECG Cosine Similarity vs compression ratio](../assets/plots/ecg_cos_light.png#only-light)
![ECG Cosine Similarity vs compression ratio](../assets/plots/ecg_cos_dark.png#only-dark)

## Model Details

### Architecture

All ECG models share the same architecture as PPG models:

- **Encoder**: Strided Conv2D blocks (stride 2 per stage) with depthwise-separable convolutions, followed by a 1×1 projection to `embedding_dim=16`
- **RVQ bottleneck**: EMA-updated codebooks with 256 entries, 8 bits/index
- **Decoder**: UpSampling2D + Conv2D mirror of the encoder
- **Training**: EMA decay 0.99, base filters 48, 200 epochs

### Compression Ratio Breakdown

| Model | Encoder Stages | Downsample | Latent Positions | RVQ Levels | Bits/Frame | CR |
|-------|---------------|------------|-----------------|------------|------------|-----|
| 02x | 1 | 2× | 256 | 2 | 4096 | 2.00× |
| 04x | 2 | 4× | 128 | 2 | 2048 | 4.00× |
| 08x | 3 | 8× | 64 | 2 | 1024 | 8.00× |
| 16x | 4 | 16× | 32 | 2 | 512 | 16.00× |
| 32x | 4 | 16× | 32 | 1 | 256 | 32.00× |

Frame size = 512 samples (2 s at 256 Hz). Raw frame = 8192 bits (16-bit). Codebook size K = 256 (8 bits/index).

![ECG effective latent sample rate](../assets/plots/ecg_effective_rate_light.png#only-light)
![ECG effective latent sample rate](../assets/plots/ecg_effective_rate_dark.png#only-dark)

### Loss Function

- Primary: MSE
- Auxiliary: Derivative loss (weight 0.1) — preserves waveform first-derivative (morphology)

## Dataset: PTB-XL

The [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset contains 21,799 12-lead ECG recordings at 500 Hz from 18,885 patients. It is publicly available under the PhysioNet Credentialed Health Data License.

compressionKIT resamples from 500 Hz to 256 Hz and uses Lead II (`lead_index=1`). The dataset is stored as HDF5 files.

### Getting Access

PTB-XL is freely available from PhysioNet:

1. Create an account at [physionet.org](https://physionet.org)
2. Complete the required training (CITI Data or Research Ethics)
3. Sign the data use agreement
4. Download from [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)

### Data Preparation

```bash
# Download and convert to HDF5
python -m compressionkit.datasets.ptbxl --download --output datasets/ptbxl
```

Or point the config `data.data_dir` to your existing PTB-XL HDF5 directory.

## Training

```bash
# Train a specific compression ratio
python -m compressionkit.recipes.train_ecg_rvq --config configs/ecg_rvq_256hz_08x_golden.yaml

# Train all five golden configs
bash run_ecg_golden.sh
```

### Output Structure

```
results/ecg_rvq_256hz_08x_golden/
├── best_model.weights.h5       # Best checkpoint
├── config.json                 # Frozen config
├── summary.json                # Training metrics
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

- Frame shape: `(1, 1, 512, 1)` float32
- Sample rate: `256 Hz`
- Default HuggingFace repo pattern: `Ambiq/compressionkit-ecg-{cr}x`

### Quickstart

```python
import numpy as np

from compressionkit.runtime import RVQCodec

t = np.arange(512, dtype=np.float32) / 256.0
signal = (0.75 * np.sin(2.0 * np.pi * 1.1 * t) + 0.12 * np.sin(2.0 * np.pi * 9.0 * t)).reshape(1, 1, 512, 1)

codec = RVQCodec.from_pretrained("Ambiq/compressionkit-ecg-4x")
indices = codec.encode(signal.astype(np.float32))
reconstruction = codec.decode(indices)
```

### Deployment Notes

- PTB-XL-based examples can be validated with synthetic waveforms; deployment does not require dataset access.
- The most common split is encoder plus codebook on-device, decoder off-device.
- ECG models can use the same `RVQCodec` runtime API as PPG even when the training recipe uses transform-domain branches internally.

## Extending

To create a new ECG model variant:

1. Copy an existing golden config: `cp configs/ecg_rvq_256hz_08x_golden.yaml configs/ecg_rvq_256hz_08x_v2.yaml`
2. Modify parameters (e.g. `base_filters`, `num_levels`, `learning_rate`)
3. Train: `python -m compressionkit.recipes.train_ecg_rvq --config configs/ecg_rvq_256hz_08x_v2.yaml`
4. Compare results against the golden baseline
