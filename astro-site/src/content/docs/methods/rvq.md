---
title: "RVQ Autoencoder"
description: "Understand the encoder, residual vector quantizer, decoder and compression ratio."
---


The **Residual Vector Quantization (RVQ) Autoencoder** is the primary compression method in compressionKIT. It combines a convolutional encoder/decoder with a multi-level discrete bottleneck to achieve high compression ratios while maintaining signal fidelity.

## Architecture

### Encoder

The encoder uses a series of stride-2 stages to downsample the input temporally:

- **First 2 stages**: Standard `Conv2D` blocks (kernel 7, stride 2)
- **Remaining stages**: Depthwise-separable `Conv2D` blocks (more efficient)
- **Head projection**: 1×1 convolution to `embedding_dim` channels

Each block includes configurable normalization (`batch`, `layer`, or `none`) and ReLU activation.

The total downsampling factor is $2^{\text{num\_stages}}$.

### RVQ Bottleneck

The **Residual Vector Quantizer** from heliaEDGE discretizes the continuous latent representation:

1. Find nearest codebook entry for each latent position
2. Compute residual (what the first codebook missed)
3. Quantize the residual with the next codebook
4. Repeat for $M$ levels

Each level uses a codebook of size $K$ (the `latent_width` parameter). Training uses the **straight-through estimator** for gradient flow, with commitment and codebook losses.

### Decoder

The decoder mirrors the encoder with upsampling stages:

- `UpSampling2D` (2×) → `Conv2D` → `SeparableConv2D` (anti-aliasing)
- Optional normalization per block
- Final 1×1 convolution to output channels

## Configuration

```yaml
model:
  embedding_dim: 16      # Latent channel dimension
  latent_width: 512      # Codebook size K
  num_levels: 2          # RVQ levels M
  num_stages: 3          # Encoder stages (2^3 = 8× downsample)
  base_filters: 48       # First stage filter count
  multiplier: 1.25       # Filter growth per stage
  beta: 0.25             # VQ commitment loss weight
  encoder_block_norm: batch
  encoder_head_norm: none
  decoder_block_norm: none
  decoder_head_norm: layer
```

## Compression Ratio

The compression ratio depends on the input bit depth, downsampling factor, codebook size, and number of levels:

$$
\text{CR} = \frac{T \times B}{\frac{T}{2^N} \times M \times \log_2(K)}
$$

| Parameter | Symbol | PPG Value | ECG Value |
|-----------|--------|-----------|----------|
| Frame size | $T$ | 320 | 512 |
| Input bit depth | $B$ | 16 | 16 |
| Num stages | $N$ | 1–4 | 1–4 |
| Num levels | $M$ | 1–2 | 1–2 |
| Codebook size | $K$ | 256 | 256 |

**Example** (PPG 8×): $\text{CR} = \frac{320 \times 16}{40 \times 2 \times 8} = 8\times$

## Golden Configurations

PPG and ECG share the following RVQ configurations. ECG also has a 64× configuration; see the [ECG model page](/compressionkit/models/ecg/).

| Name | Stages | Levels | Codebook | CR | Use Case |
|------|--------|--------|----------|----|----------|
| `02x` | 1 | 2 | 256 | 2× | Maximum fidelity |
| `04x` | 2 | 2 | 256 | 4× | High quality |
| `08x` | 3 | 2 | 256 | 8× | Compare against your quality requirements |
| `16x` | 4 | 2 | 256 | 16× | Bandwidth-constrained |
| `32x` | 4 | 1 | 256 | 32× | Extreme compression |

See the [Model Zoo](/compressionkit/models/) for full results.

## Training

The training pipeline:

1. Loads signal data (PPG or ECG) via TFRecord cache
2. Applies preprocessing (random crop + layer norm) and augmentation (Gaussian noise)
3. Builds the RVQ autoencoder using heliaEDGE components
4. Trains with Adam optimizer, MSE + derivative loss + RVQ commitment/codebook losses
5. Monitors `val_mse` for checkpointing and early stopping
6. Exports best encoder to INT8 TFLite + C header

## Deployment

The trained encoder is exported as:

- **`encoder.tflite`** — INT8 quantized TFLite model for on-device inference
- **`encoder.h`** — C header with the model weights as a byte array
- **`encoder_float32.tflite`** — FP32 LiteRT encoder for browser and host integrations

The decoder and RVQ codebooks are stored separately for server-side reconstruction. A split deployment runs the encoder and codebook lookup on the device and transmits indices to a separate decoder.
