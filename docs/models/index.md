---
icon: lucide/layers
---

# Model Zoo

compressionKIT ships golden reference models for both PPG and ECG signals. Each model is defined by a source-controlled YAML config and trained with the same architecture — an RVQ autoencoder with EMA codebook updates.

<div class="grid cards" markdown>

-   :material-heart-pulse:{ .lg .middle } **PPG Models (v1.0)**

    ---

    Five operating points from 2× to 32× compression at 64 Hz.
    Trained on the MESA dataset with HR/HRV preservation metrics.

    [:octicons-arrow-right-24: Browse PPG models](ppg.md)

-   :material-heart-flash:{ .lg .middle } **ECG Models (v1.0)**

    ---

    Six operating points from 2× to 64× compression at 256 Hz.
    Trained on PTB-XL Lead II with QRS-preserving derivative loss.

    [:octicons-arrow-right-24: Browse ECG models](ecg.md)

</div>

## At a Glance

| Signal | Rate | Frame | Compression Ratios | Dataset | Access |
|--------|------|-------|--------------------|---------|--------|
| **PPG** | 64 Hz | 5 s (320 samples) | 2× / 4× / 8× / 16× / 32× | MESA | Restricted |
| **ECG** | 256 Hz | 2 s (512 samples) | 2× / 4× / 8× / 16× / 32× / 64× | PTB-XL | Open |

!!! note "How the compression ratio is computed"
    The `NNx` label is the **true end-to-end compression ratio**:

    \[ \text{CR} = \frac{T \cdot B}{(T/D) \cdot L \cdot \log_2 K} \]

    With $B = 16$-bit input, $K = 256$ codebook entries ($\log_2 K = 8$ bits/index), and $L = 2$ RVQ levels, each latent position costs $2 \times 8 = 16$ bits — exactly the raw sample width. The CR therefore equals the encoder downsample factor $D$. The 32× tier uses $L = 1$ to double the ratio at $D = 16$.

## Versioning

Models follow a `v{major}.{minor}` scheme:

- **v1.0** — Current golden models. RVQ autoencoder with EMA codebook, UpSampling2D decoder, derivative loss, 200 epochs.

New model versions will be published when architecture changes, training recipe improvements, or dataset updates produce meaningfully different results.

## Deployment Artifacts

Every golden config produces these artifacts:

| Artifact | Format | Deployment Target |
|----------|--------|-------------------|
| Encoder | INT8 TFLite (`.tflite`) + C header (`.h`) | On-device (MCU) |
| RVQ codebook | C header (`.h`) + NumPy (`.npz`) | On-device (MCU) |
| Decoder | Keras model (`.keras`) | Server-side |
| Sample data | NumPy (`.npz`) with 50 input/target/reconstruction pairs | Validation |
| Manifest | `deploy_manifest.json` | Metadata |

## Training a Model

```bash
# PPG example (8x compression)
python -m compressionkit.recipes.train_ppg_rvq --config configs/ppg_rvq_64hz_08x_golden.yaml

# ECG example (8x compression)
python -m compressionkit.recipes.train_ecg_rvq --config configs/ecg_rvq_256hz_08x_golden.yaml
```

Results are written to `results/<run_name>/` including weights, metrics, plots, and the `deploy/` directory with all deployment artifacts.
