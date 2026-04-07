---
icon: lucide/terminal
---

# CLI Reference

## `train-ppg-rvq`

Train a PPG RVQ compression model from a YAML configuration file.

### Usage

```bash
train-ppg-rvq --config <path-to-yaml>
```

Or as a module:

```bash
python -m compressionkit.cli.train_ppg_rvq --config <path-to-yaml>
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to YAML configuration file |

### Example

```bash
train-ppg-rvq --config configs/ppg_rvq_08x_ds8_l2.yaml
```

### Configuration

The YAML file is validated against `compressionkit.configs.ppg_rvq.PpgRvqConfig`. Any fields not specified in the YAML will use their default values. See [Configs API](api/configs.md) for the full schema.

### Outputs

All outputs are written to `<results_root>/<run_name>/`:

```
results/ppg_rvq_08x_ds8_l2/
├── config.json              # Resolved configuration
├── best_model.weights.h5    # Best checkpoint weights
├── model.weights.h5         # Final checkpoint weights
├── encoder.keras            # Encoder only
├── decoder.keras            # Decoder only
├── rvq_weights.npz          # RVQ codebook weights
├── encoder.tflite           # INT8 quantized encoder
├── encoder.h                # C header for deployment
├── summary.json             # Metrics and compression stats
├── training_history_*.csv   # Per-epoch metrics
├── tensorboard/             # TensorBoard logs
├── plots/                   # Reconstruction visualizations
└── sample_*.csv             # Per-sample reconstruction data
```
