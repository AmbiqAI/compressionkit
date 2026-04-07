---
icon: lucide/play
---

# Getting Started

## Installation

compressionKIT uses [uv](https://docs.astral.sh/uv/) for package management. Install in a development environment:

```bash
# Clone the repository
git clone https://github.com/AmbiqAI/compressionkit.git
cd compressionkit

# Create virtual environment and install
uv sync

# Or install in editable mode
uv pip install -e .
```

### Dependencies

compressionKIT relies on:

- **[Keras 3](https://keras.io/)** — Model building and training
- **[heliaEDGE](https://github.com/AmbiqAI/helia-edge)** — RVQ layers, VQ trainers, preprocessing pipelines, TFLite export
- **[physioKIT](https://github.com/AmbiqAI/physiokit)** — Signal processing and physiological metric computation
- **[TensorFlow](https://www.tensorflow.org/)** — Backend and tf.data pipelines

## Project Structure

```
compressionkit/
├── cli/                # Command-line entry points
│   └── train_ppg_rvq.py
├── configs/            # Pydantic configuration models
│   └── ppg_rvq.py
├── datasets/           # Data loading and tf.data pipelines
│   └── ppg.py
├── preprocessing/      # Augmentation and synthetic generation
│   └── ppg.py
├── models/             # Model architecture builders
│   └── rvq_autoencoder.py
├── evaluation/         # Metrics and artifact generation
│   ├── metrics.py
│   └── artifacts.py
├── export/             # TFLite and C header export
│   └── tflite.py
├── trainers/           # Training orchestration
│   └── ppg_rvq.py
└── logging/            # W&B integration
    └── wandb_utils.py
```

## Training a Model

### 1. Prepare a YAML config

Create a configuration file (or use an existing one):

```yaml
run_name: ppg_rvq_08x_ds8_l2

data:
  datasets_dir: /path/to/datasets
  dataset_glob: "mesa-commercial-use/polysomnography/edfs/*.edf"
  sampling_rate: 64
  frame_size: 320
  segment_samples: 7680
  batch_size: 64
  epochs: 100

model:
  embedding_dim: 16
  latent_width: 256
  num_levels: 2
  num_stages: 3      # 2^3 = 8× downsample
  base_filters: 32
  beta: 0.25

training:
  learning_rate: 0.001
  early_stop_patience: 25
  selection_metric: val_mse

output:
  results_root: results
  tensorboard: true
```

### 2. Run training

```bash
train-ppg-rvq --config path/to/config.yaml
```

### 3. Outputs

After training completes, the results directory contains:

| File | Description |
|------|-------------|
| `best_model.weights.h5` | Best checkpoint weights selected by validation metric |
| `model.weights.h5` | Final model weights |
| `encoder.keras` | Encoder model only |
| `decoder.keras` | Decoder model only |
| `rvq_weights.npz` | RVQ codebook weights |
| `encoder.tflite` | INT8 quantized TFLite encoder |
| `encoder.h` | C header for embedded deployment |
| `summary.json` | Full metrics and compression statistics |
| `plots/` | Reconstruction visualizations |

## Dataset Modes

compressionKIT supports three data loading modes:

### In-Memory (default)
Loads all segments into RAM. Best for small-to-medium datasets.

### Streaming
Samples random windows per-subject from EDF files each epoch. Memory-efficient for large datasets.

```yaml
data:
  streaming:
    enabled: true
    windows_per_subject_train: 8
    windows_per_subject_val: 4
```

### TFRecord Cache
Builds a deterministic TFRecord cache keyed by dataset+config signature. Best balance of speed and reproducibility.

```yaml
data:
  cache:
    enabled: true
    cache_root: datasets/ppg_tfrecord_cache
    windows_per_subject_train: 64
```
