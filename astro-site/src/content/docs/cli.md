---
title: "CLI Reference"
description: "Command-line entry points for codec training, evaluation and artifact validation."
---


Run commands from the source checkout with `uv run`. For a first round trip, use [Getting started](/compressionkit/getting-started/); for a registered training workflow, use [Experiments](/compressionkit/experiments/).

## Common commands

| Goal | Command |
| --- | --- |
| List recipes | `uv run compressionkit list` |
| List registered experiments | `uv run compressionkit golden list` |
| Reproduce a PPG experiment | `uv run compressionkit golden run ppg-rvq-4x` |
| Validate a local package | `uv run compressionkit golden validate-deploy results/ppg_rvq_64hz_04x_golden/deploy` |
| Inspect available flags | `uv run compressionkit golden --help` |

Training requires the dataset specified by the configuration. Validation requires an existing deploy directory.

## The `compressionkit` multiplexer

Every registered recipe is also reachable through a single multiplexing
command:

```bash
compressionkit list                                     # show all recipes
compressionkit train-ppg-rvq --config <path-to-yaml>    # run one
```

This works for recipes shipped in the package **and** for any recipe
imported before dispatch (e.g. imported from your own module or an
entry-point plugin).

## `train-ppg-rvq`

Train a PPG RVQ compression model from a YAML configuration file.

For release reproduction, prefer the golden registry wrapper because it pins the
canonical dataset, config, export, and validation steps:

```bash
uv run compressionkit golden run ppg-rvq-8x
```

Use `train-ppg-rvq` when developing or debugging a custom recipe directly.

### Usage

```bash
train-ppg-rvq --config <path-to-yaml>
```

Or as a module:

```bash
python -m compressionkit.recipes.train_ppg_rvq --config <path-to-yaml>
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to YAML configuration file |

### Example

```bash
uv run train-ppg-rvq --config configs/ppg_rvq_64hz_08x_golden.yaml
```

### Configuration

The YAML file is validated against `compressionkit.configs.ppg_rvq.PpgRvqConfig`. Any fields not specified in the YAML will use their default values. See [PpgRvqConfig](/compressionkit/reference/api/compressionkit/configs/ppg_rvq/#compressionkit.configs.ppg_rvq.PpgRvqConfig) for the full schema.

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


## `train-ecg-rvq`

Train an ECG RVQ compression model from a YAML configuration file.

For release reproduction, prefer the golden registry wrapper:

```bash
uv run compressionkit golden run ecg-rvq-8x
```

Use `train-ecg-rvq` when developing or debugging a custom recipe directly.

<span id="usage_1"></span>

### Usage

```bash
train-ecg-rvq --config <path-to-yaml>
```

Or as a module:

```bash
python -m compressionkit.recipes.train_ecg_rvq --config <path-to-yaml>
```

<span id="arguments_1"></span>

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to YAML configuration file |

<span id="example_1"></span>

### Example

```bash
uv run train-ecg-rvq --config configs/ecg_rvq_256hz_08x_golden.yaml
```

<span id="configuration_1"></span>

### Configuration

The YAML file is validated against `compressionkit.configs.ecg_rvq.EcgRvqConfig`. Any fields not specified in the YAML will use their default values. See [EcgRvqConfig](/compressionkit/reference/api/compressionkit/configs/ecg_rvq/#compressionkit.configs.ecg_rvq.EcgRvqConfig) for the full schema.

<span id="outputs_1"></span>

### Outputs

All outputs are written to `<results_root>/<run_name>/`:

```
results/ecg_rvq_256hz_08x_golden/
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

## Adding a new command

1. Write a recipe module under `compressionkit/recipes/` and decorate
   its training function:

    ```python
    from compressionkit.configs.ppg_rvq import PpgRvqConfig
    from compressionkit.recipes import recipe

    @recipe("train-ppg-rvq-v2", config_cls=PpgRvqConfig)
    def train(cfg: PpgRvqConfig) -> dict:
        ...
    ```

    The decorator registers the recipe and attaches a `main` to the
    module — no argparse boilerplate needed.

2. (Optional) Register a dedicated console script in `pyproject.toml`
   if you want a standalone command instead of going through
   `compressionkit <name>`:

    ```toml
    [project.scripts]
    train-ppg-rvq-v2 = "compressionkit.recipes.train_ppg_rvq_v2:main"
    ```

3. Reinstall the package (`uv sync`) to pick up any new console script.
