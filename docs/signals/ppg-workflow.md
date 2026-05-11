---
icon: lucide/activity
---

# PPG Workflow

This page describes the current production-oriented workflow in compressionKIT. Today, the most complete path in the toolkit is a **PPG compression and reconstruction pipeline** built around a residual vector quantization autoencoder.

## What Is Supported Today

The current supported task is:

- Input: 64 Hz PPG signals, typically sourced from EDF recordings with a `Pleth` channel.
- Training objective: reconstruct the waveform after neural compression while preserving useful physiological structure.
- Deployment path: export the encoder as INT8 LiteRT plus a C header for embedded integration.
- Evaluation path: compare original versus reconstructed signals with waveform, reconstruction, and compression metrics.

In practice, this gives teams a path from offline dataset preparation to embedded export and browser-based inspection without changing toolchains midway through the project.

## End-To-End Flow

1. Load PPG segments from EDF data and resample to 64 Hz.
2. Build deterministic training or validation windows with cache-backed dataset support.
3. Normalize and optionally augment the input with noise and synthetic mixing.
4. Compress the signal through an RVQ autoencoder configured for a target operating point.
5. Track validation metrics and keep the best-performing checkpoint weights.
6. Export encoder artifacts for deployment and save summaries, plots, and reconstruction samples.

## Core Design Choices

### Fixed-window processing

The default flow uses a fixed frame size so training, evaluation, and deployment all share the same model assumptions. That keeps exported inference predictable on constrained targets.

### YAML-driven configuration

The full run is driven from YAML, which makes it easy to keep experiments reproducible and to compare configurations across compression ratios.

### Embedded-oriented outputs

The workflow does not stop at training. It produces deployment artifacts, evaluation summaries, and reconstruction samples in the same result directory.

## Primary Inputs And Outputs

| Stage | Input | Output |
|------|------|------|
| Data loading | EDF PPG recordings | Windowed training and validation datasets |
| Preprocessing | Raw or cached windows | Normalized model-ready tensors |
| Codec model | PPG frames | Quantized latent representation and reconstructed waveform |
| Evaluation | Original and reconstructed signals | MSE, PRD, cosine, plots, summary JSON |
| Export | Best encoder weights | LiteRT encoder and C header |

## Recommended Entry Points

Use the CLI when you want reproducible runs from config files:

```bash
train-ppg-rvq --config configs/ppg_rvq_64hz_08x_golden.yaml
```

Or call the module directly:

```bash
python -m compressionkit.recipes.train_ppg_rvq --config configs/ppg_rvq_64hz_08x_golden.yaml
```

## Four Reference Operating Points

compressionKIT currently documents four reference PPG variants for the v1 flow:

- 2× for highest fidelity.
- 4× for balanced quality and savings.
- 8× for aggressive compression with strong signal preservation.
- 16× for maximum reduction when system efficiency matters most.

The detailed walkthrough is in [PPG 2x-16x Examples](ppg-v1-examples.md).

## Related Demo

The workflow is also exposed through a customer-facing browser demo that shows reconstruction quality, compression tradeoffs, and live controls in a more visual format. See [PPG Codec Demo](../demo/ppg-codec.md).
