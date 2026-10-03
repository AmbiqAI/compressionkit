---
title: "Experiments"
description: "Choose and reproduce registered PPG and ECG codec configurations."
---

An experiment connects a configuration, dataset, recipe, and deploy-package target. Use it to reproduce a specific codec or as the starting point for a new configuration.

**Golden** identifies a registered reference experiment. Registration alone does not establish that its remote package is published or that it is validated for your device.

<span id="v1-registry"></span>

## Choose an experiment

- **RVQ:** a learned encoder, residual vector quantizer, and decoder.
- **SPIHT:** a wavelet-based DSP codec.
- **Hybrid:** a learned front end combined with a DSP codec.
- **Two-stage:** an RVQ codec with a paired entropy prior for its token stream.

### PPG

| Experiment | Compression | Structure |
| --- | --- | --- |
| [`ppg-rvq-2x`](/compressionkit/experiments/ppg-rvq-2x/) | 2× | Codec |
| [`ppg-spiht-2x`](/compressionkit/experiments/ppg-spiht-2x/) | 2× | Codec |
| [`ppg-hybrid-2x`](/compressionkit/experiments/ppg-hybrid-2x/) | 2× | Codec |
| [`ppg-rvq-4x`](/compressionkit/experiments/ppg-rvq-4x/) | 4× | Codec |
| [`ppg-spiht-4x`](/compressionkit/experiments/ppg-spiht-4x/) | 4× | Codec |
| [`ppg-hybrid-4x`](/compressionkit/experiments/ppg-hybrid-4x/) | 4× | Codec |
| [`ppg-rvq-8x`](/compressionkit/experiments/ppg-rvq-8x/) | 8× | Codec |
| [`ppg-spiht-8x`](/compressionkit/experiments/ppg-spiht-8x/) | 8× | Codec |
| [`ppg-hybrid-8x`](/compressionkit/experiments/ppg-hybrid-8x/) | 8× | Codec |
| [`ppg-rvq-16x`](/compressionkit/experiments/ppg-rvq-16x/) | 16× | Codec |
| [`ppg-spiht-16x`](/compressionkit/experiments/ppg-spiht-16x/) | 16× | Codec |
| [`ppg-hybrid-16x`](/compressionkit/experiments/ppg-hybrid-16x/) | 16× | Codec |
| [`ppg-rvq-32x`](/compressionkit/experiments/ppg-rvq-32x/) | 32× | Codec |
| [`ppg-spiht-32x`](/compressionkit/experiments/ppg-spiht-32x/) | 32× | Codec |
| [`ppg-hybrid-32x`](/compressionkit/experiments/ppg-hybrid-32x/) | 32× | Codec |
| [`ppg-rvq-4x-prior`](/compressionkit/experiments/ppg-rvq-4x-prior/) | 4× | Codec + entropy prior |
| [`ppg-rvq-8x-prior`](/compressionkit/experiments/ppg-rvq-8x-prior/) | 8× | Codec + entropy prior |

### ECG

| Experiment | Compression | Structure |
| --- | --- | --- |
| [`ecg-rvq-2x`](/compressionkit/experiments/ecg-rvq-2x/) | 2× | Codec |
| [`ecg-spiht-2x`](/compressionkit/experiments/ecg-spiht-2x/) | 2× | Codec |
| [`ecg-hybrid-2x`](/compressionkit/experiments/ecg-hybrid-2x/) | 2× | Codec |
| [`ecg-rvq-4x`](/compressionkit/experiments/ecg-rvq-4x/) | 4× | Codec |
| [`ecg-spiht-4x`](/compressionkit/experiments/ecg-spiht-4x/) | 4× | Codec |
| [`ecg-hybrid-4x`](/compressionkit/experiments/ecg-hybrid-4x/) | 4× | Codec |
| [`ecg-rvq-8x`](/compressionkit/experiments/ecg-rvq-8x/) | 8× | Codec |
| [`ecg-spiht-8x`](/compressionkit/experiments/ecg-spiht-8x/) | 8× | Codec |
| [`ecg-hybrid-8x`](/compressionkit/experiments/ecg-hybrid-8x/) | 8× | Codec |
| [`ecg-rvq-16x`](/compressionkit/experiments/ecg-rvq-16x/) | 16× | Codec |
| [`ecg-spiht-16x`](/compressionkit/experiments/ecg-spiht-16x/) | 16× | Codec |
| [`ecg-hybrid-16x`](/compressionkit/experiments/ecg-hybrid-16x/) | 16× | Codec |
| [`ecg-rvq-32x`](/compressionkit/experiments/ecg-rvq-32x/) | 32× | Codec |
| [`ecg-spiht-32x`](/compressionkit/experiments/ecg-spiht-32x/) | 32× | Codec |
| [`ecg-hybrid-32x`](/compressionkit/experiments/ecg-hybrid-32x/) | 32× | Codec |
| [`ecg-rvq-64x`](/compressionkit/experiments/ecg-rvq-64x/) | 64× | Codec |
| [`ecg-spiht-64x`](/compressionkit/experiments/ecg-spiht-64x/) | 64× | Codec |
| [`ecg-hybrid-64x`](/compressionkit/experiments/ecg-hybrid-64x/) | 64× | Codec |
| [`ecg-rvq-4x-prior`](/compressionkit/experiments/ecg-rvq-4x-prior/) | 4× | Codec + entropy prior |
| [`ecg-rvq-8x-prior`](/compressionkit/experiments/ecg-rvq-8x-prior/) | 8× | Codec + entropy prior |


## Reproduce one experiment

First [install from source](/compressionkit/getting-started/) and prepare the [dataset](/compressionkit/datasets/) named on the experiment page. The runner checks dataset availability before training.

```bash
uv run compressionkit golden run ppg-rvq-4x
```

The experiment page gives the output directory, configuration, and expected deploy artifacts. Publication is a separate release action; it is not needed to evaluate a local codec.

<span id="reproduce-every-experiment-in-a-modality"></span>

## Reproduce a group

These commands run every registered experiment for the selected signal and can require substantial training time and storage.

```bash
uv run compressionkit golden run-all --modality ppg
uv run compressionkit golden run-all --modality ecg
```

To add a method or customize a recipe, see [experiment architecture](/compressionkit/experiment-architecture/) and [adding a codec family](/compressionkit/adding-a-codec-family/).
