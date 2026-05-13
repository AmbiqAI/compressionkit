---
icon: lucide/heart-pulse
---

# `ppg-rvq-32x`

## Overview

- **Modality**: PPG
- **Family**: `codec`
- **Compression ratio**: 32×
- **Sample rate**: 64 Hz
- **Recipe**: `train-ppg-rvq`
- **Config**: [`configs/ppg_rvq_64hz_32x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_32x_golden.yaml)
- **Run name**: `ppg_rvq_64hz_32x_golden`
- **HuggingFace**: [`AmbiqAI/compressionkit-ppg-32x`](https://huggingface.co/AmbiqAI/compressionkit-ppg-32x)

## Dataset & License

- **Dataset**: [MESA (NSRR)](https://sleepdata.org/datasets/mesa) (`dataset_id: mesa`)
- **License**: NSRR Data Use Agreement (restricted)
- **Notes**: Requires an NSRR token. Set ``NSRR_TOKEN`` and call ``MesaDataset(...).download()``.

The lifecycle runner pre-flights dataset availability before training (see #26 and the
[dataset contract](../api/datasets.md)).

## Reproduction

```bash
# Single command, end-to-end.
uv run compressionkit golden run ppg-rvq-32x

# Publish the deploy package to HuggingFace (requires HF_TOKEN).
uv run compressionkit golden run ppg-rvq-32x --publish
```

Results land under `results/ppg_rvq_64hz_32x_golden/`; deploy artifacts under `results/ppg_rvq_64hz_32x_golden/deploy/`.

## Evaluation Metrics

See the modality model zoo for the full metrics table:

- [PPG models](../models/ppg.md)

Each run writes `quality_scorecard.json` and `summary.json` under its `results/<run>/`.

## Deploy Artifacts

Every successful run produces the canonical edge deploy package:

- `encoder.tflite` / `encoder.h` — INT8 encoder.
- `decoder.tflite` / `decoder.h` — decoder (float32 + optional INT8).
- `codebook.npz` / `codebook.h` — RVQ codebook tables.
- `sample_stimulus.npz` — license-safe input/output reference frames.
- `model_card.json`, `deploy_manifest.json` — metadata.

## Customization Notes

- Tweak the YAML to explore neighbouring operating points; copy the file before editing.
- For new recipes, prefer the `compressionkit/recipes/` package recipes as a starting point.
- To resume publishing without retraining, pass `--skip-train` to `compressionkit golden run ppg-rvq-32x --publish`.
