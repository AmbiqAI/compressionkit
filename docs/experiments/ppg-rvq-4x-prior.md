---
icon: lucide/heart-pulse
---

# `ppg-rvq-4x-prior`

## Overview

- **Modality**: PPG
- **Family**: `two_stage`
- **Compression ratio**: 4×
- **Sample rate**: 64 Hz
- **Recipe**: `train-rvq-prior`
- **Config**: [`configs/ppg_rvq_64hz_04x_golden_prior.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_04x_golden_prior.yaml)
- **Run name**: `ppg_rvq_64hz_04x_golden`
- **HuggingFace**: [`Ambiq/compressionkit-ppg-4x`](https://huggingface.co/Ambiq/compressionkit-ppg-4x)

## Dataset & License

- **Dataset**: [MESA (NSRR)](https://sleepdata.org/datasets/mesa) (`dataset_id: mesa`)
- **License**: NSRR Data Use Agreement (restricted)
- **Notes**: Requires an NSRR token. Set ``NSRR_TOKEN`` and call ``MesaDataset(...).download()``.

The lifecycle runner pre-flights dataset availability before training (see #26 and the
[dataset contract](../api/datasets.md)).

## Reproduction

```bash
# Single command, end-to-end.
uv run compressionkit golden run ppg-rvq-4x-prior

# Publish the deploy package to HuggingFace (requires HF_TOKEN).
uv run compressionkit golden run ppg-rvq-4x-prior --publish
```

Results land under `results/ppg_rvq_64hz_04x_golden/`; deploy artifacts under `results/ppg_rvq_64hz_04x_golden/deploy/`.

## Parent codec

This entry is the entropy-prior stage paired with [`ppg-rvq-4x`](ppg-rvq-4x.md).
Codec and prior artifacts publish to the same HuggingFace repo
([`Ambiq/compressionkit-ppg-4x`](https://huggingface.co/Ambiq/compressionkit-ppg-4x)).

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
- `prior_int8.tflite` / `prior_int8.h` / `prior_manifest.json` — entropy prior (two-stage only).

## Customization Notes

- Tweak the YAML to explore neighbouring operating points; copy the file before editing.
- For new recipes, prefer the `compressionkit/recipes/` package recipes as a starting point.
- To resume publishing without retraining, pass `--skip-train` to `compressionkit golden run ppg-rvq-4x-prior --publish`.
