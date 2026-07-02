---
icon: lucide/activity
---

# `ecg-rvq-8x-prior`

## Overview

- **Modality**: ECG
- **Structure**: `two_stage`
- **Compression ratio**: 8×
- **Sample rate**: 256 Hz
- **Recipe**: `train-rvq-prior`
- **Config**: [`configs/ecg_rvq_256hz_08x_golden_prior.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ecg_rvq_256hz_08x_golden_prior.yaml)
- **Run name**: `ecg_rvq_256hz_08x_golden`
- **HuggingFace**: [`Ambiq/compressionkit-ecg-8x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-8x-v1.0)

## Dataset & License

- **Dataset**: [PTB-XL](https://physionet.org/content/ptb-xl/) (`dataset_id: ptb-xl`)
- **License**: CC BY 4.0 (open)
- **Notes**: Auto-downloaded on first use.

The lifecycle runner pre-flights dataset availability before training (see #26 and the
[dataset contract](../api/datasets.md)).

## Reproduction

```bash
# Single command, end-to-end.
uv run compressionkit golden run ecg-rvq-8x-prior

# Publish the deploy package to HuggingFace (requires HF_TOKEN).
uv run compressionkit golden run ecg-rvq-8x-prior --publish
```

Results land under `results/ecg_rvq_256hz_08x_golden/`; deploy artifacts under `results/ecg_rvq_256hz_08x_golden/deploy/`.

## Parent codec

This entry is the entropy-prior stage paired with [`ecg-rvq-8x`](ecg-rvq-8x.md).
Codec and prior artifacts publish to the same HuggingFace repo
([`Ambiq/compressionkit-ecg-8x`](https://huggingface.co/Ambiq/compressionkit-ecg-8x)).

## Evaluation Metrics

See the modality model zoo for the full metrics table:

- [ECG models](../models/ecg.md)

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
- To resume publishing without retraining, pass `--skip-train` to `compressionkit golden run ecg-rvq-8x-prior --publish`.
