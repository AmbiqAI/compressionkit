---
icon: lucide/heart-pulse
---

# `ppg-rvq-8x-prior`

## Overview

- **Modality**: PPG
- **Structure**: `two_stage`
- **Compression ratio**: 8×
- **Sample rate**: 64 Hz
- **Recipe**: `train-rvq-prior`
- **Config**: [`configs/ppg_rvq_64hz_08x_golden_prior.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ppg_rvq_64hz_08x_golden_prior.yaml)
- **Run name**: `ppg_rvq_64hz_08x_golden`
- **HuggingFace**: [`Ambiq/compressionkit-ppg-8x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-8x-v1.0)

## Dataset & License

- **Dataset**: [Open unified PPG v1](../datasets.md) (`dataset_id: ppg-unified-strict-sanitize-v1`)
- **License**: Open (BIDMC, BUT PPG, PPG-DaLiA, WESAD — mixed open licenses, no restricted-access dependency)
- **Notes**: Sources: BIDMC, BUT PPG, PPG-DaLiA, and WESAD. Published v1 PPG goldens are MESA-free. Build the cache with `scripts/build_ppg_cache.py`.

The lifecycle runner pre-flights dataset availability before training (see #26 and the
[dataset contract](../api/datasets.md)).

## Reproduction

```bash
# Single command, end-to-end.
uv run compressionkit golden run ppg-rvq-8x-prior

# Publish the deploy package to HuggingFace (requires HF_TOKEN).
uv run compressionkit golden run ppg-rvq-8x-prior --publish
```

Results land under `results/ppg_rvq_64hz_08x_golden/`; deploy artifacts under `results/ppg_rvq_64hz_08x_golden/deploy/`.

## Parent codec

This entry is the entropy-prior stage paired with [`ppg-rvq-8x`](ppg-rvq-8x.md).
Codec and prior artifacts publish to the same HuggingFace repo
([`Ambiq/compressionkit-ppg-8x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-8x-v1.0)).

## Evaluation Metrics

See the modality model zoo for the full metrics table:

- [PPG models](../models/ppg.md)

Each run writes `quality_scorecard.json` and `summary.json` under its `results/<run>/`.

## Deploy Artifacts

Every successful run produces the canonical edge deploy package:

- `encoder.tflite` / `encoder.h` — INT8 encoder.
- `encoder.keras` — float32 Python reference encoder.
- `decoder.tflite` / `decoder.h` — decoder (float32 + optional INT8).
- `decoder.keras` — float32 Python reference decoder.
- `codebook.npz` / `codebook.h` — RVQ codebook tables.
- `sample_stimulus.npz` — license-safe input/output reference frames.
- `model_card.json`, `deploy_manifest.json` — metadata.
- `prior_int8.tflite` / `prior_int8.h` / `prior_manifest.json` — entropy prior (two-stage only).

## Customization Notes

- Tweak the YAML to explore neighbouring operating points; copy the file before editing.
- For new recipes, prefer the `compressionkit/recipes/` package recipes as a starting point.
- To resume publishing without retraining, pass `--skip-train` to `compressionkit golden run ppg-rvq-8x-prior --publish`.
