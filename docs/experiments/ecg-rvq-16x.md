---
icon: lucide/activity
---

# `ecg-rvq-16x`

## Overview

- **Modality**: ECG
- **Family**: `codec`
- **Compression ratio**: 16×
- **Sample rate**: 256 Hz
- **Recipe**: `train-ecg-rvq`
- **Config**: [`configs/ecg_rvq_256hz_16x_golden.yaml`](https://github.com/AmbiqAI/compressionkit/blob/main/configs/ecg_rvq_256hz_16x_golden.yaml)
- **Run name**: `ecg_rvq_256hz_16x_golden`
- **HuggingFace**: [`AmbiqAI/compressionkit-ecg-16x`](https://huggingface.co/AmbiqAI/compressionkit-ecg-16x)

## Dataset & License

- **Dataset**: [PTB-XL](https://physionet.org/content/ptb-xl/) (`dataset_id: ptb-xl`)
- **License**: CC BY 4.0 (open)
- **Notes**: Auto-downloaded on first use.

The lifecycle runner pre-flights dataset availability before training (see #26 and the
[dataset contract](../api/datasets.md)).

## Reproduction

```bash
# Single command, end-to-end.
uv run compressionkit golden run ecg-rvq-16x

# Publish the deploy package to HuggingFace (requires HF_TOKEN).
uv run compressionkit golden run ecg-rvq-16x --publish
```

Results land under `results/ecg_rvq_256hz_16x_golden/`; deploy artifacts under `results/ecg_rvq_256hz_16x_golden/deploy/`.

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

## Customization Notes

- Tweak the YAML to explore neighbouring operating points; copy the file before editing.
- For new recipes, prefer the `compressionkit/recipes/` package recipes as a starting point.
- To resume publishing without retraining, pass `--skip-train` to `compressionkit golden run ecg-rvq-16x --publish`.
