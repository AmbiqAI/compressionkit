---
icon: lucide/activity
---

# `ecg-hybrid-4x`

## Overview

- **Modality**: ECG
- **Structure**: `codec`
- **Compression ratio**: 4×
- **Sample rate**: 256 Hz
- **Recipe**: `None`
- **Config**: [`None`](https://github.com/AmbiqAI/compressionkit/blob/main/None)
- **Run name**: `ecg_hybrid_256hz_04x_golden`
- **HuggingFace**: [`Ambiq/compressionkit-ecg-hybrid-4x`](https://huggingface.co/Ambiq/compressionkit-ecg-hybrid-4x)

## Dataset & License

- **Dataset**: [PTB-XL](https://physionet.org/content/ptb-xl/) (`dataset_id: ptb-xl`)
- **License**: CC BY 4.0 (open)
- **Notes**: Auto-downloaded on first use.

The lifecycle runner pre-flights dataset availability before training (see #26 and the
[dataset contract](../api/datasets.md)).

## Reproduction

```bash
# Single command, end-to-end.
uv run compressionkit golden run ecg-hybrid-4x

# Publish the deploy package to HuggingFace (requires HF_TOKEN).
uv run compressionkit golden run ecg-hybrid-4x --publish
```

Results land under `results/ecg_hybrid_256hz_04x_golden/`; deploy artifacts under `results/ecg_hybrid_256hz_04x_golden/deploy/`.

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
- To resume publishing without retraining, pass `--skip-train` to `compressionkit golden run ecg-hybrid-4x --publish`.
