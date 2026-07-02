---
icon: lucide/activity
---

# `ecg-hybrid-4x`

## Overview

- **Modality**: ECG
- **Structure**: `codec`
- **Compression ratio**: 4×
- **Sample rate**: 256 Hz
- **Config**: — (operating point is fully declared in the registry; no training config)
- **Run name**: `ecg_hybrid_256hz_04x_golden`
- **HuggingFace**: [`Ambiq/compressionkit-ecg-hybrid-4x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-hybrid-4x-v1.0)

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

- `spiht_config.json` / `spiht_app_config.h` — codec parameters (language-neutral + C header).
- `c_sources/spiht.[ch]` — portable C99 SPIHT reference.
- `sample_stimulus.npz` / `reference_vectors.npz` — license-safe test frames and known-good encode/decode vectors.
- `model_card.json`, `deploy_manifest.json` — metadata.
- `denoiser_gain_model.keras`, `hybrid_manifest.json` — learned wavelet-gain denoiser (trained weights) and pipeline stage order.

## Customization Notes

- This operating point is declared directly in `compressionkit/experiments/registry.py` (no training YAML) — add a new registry entry to explore a neighbouring operating point.
- To resume publishing without retraining, pass `--skip-train` to `compressionkit golden run ecg-hybrid-4x --publish`.
