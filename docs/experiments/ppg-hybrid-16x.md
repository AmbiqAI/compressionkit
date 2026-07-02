---
icon: lucide/heart-pulse
---

# `ppg-hybrid-16x`

## Overview

- **Modality**: PPG
- **Structure**: `codec`
- **Compression ratio**: 16×
- **Sample rate**: 64 Hz
- **Config**: — (operating point is fully declared in the registry; no training config)
- **Run name**: `ppg_hybrid_64hz_16x_golden`
- **HuggingFace**: [`Ambiq/compressionkit-ppg-hybrid-16x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-hybrid-16x-v1.0)

## Dataset & License

- **Dataset**: [Open unified PPG v1](../datasets.md) (`dataset_id: ppg-unified-strict-sanitize-v1`)
- **License**: Open (BIDMC, BUT PPG, PPG-DaLiA, WESAD — mixed open licenses, no restricted-access dependency)
- **Notes**: Sources: BIDMC, BUT PPG, PPG-DaLiA, and WESAD. Published v1 PPG goldens are MESA-free. Build the cache with `scripts/build_ppg_cache.py`.

The lifecycle runner pre-flights dataset availability before training (see #26 and the
[dataset contract](../api/datasets.md)).

## Reproduction

```bash
# Single command, end-to-end.
uv run compressionkit golden run ppg-hybrid-16x

# Publish the deploy package to HuggingFace (requires HF_TOKEN).
uv run compressionkit golden run ppg-hybrid-16x --publish
```

Results land under `results/ppg_hybrid_64hz_16x_golden/`; deploy artifacts under `results/ppg_hybrid_64hz_16x_golden/deploy/`.

## Evaluation Metrics

See the modality model zoo for the full metrics table:

- [PPG models](../models/ppg.md)

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
- To resume publishing without retraining, pass `--skip-train` to `compressionkit golden run ppg-hybrid-16x --publish`.
