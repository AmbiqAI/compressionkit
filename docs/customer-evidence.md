---
icon: lucide/clipboard-check
---
# Customer Evidence Summary

This generated page summarizes the v1 release evidence in the terms a product or engineering team usually needs before selecting an operating point. It intentionally combines compression level, fidelity, physiological utility, stitching behavior, and deploy footprint instead of leading with one waveform metric.

## Release decision overview

| Signal | Published CR range | Recommended operating read | Evidence surfaces |
|---|---:|---|---|
| PPG | 2x-32x | 2x-8x for tight HR/HRV preservation; 16x-32x when storage or radio budget dominates. | truth PRD, PRDN-noise, HR/peak timing, band error, stitching, deploy footprint |
| ECG | 2x-64x | 4x-16x for morphology-focused use; 32x-64x for bandwidth-limited capture paths. | truth PRD, PRDN-noise, HR/peak timing, band error, stitching, deploy footprint |

## Compression and deploy ladder

The tables below are generated from local golden `quality_scorecard.json` files and deploy manifests. Truth PRD is measured against the clean reference when available; faithful PRD is measured against the recorded input; PRDN-noise tracks residual noise-normalized distortion. Edge payload is the encoder TFLite, decoder TFLite, and codebook NPZ size, excluding validation samples and documentation files.

### PPG release ladder

| CR | Frame | N | Truth PRD% | Faithful PRD% | PRDN-noise% | HR MAE bpm | Pulse-band error | Coherence | Seam ratio | Edge payload | Encoder | Decoder | Codebook |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2x | 5000 ms | 9788 | 0.62 | 2.59 | 0.15 | 0.19 | 0.0155 | 0.9928 | - | 88 KiB | 5 KiB | 19 KiB | 65 KiB |
| 4x | 5000 ms | 9789 | 1.14 | 3.44 | 0.23 | 0.29 | 0.5885 | 0.9861 | - | 147 KiB | 28 KiB | 54 KiB | 65 KiB |
| 8x | 5000 ms | 9789 | 2.66 | 5.10 | 0.64 | 0.49 | 0.6623 | 0.9565 | - | 196 KiB | 39 KiB | 92 KiB | 65 KiB |
| 16x | 5000 ms | 9789 | 5.75 | 7.54 | 0.79 | 0.88 | 12.5676 | 0.8991 | - | 300 KiB | 54 KiB | 180 KiB | 65 KiB |
| 32x | 5000 ms | 6336 | 12.35 | 12.08 | 0.20 | 1.79 | 0.0356 | 0.8014 | - | 267 KiB | 54 KiB | 180 KiB | 33 KiB |

### ECG release ladder

| CR | Frame | N | Truth PRD% | Faithful PRD% | PRDN-noise% | HR MAE bpm | QRS-band error | Coherence | Seam ratio | Edge payload | Encoder | Decoder | Codebook |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2x | 2000 ms | 1000 | 1.38 | 2.31 | 0.00 | 0.48 | 0.0081 | 1.0000 | - | 88 KiB | 5 KiB | 19 KiB | 65 KiB |
| 4x | 2000 ms | 1000 | 2.33 | 3.63 | 0.00 | 0.95 | 0.0104 | 1.0000 | - | 147 KiB | 28 KiB | 54 KiB | 65 KiB |
| 8x | 2000 ms | 1000 | 3.84 | 6.03 | 0.00 | 1.39 | 0.0144 | 1.0000 | - | 196 KiB | 39 KiB | 92 KiB | 65 KiB |
| 16x | 2000 ms | 1000 | 7.28 | 9.72 | 0.04 | 2.05 | 0.0246 | 1.0000 | - | 300 KiB | 54 KiB | 180 KiB | 65 KiB |
| 32x | 2000 ms | 1000 | 13.27 | 14.59 | 0.78 | 2.34 | 0.0497 | 1.0000 | 0.176 | 267 KiB | 54 KiB | 180 KiB | 33 KiB |
| 64x | 2000 ms | 1000 | 20.49 | 21.05 | 3.53 | 3.00 | 0.0835 | 1.0000 | - | 397 KiB | 76 KiB | 289 KiB | 33 KiB |

## How to use this page

- Start with the CR ladder to identify feasible compression levels for memory, radio, or storage targets.
- Use Truth PRD and PRDN-noise together when judging noisy recordings; faithful PRD alone can penalize codecs that suppress artifact energy.
- Use HR MAE and band error as utility checks before moving to waveform examples and full scorecards.
- Use seam ratio as the first-pass long-recording stitching check; model pages provide the visual and method-specific context.
- Use edge-payload columns for target budgeting; parameter and MAC summaries can be added when those are reliably exported for every bundle.

Detailed modality pages remain the source for waveform examples, robustness plots, and reproduction commands: [PPG models](models/ppg.md), [ECG models](models/ecg.md), [PPG CR vs fidelity](methods/cr_vs_fidelity_ppg.md), and [ECG CR vs fidelity](methods/cr_vs_fidelity_ecg.md).
