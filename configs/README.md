# Configs

This directory contains the canonical configuration files for reproducing golden training runs.

## Naming Convention

```
{modality}_rvq_{sample_rate}hz_{cr}x_golden.yaml
```

Example: `ecg_rvq_256hz_32x_golden.yaml` — ECG RVQ codec at 256 Hz, 32× compression ratio.

## Golden Configs (Primary)

These reproduce the published golden models:

| Modality | CRs | Pattern |
|----------|-----|---------|
| ECG | 2×, 4×, 8×, 16×, 32×, 64× | `ecg_rvq_256hz_{cr}x_golden.yaml` |
| PPG | 2×, 4×, 8×, 16×, 32× | `ppg_rvq_64hz_{cr}x_golden.yaml` |
| PPG (H5) | 2×, 4×, 8×, 16× | `ppg_h5_rvq_{cr}x_mixed_golden_sched.yaml` |

## Variant Configs

- `*_spectral_lo.yaml` — Golden + spectral loss with lower weighting (used in evaluation sweeps)
- `*_hier.yaml` — Golden + hierarchical codebook training
- `*_denoise.yaml` — Golden + denoising objective

## Supporting Configs

- `ecg_decimate_256hz_*.yaml` — Decimation baselines for CR comparison
- `ppg_two_stream_*.yaml` — Two-stream architecture variants
- `ppg_rvq_64hz_*_unified_expA2_200ep.yaml` — Extended training experiments
- `ppg_rvq_smoketest.yaml` — Fast CI smoke test

## Archive

Experimental configs from development are preserved in `configs/archive/`.
These are not needed to reproduce golden results but may be useful for reference.
