---
icon: lucide/heart
---

# ECG (Electrocardiography)

## What is ECG?

Electrocardiography (ECG/EKG) records the electrical activity of the heart over time using electrodes placed on the skin. It's the gold standard for cardiac monitoring and is used to detect:

- **Arrhythmias** (atrial fibrillation, ventricular tachycardia)
- **Myocardial infarction** (heart attack)
- **Conduction abnormalities**
- **Heart rate and HRV**

The ECG waveform consists of the characteristic P-QRS-T complex, where the sharp QRS complex represents ventricular depolarization.

## Signal Characteristics

| Property | Value |
|----------|-------|
| Typical sampling rate | 250–500 Hz |
| Useful bandwidth | 0.5–40 Hz |
| Morphology | Sharp QRS complexes, smooth P and T waves |
| Key challenge | Preserving QRS timing and amplitude |

## ECG in compressionKIT

!!! note "TODO"
    ECG support is currently tracked as follow-on work. Some earlier experimentation remains available in `legacy/`, and a modular pipeline matching the PPG architecture is planned.

### Current State

The currently documented ECG path is not yet at parity with PPG, but the earlier ECG experiments used the same broad RVQ autoencoder approach:

- **Data source**: HDF5 files with pre-segmented ECG arrays
- **Configurations available**: 8× to 64× compression ratios
- **Model**: Same Conv2D encoder/decoder + RVQ bottleneck

### Reference Configurations

| Config | Stages | Levels | Compression | Notes |
|--------|--------|--------|-------------|-------|
| `ecg_rvq_08x_ds16_l4` | 4 | 4 | 8× | High quality |
| `ecg_rvq_16x_ds16_l2` | 4 | 2 | 16× | Balanced |
| `ecg_rvq_32x_ds16_l1` | 4 | 1 | 32× | High compression |
| `ecg_rvq_64x_ds32_l1` | 5 | 1 | 64× | Maximum compression |

### Planned Improvements

- [ ] Migrate to modular `compressionkit.datasets.ecg` module
- [ ] Add Pydantic config (`EcgRvqConfig`)
- [ ] CLI entry point (`train-ecg-rvq`)
- [ ] QRS-aware loss functions
- [ ] Wearable single-lead ECG support
