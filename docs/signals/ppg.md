---
icon: lucide/heart-pulse
---

# PPG (Photoplethysmography)

PPG is the most complete signal path in compressionKIT today. The toolkit includes a documented training flow, golden operating points, export artifacts, and a customer-ready live demo centered on this signal type.

## What is PPG?

Photoplethysmography (PPG) measures blood volume changes in the microvascular bed of tissue using an optical sensor. It's the technology behind:

- **Pulse oximeters** (SpO2 measurement)
- **Smartwatch heart rate monitors**
- **Wearable health trackers**

A PPG signal captures the pulsatile component of blood flow, producing a characteristic waveform with systolic peaks and diastolic troughs that repeat with each heartbeat.

## Signal Characteristics

| Property | Value |
|----------|-------|
| Typical sampling rate | 50–500 Hz |
| Useful bandwidth | 0.5–8 Hz |
| Default in compressionKIT | 64 Hz |
| Default frame size | 320 samples (5 seconds) |
| Morphology | Smooth, quasi-sinusoidal |

## PPG in compressionKIT

For a workflow-level view of what is supported today, see [PPG Workflow](ppg-workflow.md). For the shipped 2×, 4×, 8×, and 16× examples, see [PPG 2x-16x Examples](ppg-v1-examples.md).

### Data Source

compressionKIT loads PPG data from **EDF files** (European Data Format), specifically targeting the `Pleth` channel commonly found in polysomnography recordings such as the [MESA dataset](https://sleepdata.org/datasets/mesa).

```python
from compressionkit.datasets.ppg import load_ppg_signal

signal = load_ppg_signal(
    edf_file,
    target_rate=64,
    target_label="Pleth",
    offset_samples=192,   # skip first 3 seconds
    num_samples=7680,      # 2 minutes at 64 Hz
)
```

### Preprocessing Pipeline

The PPG preprocessing pipeline uses heliaEDGE augmentation layers:

1. **Random crop** — Extract a `frame_size` window from the full segment
2. **Layer normalization** — Zero-mean, unit-variance per sample
3. **Gaussian noise** — Additive noise augmentation (training only)

```python
from compressionkit.preprocessing.ppg import build_preprocessor, build_augmenter

preprocessor = build_preprocessor(frame_size=320, epsilon=1e-3)
augmenter = build_augmenter(noise_factor=(0.01, 0.1))
```

### Optional Filtering

Bandpass filtering can be applied to inputs and/or targets independently:

```yaml
data:
  input_filter:
    enabled: true
    low_hz: 0.5
    high_hz: 8.0
    order: 3
  target_filter:
    enabled: true
    low_hz: 0.5
    high_hz: 8.0
    order: 3
```

### Synthetic PPG Generation

For data augmentation, compressionKIT can generate synthetic PPG signals via physioKIT:

```yaml
data:
  synthetic_mix:
    enabled: true
    fraction: 0.1          # 10% synthetic data
    heart_rate_bpm: [50, 120]
    frequency_modulation: [0.1, 0.5]
    ibi_randomness: [0.02, 0.2]
```

### Evaluation Metrics

PPG reconstruction quality is assessed using:

- **MSE** — Mean squared error
- **PRD** — Percent root-mean-square difference
- **Cosine similarity** — Waveform shape preservation
- **Band-limited metrics** — Metrics computed after bandpass filtering to 0.5–8 Hz
- **physioKIT alignment** — HR, RMSSD, SDNN comparison between original and reconstructed signals
