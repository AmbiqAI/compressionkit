---
icon: lucide/activity
---

# Supported Signal Types

compressionKIT is designed for **physiological signal compression** — signals that originate from the human body and are captured by wearable or clinical sensors. These signals share common properties that make them good candidates for learned compression:

- **Quasi-periodic** — heartbeat-driven repetitive structure
- **Low bandwidth** — typically 0.5–40 Hz of useful content
- **Low sampling rates** — 64–500 Hz (compared to audio at 44 kHz)
- **High redundancy** — temporal and morphological redundancy across beats

## Currently Supported

| Signal | Module | Sampling Rate | Frame Size | Status |
|--------|--------|---------------|------------|--------|
| [PPG](ppg.md) | `compressionkit.datasets.mesa` | 64 Hz | 320 (5 s) | Production — [golden models](../models/ppg.md) |
| [ECG](ecg.md) | `compressionkit.datasets.ptbxl` | 256 Hz | 512 (2 s) | Production — [golden models](../models/ecg.md) |

## PPG Section

These pages cover the PPG signal path:

1. [PPG](ppg.md) for signal context and preprocessing details.
2. [PPG Workflow](ppg-workflow.md) for the end-to-end supported task.
3. [PPG Models (v1.0)](../models/ppg.md) for the five golden reference operating points.
4. [PPG Codec Demo](../demo/ppg-codec.md) for the customer-facing browser and hardware experience.

## ECG Section

These pages cover the ECG signal path:

1. [ECG](ecg.md) for signal context and preprocessing details.
2. [ECG Models (v1.0)](../models/ecg.md) for the five golden reference operating points.

## Signal Properties Comparison

| Property | PPG | ECG |
|----------|-----|-----|
| Typical bandwidth | 0.5–8 Hz | 0.5–40 Hz |
| Morphology | Smooth, pulse-shaped | Sharp QRS complexes |
| Compression challenge | Preserve pulse shape | Preserve QRS timing & amplitude |
| Key clinical metrics | HR, SpO2, HRV | HR, HRV, ST-segment, arrhythmia |
| Typical sensor | Optical (wrist/finger) | Electrode (chest/wrist) |
