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

| Signal | Module | Sampling Rate | Frame Size | Notes |
|--------|--------|---------------|------------|-------|
| [PPG](ppg.md) | `compressionkit.datasets.ppg` | 64 Hz | 320 (5s) | Production-ready with documented workflow, v1 examples, and demo |
| [ECG](ecg.md) | TODO | 250–500 Hz | 2048 | Modular migration planned |

## PPG Section

These pages belong specifically to the PPG signal path:

If you are starting with the current production flow, use the PPG pages in this order:

1. [PPG](ppg.md) for signal context and preprocessing details.
2. [PPG Workflow](ppg-workflow.md) for the end-to-end supported task.
3. [PPG 2x-16x Examples](ppg-v1-examples.md) for the four reference operating points.
4. [PPG Codec Demo](../demo/ppg-codec.md) for the customer-facing browser and hardware experience.

## ECG Section

These pages belong specifically to the ECG signal path:

1. [ECG](ecg.md) for the current status, prior experiment context, and planned migration work.

ECG does not yet have workflow, examples, or demo pages matching the current PPG documentation depth.

## Signal Properties Comparison

| Property | PPG | ECG |
|----------|-----|-----|
| Typical bandwidth | 0.5–8 Hz | 0.5–40 Hz |
| Morphology | Smooth, pulse-shaped | Sharp QRS complexes |
| Compression challenge | Preserve pulse shape | Preserve QRS timing & amplitude |
| Key clinical metrics | HR, SpO2, HRV | HR, HRV, ST-segment, arrhythmia |
| Typical sensor | Optical (wrist/finger) | Electrode (chest/wrist) |
