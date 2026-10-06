---
title: "Supported Signal Types"
description: "Choose a PPG or ECG signal workflow and inspect frame and sample-rate requirements."
---


compressionKIT is designed for **physiological signal compression** — signals that originate from the human body and are captured by wearable or clinical sensors. These signals share common properties that make them good candidates for learned compression:

- **Quasi-periodic** — heartbeat-driven repetitive structure
- **Low bandwidth** — typically 0.5–40 Hz of useful content
- **Low sampling rates** — 64–500 Hz (compared to audio at 44 kHz)
- **High redundancy** — temporal and morphological redundancy across beats

## Currently Supported

| Signal | Module | Sampling Rate | Frame Size | Status |
|--------|--------|---------------|------------|--------|
| [PPG](/compressionkit/signals/ppg/) | `compressionkit.datasets.ppg` | 64 Hz | 320 (5 s) | RVQ, SPIHT and hybrid configurations — [models](/compressionkit/models/ppg/) |
| [ECG](/compressionkit/signals/ecg/) | `compressionkit.datasets.ptbxl` | 256 Hz | 512 (2 s) | RVQ, SPIHT and hybrid configurations — [models](/compressionkit/models/ecg/) |

## PPG Section

These pages cover the PPG signal path:

1. [PPG](/compressionkit/signals/ppg/) for signal context and preprocessing details.
2. [PPG Workflow](/compressionkit/signals/ppg-workflow/) for the end-to-end supported task.
3. [PPG Models (v1.1 exports)](/compressionkit/models/ppg/) for the five golden reference operating points.
4. [PPG Codec Demo](/compressionkit/demo/ppg-codec/) for the browser and hardware evaluation experience.

## ECG Section

These pages cover the ECG signal path:

1. [ECG](/compressionkit/signals/ecg/) for signal context and preprocessing details.
2. [ECG Models (v1.1 exports)](/compressionkit/models/ecg/) for the six golden reference operating points.

## Signal Properties Comparison

| Property | PPG | ECG |
|----------|-----|-----|
| Typical bandwidth | 0.5–8 Hz | 0.5–40 Hz |
| Morphology | Smooth, pulse-shaped | Sharp QRS complexes |
| Compression challenge | Preserve pulse shape | Preserve QRS timing & amplitude |
| Example downstream measurements | HR, SpO2, HRV | HR, HRV, ST-segment, arrhythmia |
| Typical sensor | Optical (wrist/finger) | Electrode (chest/wrist) |
