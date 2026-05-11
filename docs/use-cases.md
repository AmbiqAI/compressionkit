---
icon: lucide/target
---

# Use Cases & Customer Value

compressionKIT exists for one reason: **help our customers ship continuous‑sensing products that were previously impossible on a wearable power and memory budget.** This page summarizes where the toolkit adds concrete value and quantifies the savings a typical integration delivers.

---

## Who benefits

<div class="grid cards" markdown>

-   :material-watch-variant:{ .lg .middle } **Consumer wearables**

    ---

    Smartwatches, rings, patches, earbuds — any device with a PPG or ECG sensor and a tight battery / flash budget.

-   :material-hospital-box-outline:{ .lg .middle } **Remote patient monitoring**

    ---

    Multi-day Holter patches, hospital-at-home kits, post-op monitors where storage and cellular cost dominate the BOM.

-   :material-sleep:{ .lg .middle } **Sleep & wellness**

    ---

    Overnight PPG for HR, HRV, and sleep staging on a coin cell, with a week of memory headroom.

-   :material-flask-outline:{ .lg .middle } **Clinical & pharma research**

    ---

    Large ambulatory studies where raw-waveform retention is required but storage costs scale with the cohort.

-   :material-antenna:{ .lg .middle } **Industrial / animal health**

    ---

    Livestock, equine, and telemetry applications operating over LoRa, sub-GHz, or NB-IoT links.

-   :material-chip:{ .lg .middle } **OEMs building on Ambiq silicon**

    ---

    Drop-in INT8 encoder + C headers for Apollo-class MCUs, with server-side decoders and bring-up sample data included.

</div>

---

## Headline benefits

<div class="grid cards" markdown>

-   :material-compress:{ .lg .middle } **2× – 32×**

    Compression operating points, one codec

-   :material-memory:{ .lg .middle } **INT8**

    Quantized on-device encoder

-   :material-vector-line:{ .lg .middle } **≥ 0.96 cosine**

    Waveform fidelity at 32×

-   :material-heart-pulse:{ .lg .middle } **< 1 bpm**

    Heart-rate error (PPG, all ratios)

</div>

!!! tip "Design principle"
    The compression is applied **once, at the sensor**. Every downstream system — flash, radio, gateway, cloud ingestion, ML training — inherits the savings automatically.

---

## Benefit 1 — On‑device memory

compressionKIT lets customers keep **much more continuous waveform** in the same MCU flash partition or PSRAM region.

### PPG — continuous recording capacity

Frame = 5 s @ 64 Hz, 16‑bit. RVQ bit budgets: 2560 / 1280 / 640 / 320 / 160 bits per frame.

| Storage | Raw | 2× | 4× | 8× | 16× | 32× |
|---------|----:|---:|---:|---:|----:|----:|
| **128 KB** | 1.7 h | 3.4 h | 6.8 h | 13.7 h | 27.3 h | **~2.3 days** |
| **1 MB** | 13.7 h | 1.1 days | 2.3 days | 4.6 days | 9.1 days | **~18 days** |
| **8 MB PSRAM** | 4.6 days | 9.1 days | 18 days | 36 days | 73 days | **~5 months** |

### ECG — continuous recording capacity

Frame = 2 s @ 256 Hz, 16‑bit. RVQ bit budgets: 4096 / 2048 / 1024 / 512 / 256 bits per frame.

| Storage | Raw | 2× | 4× | 8× | 16× | 32× |
|---------|----:|---:|---:|---:|----:|----:|
| **128 KB** | 0.3 h | 0.6 h | 1.1 h | 2.3 h | 4.6 h | **9.1 h** |
| **1 MB** | 2.1 h | 4.3 h | 8.5 h | 17 h | 34 h | **2.8 days** |
| **8 MB PSRAM** | 17.1 h | 1.4 days | 2.8 days | 5.7 days | 11 days | **23 days** |

!!! example "What this unlocks"
    At **32× PPG compression**, a single 1 MB flash partition holds **two weeks** of continuous PPG — enough for a full RPM study without a single BLE sync.

---

## Benefit 2 — Radio, bandwidth & energy

Radio airtime is the dominant battery cost in most wearable designs. compressionKIT shrinks the per‑second payload directly.

| Signal | Raw | 2× | 4× | 8× | 16× | 32× |
|--------|----:|---:|---:|---:|----:|----:|
| PPG (64 Hz) | 1024 bps | 512 bps | 256 bps | 128 bps | 64 bps | **32 bps** |
| ECG (256 Hz) | 4096 bps | 2048 bps | 1024 bps | 512 bps | 256 bps | **128 bps** |
| 3‑lead ECG | 12.3 kbps | 6.1 kbps | 3.1 kbps | 1.5 kbps | 768 bps | **384 bps** |

At aggressive ratios this enables:

- **BLE‑advertising‑only designs** — a single advertisement or connection event per minute carries the full waveform
- **Sub‑GHz / LoRa / NB‑IoT telemetry** for continuous cardiac data
- **Longer connection intervals** → fewer wake‑ups → months on a CR2032

---

## Benefit 3 — Cloud storage & ingestion

Per‑patient, per‑year waveform retention — and what that scales to at fleet level:

| Deployment | Raw | @ 8× | @ 32× |
|------------|----:|-----:|------:|
| 1 patient, PPG 24/7 | 4.0 GB | 510 MB | **130 MB** |
| 1 patient, ECG 24/7 | 16.1 GB | 2.0 GB | **520 MB** |
| 100k patient fleet, ECG 24/7 | 1.6 PB | 200 TB | **52 TB** |

→ ~30× lower S3 / object‑store cost, faster ETL, and — critically — **faster model training over the full fleet** because the input pipeline is no longer bandwidth‑bound.

---

## Benefit 4 — Fidelity that survives downstream algorithms

Unlike naive downsampling or bit‑depth truncation, compressionKIT learns the signal manifold with a **derivative‑aware loss** that preserves slopes and peak morphology.

- **Cosine similarity ≥ 0.96** even at 32× on both PPG and ECG
- **HR error grows gracefully** with compression (≤ 0.22 bpm MAE up through 32× on PPG)
- **HRV error stays in wellness range** up through 16× — suitable for HRV‑driven dashboards
- ECG **QRS morphology** is visually indistinguishable at 8× — see the [reconstruction gallery](signals/ppg-v1-examples.md)

→ Existing HR detectors, SpO₂ estimators, and arrhythmia classifiers keep working on reconstructed data with minimal re‑tuning.

---

## Choosing an operating point

Every product has a different tradeoff. These are the default recommendations we make to customers:

| Goal | PPG | ECG |
|------|-----|-----|
| **Near‑lossless archival** | 2× | 2× – 4× |
| **Clinical‑grade HR / HRV / rhythm** | 4× – 8× | 4× – 8× |
| **Wellness / ambulatory monitoring** | 8× – 16× | 8× – 16× |
| **Event logging / screening / triage** | 16× – 32× | 16× – 32× |
| **Ultra‑low bandwidth telemetry** | 32× | 32× |

All five operating points ship as golden configs. See the [PPG Model Zoo](models/ppg.md) and [ECG Model Zoo](models/ecg.md) for measured metrics at each ratio.

---

## Getting started as a customer

1. **Evaluate** — pull the v1.0 golden models and run them on your own data ([Getting Started](getting-started.md))
2. **Retrain** — use the YAML config system to fine‑tune on your sensor / front‑end characteristics
3. **Deploy** — drop the `encoder.tflite` + `encoder.h` + `codebook.h` bundle into an Apollo reference project
4. **Integrate** — use the Keras decoder on the cloud side or ship it with the companion mobile app

Your Ambiq FAE can walk through the bring‑up in a single session. See the [CLI reference](cli.md) for the full workflow.
