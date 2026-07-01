---
icon: lucide/target
---

# Use Cases & Value

compressionKIT helps wearable teams keep more waveform data without paying the
full memory, radio, and cloud-ingestion cost of raw sampling. This page summarizes
where compression usually creates value and gives order-of-magnitude sizing
examples for product planning.

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

-   :material-compress:{ .lg .middle } **2× – 64×**

    Compression operating points, one codec

-   :material-memory:{ .lg .middle } **INT8**

    Quantized on-device encoder

-   :material-vector-line:{ .lg .middle } **≥ 0.96 cosine**

    Waveform fidelity through aggressive ECG/PPG operating points

-   :material-heart-pulse:{ .lg .middle } **< 1 bpm**

    PPG HR error across the published RVQ CR ladder

</div>

!!! tip "Design principle"
    The compression is applied **once, at the sensor**. Every downstream system — flash, radio, gateway, cloud ingestion, ML training — inherits the savings automatically.

---

## Benefit 1 — On‑device memory

compressionKIT lets devices keep **much more continuous waveform** in the same MCU flash partition or PSRAM region.

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

Unlike naive downsampling or bit-depth truncation, compressionKIT evaluates codecs
against waveform and physiology-aware scorecards. The published RVQ bundles are
the easiest packages to load today; SPIHT and hybrid lanes provide standardized
comparison points for clean, noisy, and artifact-heavy regimes.

- PPG and ECG RVQ bundles include measured PRD, MSE, cosine, and modality-specific physiological metrics.
- Noise-aware scorecards separate clean-frame faithfulness from recoverable signal quality under empirical noise and artifacts.
- SPIHT is a strong clean-signal baseline; hybrid lanes are important when wearable noise or contact artifacts dominate.
- ECG and PPG model pages now include the current crossover heatmaps and links to CR-vs-fidelity tables.

→ The practical question becomes which operating point preserves the observables
your product needs, not which single aggregate metric is smallest.

---

## Choosing an operating point

Every product has a different tradeoff. These are the default recommendations for each operating point:

| Goal | PPG | ECG |
|------|-----|-----|
| **Near‑lossless archival** | 2× | 2× – 4× |
| **Clinical‑grade HR / HRV / rhythm** | 4× – 8× | 4× – 8× |
| **Wellness / ambulatory monitoring** | 8× – 16× | 8× – 16× |
| **Event logging / screening / triage** | 16× – 32× | 16× – 32× |
| **Ultra‑low bandwidth telemetry** | 32× | 32× – 64× |

All PPG (2x-32x) and ECG (2x-64x) RVQ operating points ship as published bundles,
with DSP/hybrid comparison lanes registered for local reproduction. See the
[PPG Model Zoo](models/ppg.md) and [ECG Model Zoo](models/ecg.md) for measured
metrics and noise/artifact tradeoffs at each ratio.

---

## Getting started

1. **Try** — load a published HuggingFace bundle and round-trip a synthetic frame ([Getting Started](getting-started.md))
2. **Compare** — inspect RVQ, SPIHT, and hybrid tradeoffs on the model pages
3. **Evaluate** — run the codec on your own waveform and score the observables you care about
4. **Deploy** — validate the `deploy/` package before integrating encoder/codebook artifacts into firmware

See the [Deployment Guide](deployment.md) and [CLI reference](cli.md) for the full workflow.
