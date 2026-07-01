---
icon: lucide/layers
---

# Compression Methods

compressionKIT compares multiple codec families through the same release and
scorecard surfaces. The v1 HuggingFace bundles are RVQ neural codecs; SPIHT and
hybrid lanes are registered locally so clean-signal and noisy-wearable tradeoffs
can be measured in a standardized way.

## Method Comparison

| Method | Type | Current role | Deployment status |
|--------|------|--------------|-------------------|
| [**RVQ Autoencoder**](rvq.md) | Learned neural codec | Published v1 PPG/ECG bundles and primary runtime path | INT8 LiteRT/TFLite encoder + codebook artifacts |
| **Wavelet + SPIHT** | Classical DSP codec | Clean-signal and faithfulness baseline; registered golden comparison lane | Local deploy package support; not in v1 HuggingFace bundles |
| **Hybrid + SPIHT** | Learned/DSP hybrid | Wearable-noise and artifact comparison lane | Local deploy package support; not in v1 HuggingFace bundles |
| **Decimation** | Simple baseline | Sanity baseline for compression-ratio studies | Trivial local implementation |

## How to choose what to inspect first

| Question | Best starting page |
|----------|--------------------|
| What can I download today? | [Model Zoo](../models/index.md) |
| How do PPG methods behave under empirical noise or motion? | [PPG Models](../models/ppg.md) |
| How do ECG methods behave under SNR and contact-artifact sweeps? | [ECG Models](../models/ecg.md) |
| What are the CR-vs-fidelity numbers? | [PPG CR vs Fidelity](cr_vs_fidelity_ppg.md), [ECG CR vs Fidelity](cr_vs_fidelity_ecg.md) |
| What do the metrics mean? | [Validation Scorecard](../validation-scorecard.md) |

## Design principles

All release-facing methods in compressionKIT follow these principles:

1. **Portable to embedded C** — No dynamic allocation, fixed memory layouts
2. **Quantization-friendly** — Only operators that work with INT8/INT16x8 quantization
3. **Configurable via YAML** — Major parameters exposed through configuration files
4. **Evaluated on signal utility** — Not just MSE, but modality-specific metrics and noise/artifact behavior

## CR vs. Fidelity Decision Artefacts

Summary tables of the v1 RVQ goldens with codec compression ratios, optional
codec+prior **effective** compression ratios where available, and noise-aware
fidelity metrics (PRD, PRDN-noise, HR MAE, QRS- / pulse-band PSD error,
coherence, stitching seam ratio):

- [ECG CR vs. Fidelity](cr_vs_fidelity_ecg.md)
- [PPG CR vs. Fidelity](cr_vs_fidelity_ppg.md)

Regenerate from existing scorecards with `python scripts/build_cr_vs_fidelity.py --modality {ecg,ppg}`.
