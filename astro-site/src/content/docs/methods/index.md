---
title: "Compression Methods"
description: "Compare neural RVQ, wavelet SPIHT and hybrid codec methods."
---


Compare methods using the same signal, reference, and metric definitions. The toolkit includes neural, DSP, and hybrid codec paths; their tradeoffs depend on the recording and compression ratio.

## Method comparison

| Method | Approach | Where to start |
| --- | --- | --- |
| **RVQ autoencoder** | A learned encoder and decoder with residual vector quantization | [Architecture and configuration](/compressionkit/methods/rvq/) |
| **Wavelet + SPIHT** | A wavelet transform with progressive coefficient coding | [SPIHT guide](/compressionkit/methods/spiht/) |
| **Hybrid + SPIHT** | A learned front end followed by a DSP codec | [Hybrid guide](/compressionkit/methods/hybrid/) |
| **Decimation** | A simple comparison baseline that reduces sample rate | Compare against the same signal and reconstruction requirements |

The experiment registry records package targets. Inspect the linked bundle to confirm remote availability, or reproduce a local package.

## How to choose what to inspect first

| Question | Best starting page |
|----------|--------------------|
| What can I download today? | [Model Zoo](/compressionkit/models/) |
| How do PPG methods behave under empirical noise or motion? | [PPG Models](/compressionkit/models/ppg/) |
| How do ECG methods behave under SNR and contact-artifact sweeps? | [ECG Models](/compressionkit/models/ecg/) |
| What are the CR-vs-fidelity numbers? | [PPG CR vs Fidelity](/compressionkit/methods/cr_vs_fidelity_ppg/), [ECG CR vs Fidelity](/compressionkit/methods/cr_vs_fidelity_ecg/) |
| What do the metrics mean? | [Validation Scorecard](/compressionkit/validation-scorecard/) |

## A practical selection workflow

1. Choose PPG or ECG and confirm the package sample rate and frame size match your input preparation.
2. Run a published package before attempting training. Use SPIHT as a DSP comparison, RVQ for a learned codec, and hybrid when evaluating learned preprocessing plus compression.
3. Compare operating points on the same recordings and metric definitions. Select a quality requirement before choosing the largest compression ratio.
4. Measure the complete integration, including framing, side information, transport overhead, execution time and memory.

The model tables are evidence for their recorded evaluation conditions. They do not select a codec for a different sensor, population or device automatically.

## Design principles

All release-facing methods in compressionKIT follow these principles:

1. **Plan for embedded integration**: measure the chosen implementation on the target; a Python codec does not by itself provide a portable C runtime.
2. **Quantization-friendly** — Only operators that work with INT8/INT16x8 quantization
3. **Configurable via YAML** — Major parameters exposed through configuration files
4. **Evaluated on signal utility** — Not just MSE, but modality-specific metrics and noise/artifact behavior

## CR vs. Fidelity Decision Artefacts

Summary tables of the v1 RVQ goldens with codec compression ratios, optional
codec+prior **effective** compression ratios where available, and noise-aware
fidelity metrics (PRD, PRDN-noise, HR MAE, QRS- / pulse-band PSD error,
coherence, stitching seam ratio):

- [ECG CR vs. Fidelity](/compressionkit/methods/cr_vs_fidelity_ecg/)
- [PPG CR vs. Fidelity](/compressionkit/methods/cr_vs_fidelity_ppg/)

Regenerate from existing scorecards with `python scripts/build_cr_vs_fidelity.py --modality {ecg,ppg}`.
