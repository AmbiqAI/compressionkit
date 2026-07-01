---
icon: lucide/layers
---

# Model Zoo

The model zoo is the release-facing view of compressionKIT artifacts. It shows
which codec packages are currently supported, what signal surfaces they cover,
and where to find the measured scorecards. It is not meant to imply that these
are the only codec families compressionKIT can support.

## Current release surface

| Signal | Current release status | Published bundles | Deeper results |
|--------|------------------------|-------------------|----------------|
| **PPG** | v1 neural codec goldens at 64 Hz | 2x, 4x, 8x, 16x, 32x | [PPG models](ppg.md), [PPG CR vs fidelity](../methods/cr_vs_fidelity_ppg.md) |
| **ECG** | v1 neural codec goldens at 256 Hz | 2x, 4x, 8x, 16x, 32x, 64x | [ECG models](ecg.md), [ECG CR vs fidelity](../methods/cr_vs_fidelity_ecg.md) |
| **IMU** | Extension target | Not published in v1 | Define preservation metrics first: event timing, activity-band energy, orientation/motion features |

DSP, hybrid, and entropy-prior lanes are registered and locally reproducible for
comparison and release packaging work. Their HuggingFace publication status is
tracked in the [golden experiments](../experiments/index.md) registry.

## Evidence router

Start from the customer question, then move to the page that carries the right
context. The model zoo index intentionally avoids embedding every metric plot;
plots belong beside their scorecard tables, sample counts, noise buckets, and
reproduction commands.

| Question | First page to open | What to look for |
|----------|--------------------|------------------|
| Which compression ratios are published? | [Customer evidence](../customer-evidence.md) | CR ladder, frame duration, edge payload, recommended operating bands |
| How does PPG behave across CR? | [PPG models](ppg.md) | HR/HRV preservation, pulse-band error, waveform fidelity, robustness links |
| How does ECG behave across CR? | [ECG models](ecg.md) | R-peak timing, QRS-band error, morphology behavior, robustness links |
| What happens in clean vs noisy regimes? | [PPG CR vs fidelity](../methods/cr_vs_fidelity_ppg.md), [ECG CR vs fidelity](../methods/cr_vs_fidelity_ecg.md) | truth PRD, faithful PRD, PRDN-noise, clean/median/noisy tertiles |
| Is the package deployable? | [Deployment](../deployment.md), [V1 release contract](../release-contract.md) | manifests, checksums, TFLite models, C headers, reference vectors |

## What each model page should answer

| Question | Where it appears |
|----------|------------------|
| What package can I download? | HuggingFace links on the PPG/ECG model pages |
| What compression ratios are supported? | Per-modality result tables |
| What happens to physiological observables? | HR/HRV, QRS-band, coherence, morphology, and bucketed scorecard rows |
| How does performance change under noise or artifact regimes? | CR-vs-fidelity and validation-scorecard pages |
| Can I reproduce or validate the package? | Golden experiment pages and deploy validation commands |

## Standard comparison rule

New candidates should be compared against currently supported goldens through the
same generated surfaces:

1. same dataset split or documented fixture
2. same compression-ratio accounting
3. same waveform and physiology metrics
4. same clean/median/noisy or SNR/artifact buckets
5. same deploy-package validation contract

That structure allows newer releases to be compared against older still-supported
versions without relying on hand-written claims that drift over time.

## Deployment artifacts

Every release-grade package is expected to carry enough information to be loaded,
validated, and compared without the original training environment:

| Artifact | Purpose |
|----------|---------|
| `deploy_manifest.json` / `codec_spec.json` | Runtime hydration and package identity |
| LiteRT/TFLite models and C headers | Edge and host deployment |
| Codebooks or DSP operating-point files | Codec state needed for encode/decode |
| `reference_vectors.npz` | Known-good conformance vectors |
| `scorecard.json` | Frozen release metric summary |
| `checksums.json` | Integrity validation |

See [Deployment](../deployment.md) and [V1 Release Contract](../release-contract.md)
for the full artifact contract.
