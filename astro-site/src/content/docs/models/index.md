---
title: "Model zoo"
description: "Find PPG and ECG codec configurations, measured results and reproduction steps."
---

Browse codec configurations by signal, inspect the recorded evaluation results, and follow the linked experiment to reproduce a candidate.

<span id="current-release-surface"></span>

## Choose a signal

| Signal | Input frames | RVQ configurations | Results |
| --- | --- | --- | --- |
| **PPG** | 320 samples at 64 Hz (5 seconds) | 2×, 4×, 8×, 16×, 32× | [PPG models](/compressionkit/models/ppg/) |
| **ECG** | 512 samples at 256 Hz (2 seconds) | 2×, 4×, 8×, 16×, 32×, 64× | [ECG models](/compressionkit/models/ecg/) |

The experiment registry also includes SPIHT, hybrid, and paired entropy-prior configurations. A registered configuration or Hugging Face target is not proof that a bundle has been published. Use each experiment's package link to inspect the available files, or build a local deploy package.

<span id="evidence-router"></span>

<span id="what-each-model-page-should-answer"></span>

<span id="standard-comparison-rule"></span>

## Compare candidates

- **Start with the summary:** [Customer evidence](/compressionkit/customer-evidence/) shows recorded quality and package sizes across compression ratios.
- **Inspect the signal:** the PPG and ECG pages contain reconstruction metrics, noise comparisons, and architecture details.
- **Read detailed measurements:** [PPG compression vs fidelity](/compressionkit/methods/cr_vs_fidelity_ppg/) and [ECG compression vs fidelity](/compressionkit/methods/cr_vs_fidelity_ecg/) separate aggregate results from noise buckets.
- **Reproduce a configuration:** [Experiments](/compressionkit/experiments/) links to source configs and dataset requirements.

Compare results only when their dataset, reference signal, frame selection, and metric definitions match. The model-page validation results and the noise-aware scorecard results come from different evaluation views.

<span id="deployment-artifacts"></span>

## Load or validate a package

The [Hugging Face guide](/compressionkit/huggingface/) shows how to load a remote bundle or a local directory. The [deployment guide](/compressionkit/deployment/) covers package validation and encoder/decoder integration.

A deploy package describes the codec and includes the files needed to run it. See the [artifact contract](/compressionkit/release-contract/) for manifests, checksums, reference vectors, and scorecard requirements.
