---
title: "RVQ v1.1 corrected exports"
description: "Pinned releases, export parity and packet compatibility for the corrected single-stage RVQ bundles."
---

The 2026-10-06 v1.1 release corrects the deployment exports of eleven existing
RVQ checkpoints. It removes EMA embedding-sum accumulators that were incorrectly
exported as extra codebooks. No models were retrained.

The [release inventory](https://github.com/AmbiqAI/compressionkit/blob/9a3b07806a76de718a0dfa7105a23097853e2fac/releases/rvq-v1.1.json)
records every published repository, immutable revision, checksum-manifest hash,
checkpoint provenance, precision report and complete packet size.

## Published bundles

| Modality | Nominal ratios | Repository pattern |
|---|---|---|
| PPG | 2x, 4x, 8x, 16x, 32x | `Ambiq/compressionkit-ppg-{cr}x-v1.1` |
| ECG | 2x, 4x, 8x, 16x, 32x, 64x | `Ambiq/compressionkit-ecg-{cr}x-v1.1` |

Download an exact revision from the inventory for reproducible evaluation:

```python
from huggingface_hub import snapshot_download
from compressionkit.runtime import RVQCodec

path = snapshot_download(
    "Ambiq/compressionkit-ppg-8x-v1.1",
    revision="24caa26b155a8173222620f4e5deb25a07d49879",
)
codec = RVQCodec(path)
```

These bundles preserve canonical deploy filenames alongside older Hugging Face
aliases. Each includes independent training reference vectors, synthetic sample
inputs, checksums, model cards, provenance and all four encoder precisions.

## Validation

All eleven local packages, staged upload packages and revision-pinned downloads
passed strict release validation with ten reference frames each. Complete packet
round trips passed for ten synthetic frames per bundle and encoder precision:
440 checks across FP32, INT8, FP16 and INT16x8. Downloaded files matched their
published checksum manifests. The original local artifacts and v1.0 repository
heads remained unchanged.

Measured packet sizes include the four-byte float16 mean/scale header and
fixed-width RVQ tokens. ECG 8x uses 132 bytes per frame; PPG 8x uses 84 bytes.
The nominal ratio is an operating-point label; consult the inventory for the
actual ratio against 16-bit input, including the header.

INT8 calibration and precision evaluation use content-disjoint real frames with
the saved training preprocessing. Their reports measure differences from the
FP32 deployed reconstruction. They do not measure physiological accuracy or
MCU kernel equivalence. ECG 64x INT8 has 13.64% P90 PRD and PPG 32x INT8 has
11.31%, above the 10% precision recommendation; their cards flag the higher
error. FP32 remains the default encoder.

## Compatibility and evidence

Bind stored packets to their exact repository and revision. Original v1.0 packets
require their original codebooks and decoder; the corrected level counts change
packet interpretation. The old bundles remain available.

Physiological scorecards are historical results, explicitly labeled in each model
card and `release_provenance.json`. This release establishes export parity and
refreshes precision comparisons; it does not establish new physiological quality,
unseen-device generalization or encoder energy measurements. Saved configuration
sources and checkpoint hashes are recorded; complete pretraining ancestry was not
independently audited in this repair.

The optional entropy priors remain on their historical v1.0 track and were not
included or requalified. Golden publication rejects a prior when its release
track differs from its parent's until that prior is requalified.
