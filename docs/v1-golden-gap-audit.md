---
icon: lucide/clipboard-list
---

# V1 Golden Gap Audit

This page captures the current gap between the **first-cut v1 target** and the
golden ECG and PPG artifacts currently present in `results/`.

The scope here is intentionally narrow:

- official ECG and PPG AI deploy artifacts currently present in `results/`
- reproducibility from config and registered runner
- release-grade scorecards
- deploy package completeness
- HuggingFace-ready publication artifacts

The registry now also carries the matching ECG and PPG DSP SPIHT operating-point
matrix. This page remains focused on the artifact-completeness gap for the
already-materialized AI runs in `results/`; DSP matrix coverage is now a registry
and runner concern first, with package backfill tracked separately.

For the target plan, see [V1 Roadmap](v1-roadmap.md).
For the artifact requirements, see [V1 Release Contract](release-contract.md).

## Audit Scope

Audited goldens:

- `ppg-rvq-2x`
- `ppg-rvq-4x`
- `ppg-rvq-8x`
- `ppg-rvq-16x`
- `ppg-rvq-32x`
- `ecg-rvq-2x`
- `ecg-rvq-4x`
- `ecg-rvq-8x`
- `ecg-rvq-16x`
- `ecg-rvq-32x`
- `ecg-rvq-64x`

Audit criteria:

- root run artifacts: `config.json`, `summary.json`, `quality_scorecard.json`
- deploy package existence: `results/<run_name>/deploy/`
- first-cut v1 release extras in deploy package:
  `deploy_manifest.json`, `codec_spec.json`, `checksums.json`,
  `reference_vectors.npz`, `model_card.json`, `README.md`
- core AI deploy artifacts:
  `encoder.tflite`, `encoder.h`, `codebook.npz`, `codebook.h`
- one packaged sample bundle:
  `sample_data.npz` or `sample_stimulus.npz`

## High-Level Findings

1. The documented ECG and PPG RVQ goldens mostly exist as run directories and already have deploy subdirectories.
2. The core AI inference artifacts are present across the audited goldens: `encoder.tflite`, `encoder.h`, `codebook.npz`, `codebook.h`, and a sample bundle are already there.
3. The main v1 blocker is **package completeness**, not total absence of deployment outputs.
4. The current deploy packages are consistently missing the first-cut v1 contract files `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `model_card.json`, and `README.md`.
5. Two documented PPG goldens still lack root `quality_scorecard.json`: `ppg_rvq_64hz_16x_golden` and `ppg_rvq_64hz_32x_golden`.
6. The codebase already contains infrastructure for the missing contract pieces, so this is partly a packaging and backfill problem rather than a greenfield implementation problem.

## Current State Table

| Experiment | Run dir exists | Root scorecard | Deploy dir | Core AI artifacts | First-cut v1 release extras |
|------------|----------------|----------------|------------|-------------------|-----------------------------|
| `ppg-rvq-2x` | Yes | Yes | Yes | Present | Missing `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `model_card.json`, `README.md` |
| `ppg-rvq-4x` | Yes | Yes | Yes | Present | Missing `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `model_card.json`, `README.md` |
| `ppg-rvq-8x` | Yes | Yes | Yes | Present | Missing `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `model_card.json`, `README.md` |
| `ppg-rvq-16x` | Yes | No | Yes | Present | Missing `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `model_card.json`, `README.md` |
| `ppg-rvq-32x` | Yes | No | Yes | Present | Missing `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `model_card.json`, `README.md` |
| `ecg-rvq-2x` | Yes | Yes | Yes | Present | Missing `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `model_card.json`, `README.md` |
| `ecg-rvq-4x` | Yes | Yes | Yes | Present | Missing `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `model_card.json`, `README.md` |
| `ecg-rvq-8x` | Yes | Yes | Yes | Present | Missing `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `model_card.json`, `README.md` |
| `ecg-rvq-16x` | Yes | Yes | Yes | Present | Missing `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `model_card.json`, `README.md` |
| `ecg-rvq-32x` | Yes | Yes | Yes | Present | Missing `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `model_card.json`, `README.md` |
| `ecg-rvq-64x` | Yes | Yes | Yes | Present | Missing `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `model_card.json`, `README.md` |

## Deploy Package Observations

The current deploy directories are not empty or provisional. They already look like partially modernized release bundles.

Common contents across most audited goldens:

- `deploy_manifest.json`
- `encoder.tflite`
- `encoder.h`
- `codebook.npz`
- `codebook.h`
- `decoder.keras`
- `sample_data.npz`

Some lower-CR runs also already include extra decode artifacts such as:

- `decoder.tflite`
- `decoder.h`
- `decoder_float32.tflite`
- `_decoder_float32.h`

That means the first-cut v1 work is mostly about **standardizing** the release bundle and **backfilling** missing metadata and validation files, not inventing the deploy package from scratch.

## Infrastructure Already Present In Code

The repo already has strong support for the missing contract pieces:

- [compressionkit/export/deploy.py](../compressionkit/export/deploy.py) writes `codec_spec.json`, `checksums.json`, and `reference_vectors.npz`.
- [compressionkit/export/validate.py](../compressionkit/export/validate.py) already validates `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, and strict release extras.
- [compressionkit/export/model_card.py](../compressionkit/export/model_card.py) already supports release-facing model-card generation.
- [compressionkit/experiments/registry.py](../compressionkit/experiments/registry.py) already defines the canonical ECG and PPG golden registry and HuggingFace targets.
- [compressionkit/experiments/runner.py](../compressionkit/experiments/runner.py) and [compressionkit/experiments/cli.py](../compressionkit/experiments/cli.py) already route golden runs through validation.

The gap is therefore not “missing v1 concepts.” The gap is that the **official current golden artifacts in `results/` are not yet uniformly regenerated through the full first-cut v1 contract**.

## Priority Gaps

### Gap 1: Backfill release extras into all ECG and PPG goldens

Why it matters:

- this is the main blocker to calling the current goldens release-grade under the first-cut contract

Required work:

- ensure every promoted golden emits `codec_spec.json`
- ensure every promoted golden emits `checksums.json`
- ensure every promoted golden emits `reference_vectors.npz`
- ensure every promoted golden emits `model_card.json`
- ensure every promoted golden emits deploy `README.md`

### Gap 2: Finish missing scorecards for documented PPG goldens

Why it matters:

- a release-grade golden without a frozen scorecard does not meet the current v1 bar

Current missing scorecards:

- `results/ppg_rvq_64hz_16x_golden/quality_scorecard.json`
- `results/ppg_rvq_64hz_32x_golden/quality_scorecard.json`

### Gap 3: Standardize the golden packaging path

Why it matters:

- the deploy contents suggest multiple historical export paths or partial repackaging
- users need one obvious golden release path, not per-run special cases

Required work:

- make `compressionkit golden run <id>` the canonical path for generating the full first-cut v1 release bundle
- keep that path close to normal user flows by reusing the same blocks and adding release hooks at the boundary
- ensure manual scripts such as [scripts/package_golden_release.py](/workspaces/compressionkit/scripts/package_golden_release.py#L1) either emit the full contract or are clearly marked as transitional

### Gap 4: Publishability and reproducibility need to stay coupled

Why it matters:

- the published result should be reproducible from the same config and runner the user sees

Required work:

- each golden docs page should include the exact reproduction command
- publish should run from a validated package, not from a side path
- release docs and HuggingFace assets should be derived from packaged artifacts, not hand-maintained tables

## Recommended Execution Order

1. Regenerate the documented ECG and PPG goldens through one canonical packaging path.
2. Backfill missing scorecards, especially `ppg-rvq-16x` and `ppg-rvq-32x`.
3. Make strict release validation part of the golden export and publish path by default.
4. Verify that every golden registry entry maps to a complete package, scorecard, and HF target.
5. Clean up the code path so the golden flow reads as “user experiment plus release hooks,” not as a separate orchestration silo.

## Immediate Definition Of Done

The first-cut v1 ECG and PPG golden set is ready when:

- every documented ECG and PPG RVQ golden has a complete root scorecard
- every documented ECG and PPG RVQ golden has a complete first-cut v1 deploy bundle
- every documented ECG and PPG SPIHT golden is reproducible from the registered golden runner and emits the same release-contract package shape
- every documented golden passes strict deploy validation
- every golden docs page tells the user how to reproduce the result
- every promoted golden can be exported and published through one obvious release path
