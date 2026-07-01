---
icon: lucide/badge-check
---

# V1 Release Contract

This page defines what it means for a compressionKIT artifact to be release-grade for v1.

For the execution plan that turns this contract into epics, milestones, and the target golden matrix, see [V1 Roadmap](v1-roadmap.md).

The core rule is simple: a v1 release is anchored on a frozen deploy package, not only on a training config. Configs remain the recipe for reproducing a run, but the deploy package is the contract used for runtime hydration, validation, HuggingFace publication, and website rendering.

Just as important: release-grade packaging must not force experiments into a single framework-shaped entry point. Experiments should be able to adopt reusable blocks incrementally and only opt into the release contract when they want a packageable golden artifact.

For the broader experiment design rules behind this contract, see [Experiment Architecture](experiment-architecture.md).

## Composition Principle

compressionKIT should prefer composition over orchestration.

That means:

- experiments pull in reusable blocks when they need them
- helpers reduce boilerplate but do not dictate one required control flow
- package validation targets artifact outputs, not how the experiment was written
- release tooling should accept any experiment that produces the contract, regardless of whether it used a shared base class or custom script

This is an explicit design goal for v1. The repo should provide convenient blocks for:

- dataset preparation
- preprocessing and augmentation
- model construction
- evaluation and scorecard generation
- deploy packaging
- deploy validation

But those blocks should stay decoupled enough that a new experiment can start small, import only what it needs, and gradually adopt more of the contract without first wiring itself into a large nested configuration or a mandatory base framework.

## Goals

The v1 release contract must support:

- PPG and ECG modalities in the first v1 cut.
- DSP and AI methods through one common release boundary, with both families represented in the canonical ECG and PPG golden matrix.
- Reproducible golden runs with frozen scorecards.
- Portable deployment artifacts for Python, embedded C, and browser demos.
- Runtime hydration from the packaged artifact instead of re-deriving settings from YAML.

IMU remains an intended extension of the contract, but it does not block the first v1 release cut.

## Golden Matrix

The v1 contract supports multiple method families, but the current public
release surface separates downloadable bundles from local comparison lanes:

| Modality | Published RVQ bundles | Local comparison lanes | Deferred |
|----------|------------------------|------------------------|----------|
| PPG | 2x, 4x, 8x, 16x, 32x | SPIHT and hybrid comparisons at matching operating points | entropy-prior bundles, IMU |
| ECG | 2x, 4x, 8x, 16x, 32x, 64x | SPIHT and hybrid comparisons at matching operating points | entropy-prior bundles, IMU |

DSP and AI should still be treated symmetrically at the registry and package
boundary when a lane is promoted to release grade. Hybrid remains supported, but
it should not block v1 unless a specific hybrid operating point is intentionally
promoted to release-grade.

Each golden experiment must have:

- one canonical `experiment_id`
- one canonical `run_name`
- one frozen deploy package under `results/<run_name>/deploy/`
- one frozen scorecard
- one HuggingFace destination repo when the package is published
- one rendered website page when the package is promoted into the public release surface

## Release Artifact Model

Every v1 golden package must contain a top-level manifest plus a stable codec specification.

### Required common files

| File | Purpose |
|------|---------|
| `deploy_manifest.json` | Top-level package manifest and artifact index |
| `codec_spec.json` | Canonical runtime hydration contract |
| `scorecard.json` | Frozen evaluation summary used for release checks and docs |
| `reference_vectors.npz` | Known-good encode/decode vectors for cross-runtime validation |
| `sample_stimulus.npz` | License-safe demo and smoke-test inputs |
| `model_card.json` | Release metadata and provenance |
| `checksums.json` | File integrity for CI, publication, and downstream consumers |
| `README.md` | Human-readable usage and integration notes |

AI packages may expose their demo frames as `sample_data.npz` when the file carries model inputs, targets, and reconstructions rather than synthetic stimulus alone. In strict release validation this is treated as the AI-family equivalent of `sample_stimulus.npz`.

### AI-specific files

| File | Purpose |
|------|---------|
| `encoder.tflite` | Edge encoder |
| `encoder.h` | Embedded encoder header |
| `decoder.keras` or `decoder_float32.tflite` | Host reference decode |
| `decoder.tflite` and `decoder.h` | Optional on-device decode |
| `codebook.npz` | Python codebook tables |
| `codebook.h` | Embedded codebook tables |

### DSP-specific files

| File | Purpose |
|------|---------|
| `dsp_config.json` | Language-neutral DSP operating point |
| `spiht_app_config.h` or equivalent | Embedded operating-point header |
| `c_reference/` or equivalent pinned bundle | Reference implementation for non-Python users |
| `wasm/` or equivalent optional demo bundle | Browser-ready package for demos |

## Manifest Responsibilities

`deploy_manifest.json` should remain the top-level package index. It should answer:

- what family this package belongs to
- which modality it targets
- which runtime contract version it follows
- where each artifact lives
- which file is the canonical hydration spec
- which scorecard is the frozen release summary

A v1 manifest should minimally include these top-level fields:

```json
{
  "manifest_version": 1,
  "package_version": "1.0.0",
  "family": "rvq",
  "method": "ai",
  "modality": "ppg",
  "experiment_id": "ppg-rvq-4x",
  "run_name": "ppg_rvq_64hz_04x_golden",
  "compression_ratio": 4,
  "spec": "codec_spec.json",
  "scorecard": "scorecard.json",
  "artifacts": {},
  "checksums": "checksums.json"
}
```

The exact shape can evolve, but the package must always make it obvious which file is authoritative for runtime hydration.

## Canonical Hydration Spec

`codec_spec.json` is the single most important addition for v1. It should be runtime-oriented rather than training-oriented.

This file should freeze:

- `family`: `rvq`, `spiht`, or `hybrid`
- `modality`
- `sample_rate`
- `frame_size`
- `compression_ratio`
- input tensor or frame contract
- preprocessing contract
- packet and bitstream contract
- output reconstruction contract
- artifact names needed by that runtime

### DSP fields

For DSP packages, the spec must include enough information to hydrate the codec without reading Python code or reconstructing a YAML config. At minimum:

- wavelet name
- decomposition levels
- frame size
- sample rate
- target compression ratio
- max bits or bytes per frame
- entropy coder enabled or disabled
- normalization or scaling rules
- stitch or overlap rules if they affect reconstruction behavior
- payload format and byte ordering

### AI fields

For AI packages, the spec must include:

- encoder input shape and dtype
- latent layout
- token layout
- codebook structure
- decoder availability and target
- quantization contract
- any preprocessing required for bit-exact reproduction

## Validation Contract

Every golden package must be self-validating. The validation contract should not depend on re-running training.

Required checks:

1. Manifest loads and points to all required files.
2. Runtime hydrates from `codec_spec.json` without reading the source YAML.
3. Reference vectors round-trip in Python.
4. Embedded or C reference implementation matches the package vectors.
5. Published HuggingFace artifact matches the local package checksums.
6. Website metrics are rendered from the packaged scorecard, not copied by hand.

The packaged validator is available as:

```bash
compressionkit golden validate-deploy results/<run_name>/deploy --strict-release
```

`compressionkit golden run` and `compressionkit golden run-all` validate the deploy package before optional HuggingFace publication by default. Use `--skip-validation` only for local iteration on incomplete packages; use `--strict-release-validation` for release candidates.

For DSP packages, the package must be usable without Python. That means the release should include either vendored reference sources or a pinned, checksum-verified reference module release.

## Publication Contract

For each golden release candidate:

1. Run the registered golden experiment.
2. Freeze the deploy package.
3. Freeze the scorecard.
4. Generate the model card.
5. Validate package hydration and reference vectors.
6. Publish the package to HuggingFace.
7. Render docs from the packaged manifest and scorecard.

The website should consume packaged release artifacts as inputs. It should not maintain separate hand-authored tables as the long-term source of truth.

## Repository Implications

The existing registry and runner already provide most of the release scaffolding:

- `compressionkit.experiments.registry` defines the canonical golden matrix.
- `compressionkit.experiments.runner` already dispatches by method family.
- `compressionkit.export.deploy` already emits AI deployment artifacts.
- `compressionkit.export.spiht_deploy` already emits DSP deployment artifacts.

The main repo work needed for v1 is to unify the top-level package contract across AI and DSP methods and to make `codec_spec.json` the authoritative hydration input.

That does not imply a single required experiment runner. Shared runners and base trainers are convenience layers for reducing repetition in common golden flows; they are not part of the contract itself.

## Milestones

### Milestone 1: Unified package contract

- Add `codec_spec.json` and `checksums.json` to both AI and DSP deploy packages.
- Normalize `deploy_manifest.json` so both paths share the same top-level fields.
- Rename or alias method-specific config files so every package exposes one canonical hydration file.

### Milestone 2: Cross-runtime validation

- Add Python validation against `reference_vectors.npz`.
- Add C validation for DSP packages.
- Add optional WASM or browser validation for demo bundles.

### Milestone 3: IMU goldens

- Add IMU entries to the golden registry.
- Define IMU operating points and scorecard requirements.
- Publish HF repos and rendered docs for IMU DSP and AI tiers.

### Milestone 4: Docs and release automation

- Generate website model pages from packaged scorecards.
- Publish HuggingFace from the package directory only.
- Add CI checks that reject releases with incomplete package contracts.

## Non-goals for v1

These are intentionally out of scope for the first v1 contract:

- a custom binary container format
- a LiteRT-style flatbuffer package wrapper
- forcing every package into a single-file bundle
- making hybrid a mandatory release family for all modalities
- forcing all experiments through one inheritance-heavy or nested-config-heavy entry point

A versioned JSON manifest plus strict checksums and reference vectors is enough for v1.

## Decision Summary

The deploy package is the release artifact.

Configs are for reproduction.

`codec_spec.json` is for hydration.

`scorecard.json` is for release evaluation.

`reference_vectors.npz` is for conformance.

If a runtime can satisfy those contracts, it should be able to load and validate a golden package without depending on the original training environment.
