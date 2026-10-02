---
title: "Artifact contract"
description: "Files, metadata and validation required for a compressionKIT deploy package."
---

A deploy package carries the codec files, runtime settings, and validation evidence needed to load and check a codec independently of its training run. This page describes the release contract; inspect the package manifest to see what an individual export contains.

Use the [deployment guide](/compressionkit/deployment/) for loading examples. Use [experiments](/compressionkit/experiments/) to reproduce a configuration.

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
| `encoder.keras` | Host reference encode |
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

`codec_spec.json` records runtime settings independently of the training configuration.

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
4. Where a separate embedded or C implementation is provided, verify it against the package vectors in that runtime.
5. Published HuggingFace artifact matches the local package checksums.
6. Confirm that reported metrics match the scorecard being released.

The packaged validator is available as:

```bash
compressionkit golden validate-deploy results/<run_name>/deploy --strict-release
```

`compressionkit golden run` and `compressionkit golden run-all` validate the deploy package before optional HuggingFace publication by default. Use `--skip-validation` only for local iteration on incomplete packages; use `--strict-release-validation` for release candidates.

Since `results/` is gitignored, a local deploy package can silently go stale relative to the current schema (e.g. a manifest field's semantics change but nobody re-runs the golden that produced an old package). Sweep every registered golden's existing local package in one pass with:

```bash
compressionkit golden validate-all --strict-release
```

Run this before cutting a release, and after any change to manifest schema, family semantics, or `export/validate.py`'s requirements.

Family-specific dispatch (which runtime class loads a package, which files are required, which model-card generator applies, the default license) is centralized in one place — `compressionkit.export.family_registry.FAMILY_REGISTRY` — rather than re-derived independently by the loader, validator, model-card generator, and HuggingFace publisher. See [Adding a Codec Family](/compressionkit/adding-a-codec-family/) for the full promotion path from concept to a published golden.

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
