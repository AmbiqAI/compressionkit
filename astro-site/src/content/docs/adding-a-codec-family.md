---
title: "Adding a Codec Family"
description: "Connect a new codec to runtime loading, export, validation and experiment tooling."
---


This page is the concrete, operational companion to
[Experiment Architecture](/compressionkit/experiment-architecture/): it maps the "concept →
helper → feature → golden → release → docs" promotion path onto the actual
files and commands in this repo, and it documents the one place that must stay
in sync whenever a new codec **family** (today: `rvq`, `spiht`, `hybrid`) is
added.

<span id="two-different-meanings-of-family-dont-conflate-them"></span>

## Codec family and experiment structure

This repo uses "family" for two unrelated concepts. Keep them distinct:

- **Codec family** (`compressionkit.runtime.base.CodecFamily`, values
  `"rvq"` / `"spiht"` / `"hybrid"`) — *how* a codec is implemented and how its
  deploy manifest, runtime loader, and HuggingFace bundle are shaped. This is
  what [`CodecFamilySpec`](#the-codecfamilyspec-registry) below is about.
- **Golden structure** (`GoldenExperiment.structure`, values `"codec"` /
  `"two_stage"`) — whether a golden release is a single artifact or a codec
  paired with an entropy prior. Unrelated to codec family; a `two_stage` entry
  still has an underlying `method` (today always `"rvq"`).


## The `CodecFamilySpec` registry

Every consumer that needs to know "given family X, which runtime class loads
it / which files are required / which model-card generator applies / what's
the default license" reads that from one place:
[`compressionkit.export.family_registry.FAMILY_REGISTRY`](https://github.com/AmbiqAI/compressionkit/blob/main/compressionkit/export/family_registry.py).

Before this registry existed, that same knowledge was re-derived independently
in the runtime loader, the deploy validator, the model-card generator, and the
HuggingFace publisher — each with its own copy of the known-families list.
That duplication is exactly how a real bug shipped: the hybrid lane's
`family` field went stale in the exporter without the loader, validator, or
publisher ever noticing, because nothing forced them to agree. Adding a family
now means adding **one** `CodecFamilySpec` entry, not editing five modules and
hoping none are missed.

## Promotion path

| Stage | What happens | Where it lives |
|---|---|---|
| **Concept** | Prototype the idea with no contract yet. | `experiments/` (gitignored scratch) or a notebook. |
| **Initial helpers** | Extract genuinely reusable pieces — a layer, a loss, a DSP transform, a preprocessing block. | `compressionkit/{layers,losses,pipeline,dsp}/` |
| **Feature iteration** | Compare against existing goldens using the *same* eval harness (scorecard/metrics), not a bespoke one-off script. | `experiments/configs/`, `experiments/scripts/` |
| **Hook → golden** | Register a stable `experiment_id` + config/operating point. If it's a genuinely new codec *family* (not just a new operating point of an existing one), add a `CodecFamilySpec` entry too. | `compressionkit/experiments/registry.py`, `compressionkit/export/family_registry.py` |
| **Produce results** | Build the deploy package through the registered entry point, never by hand-assembling manifest JSON. | `compressionkit golden run <id>` (never invoke a golden script directly if you can help it — see below) |
| **Tie into release** | Publish via the same family spec so staging/model-card/license logic can't drift from the loader. | `compressionkit golden run <id> --publish`, `scripts/publish_to_huggingface.py` |
| **Docs** | One row in the golden matrix / registry table. | [V1 Release Contract](/compressionkit/release-contract/), `scripts/render_experiment_docs.py` output |

### Adding a new `CodecFamilySpec` entry

A new family needs, at minimum:

1. A runtime class implementing the `Codec` protocol (`.compress`/`.decompress`/
   `.name`/`.modality`/`.sample_rate`/`.frame_size`/`.target_cr`) under
   `compressionkit/runtime/`.
2. An exporter that writes `deploy_manifest.json` with the new `family` value,
   plus whatever artifacts the runtime needs to reload.
3. A `CodecFamilySpec` entry in `compressionkit/export/family_registry.py`
   declaring: the loader, required/release-extra artifact lists, the HF
   file-rename map (add one to `compressionkit/export/artifact_contract.py`
   if the family's files don't already have one), whether it ships a
   `c_sources/` C reference, whether it carries trained weights, a model-card
   generator, and a default license.
4. A `GoldenMethod` entry in `compressionkit/experiments/registry.py` if it
   should appear in the golden matrix, following the existing `_ppg_spiht`/
   `_ppg_hybrid`-style builder functions.

That's it — the runtime loader, deploy validator, and HuggingFace publisher
all read the same spec and need no further changes.

## Validate before you trust a local build

`results/` is gitignored, so a local deploy package can silently go stale
relative to the current code (e.g. a manifest-schema change lands but nobody
re-runs the golden that produced an old package). Two guards exist for this:

- Every golden-producing script (`scripts/run_hybrid_golden.py`,
  `scripts/run_spiht_golden_{ppg,ecg}.py`, and the RVQ training recipe via
  `compressionkit/recipes/base_rvq.py`) self-validates its own output with
  `validate_deploy_package(deploy_dir, strict_release=True)` as its last step
  and logs loudly (without aborting a completed training run) if it fails.
- `compressionkit golden validate-all --strict-release` sweeps every
  registered golden's existing local deploy package in one pass. Run this
  before cutting a release, or after any change to manifest schema, family
  semantics, or `export/validate.py`'s requirements.
