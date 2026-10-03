---
title: "Experiment Architecture"
description: "Compose datasets, models, evaluation and export into a reproducible codec workflow."
---


compressionKIT should make common experiments easy to rerun without making new experiments hard to invent.

The guiding principle is: experiments pull in capabilities; capabilities should not pull experiments into one mandatory framework.

## Layers

### Blocks

Blocks are small, importable capabilities. They should be useful from a notebook, a custom script, a shipped recipe, or a golden runner.

Examples:

- dataset loaders and cache builders
- preprocessing and augmentation layers
- model builders
- losses, metrics, and callbacks
- scorecard builders
- deploy exporters
- deploy validators
- runtime loaders

Blocks should avoid hidden global state, mandatory registries, and deep nested dictionaries. When structured data is needed, prefer typed objects or small explicit arguments.

### Ready-made experiments

Ready-made experiments are readable end-to-end flows that users can rerun or copy.

They should:

- show a coherent default path
- keep the sequence of operations easy to follow
- use blocks directly where possible
- add helper classes only when they remove meaningful repetition
- be easy to fork into a new experiment

Ready-made experiments can be opinionated, but they should not be the only path to use the package.

### Golden releases

Golden releases are stricter than normal experiments. They need stable IDs, reproducible configs, scorecards, deploy packages, HuggingFace publication, and docs.

This extra structure is acceptable because goldens are release artifacts. The structure should remain at the release boundary: an experiment becomes golden by producing and validating the package contract, not by being rewritten around a mandatory base class.

## Contract Boundary

The v1 contract applies to artifacts:

- `deploy_manifest.json`
- `codec_spec.json`
- `checksums.json`
- `reference_vectors.npz`
- `scorecard.json` for release candidates
- model cards, README files, and published HuggingFace packages

The contract does not require:

- one training entry point
- one base class
- one registry for all experiments
- one deeply nested config shape
- one orchestration framework

This lets a custom experiment start small:

The following pseudocode shows the sequence; the builders depend on your experiment.

```python
dataset = build_dataset(...)
model = build_model(...)
history = model.fit(dataset.train, validation_data=dataset.val)
results = evaluate_model(model, dataset.val)
export_for_deployment(...)
validate_deploy_package("results/my_experiment/deploy")
```

The same experiment can later opt into golden release mechanics by adding a registry entry, frozen config, scorecard, publication target, and docs.

## Build an experiment

1. Start from a [registered experiment](/compressionkit/experiments/) close to your signal and codec family.
2. Copy its configuration and use a separate output directory.
3. Adapt dataset preparation, the model, and evaluation to your requirements.
4. Export a deploy package and run the [validation checks](/compressionkit/deployment/#recommended-validation).

See [adding a codec family](/compressionkit/adding-a-codec-family/) for runtime loading and exporter integration. The [artifact contract](/compressionkit/release-contract/) defines the package boundary.
