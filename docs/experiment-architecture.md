---
icon: lucide/blocks
---

# Experiment Architecture

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

```python
dataset = build_dataset(...)
model = build_model(...)
history = model.fit(dataset.train, validation_data=dataset.val)
results = evaluate_model(model, dataset.val)
export_for_deployment(...)
validate_deploy_package("results/my_experiment/deploy")
```

The same experiment can later opt into golden release mechanics by adding a registry entry, frozen config, scorecard, publication target, and docs.

## Codebase Audit

Current alignment:

- `compressionkit.export.deploy.export_for_deployment()` is a reusable block.
- `compressionkit.export.spiht_deploy.export_spiht_deploy()` is a reusable DSP packaging block.
- `compressionkit.export.validate.validate_deploy_package()` validates artifacts without caring how they were produced.
- `compressionkit.runtime.load_codec()` hydrates from deploy packages, not training configs.
- `compressionkit.experiments.registry` and `compressionkit.experiments.runner` are suitable for golden release automation.

Watch items:

- `BaseRVQTrainer` should remain optional convenience glue for shipped RVQ recipes. Pull reusable behavior out into smaller functions when it starts accumulating release, export, or evaluation policy.
- `compressionkit.experiments.runner` should stay focused on registered goldens. It should not become the only supported experiment path.
- Export metadata should move toward small typed helpers instead of ad hoc nested dictionaries where the shape becomes shared behavior.
- Scorecard and model-card generation should be callable as standalone blocks before being wired into golden automation.

## Refactoring Direction

When adding new capability, prefer this order:

1. Add a focused block with a small API.
2. Use the block from one ready-made experiment.
3. Add golden automation only after the block proves stable.
4. Validate artifacts at the package boundary.

If a base class grows new responsibilities, consider extracting those responsibilities into standalone helpers instead of adding another abstract hook.

Good candidates:

- `build_release_metadata(...)`
- `write_reference_vectors(...)`
- `write_checksums(...)`
- `write_scorecard_artifact(...)`
- `validate_deploy_package(...)`
- `render_model_card(...)`

## HeliaEdge Candidates

HeliaEdge is the Keras 3 extension package for reusable training and deployment blocks suited for Edge AI. Some compressionKIT capabilities should move there once they are generic enough and not specific to physiological compression experiments.

Likely candidates:

- RVQ architecture blocks that are not tied to PPG, ECG, or IMU assumptions
- vector-quantization layers, codebook utilities, and rate-distortion training helpers
- Keras metrics and loss functions that apply broadly to compact edge models
- lightweight callbacks for schedules, codebook health, and training diagnostics
- LiteRT export wrappers and C-header generation helpers
- generic deployment manifest/checksum helpers
- reference-vector conformance utilities
- reusable quantized-runtime helpers

Keep in compressionKIT for now:

- PPG, ECG, and IMU dataset contracts
- physiology-specific scorecards
- compressionKIT golden registry and HuggingFace naming
- signal-specific preprocessing and evaluation policy
- SPIHT/RVQ package schemas while they are still evolving toward v1

The decision rule is: if the block is broadly useful for Keras 3 edge-model training or deployment independent of physiological compression, consider moving it into HeliaEdge. If it encodes modality, dataset, scorecard, or release policy for compressionKIT, keep it here.

## Review Checklist

Before adding a new experiment-facing abstraction, ask:

- Can this be a standalone function or small object?
- Can a custom script use it without joining a registry?
- Does it require fewer hooks, not more?
- Is the contract expressed in artifacts rather than control flow?
- Would this be useful in HeliaEdge if project-specific assumptions were removed?

If the answer is no, keep the abstraction local to the ready-made experiment until the repeated pattern is clearer.
