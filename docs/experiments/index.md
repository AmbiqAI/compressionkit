---
icon: lucide/flask-conical
---

# Golden Experiments

A **golden experiment** is a release-grade run that ships with compressionKIT for v1.
Each entry is fully declarative: a source-controlled config or fully declared DSP
operating point, a registered runner path, and a fixed dataset + HuggingFace repo target.
The lifecycle runner (`compressionkit golden run <id>`) reproduces it end-to-end from a
clean checkout.

Three families exist:

- **`codec` + `rvq`** — single-stage AI codec (encoder → RVQ → decoder).
- **`codec` + `spiht`** — single-stage DSP codec with a fully declared operating point in the registry.
- **`two_stage`** — paired entropy prior on top of a parent codec. Trains a small
  causal-transformer prior over the codec's token stream and adds `prior_int8.tflite`
  to a local deploy package. Prior-augmented bundles are reproducible today but are not
  part of the published v1 HuggingFace packages yet.

For first-cut v1, ECG and PPG both carry a dual-family golden matrix: DSP SPIHT and AI RVQ
at the chosen compression ratios. Use `compressionkit golden list --method rvq` or
`compressionkit golden list --method spiht` to slice the registry by family.

## Choose the lightest path that answers your question

compressionKIT uses progressive disclosure for release workflows: start with the
published artifact when you only need to try a codec, then move down toward full
reproduction only when you need stronger evidence or a modified experiment.

| Goal | Use this path | Dataset required? |
|------|---------------|-------------------|
| Try a published v1 codec | Load the HuggingFace bundle with `RVQCodec.from_pretrained(...)` | No |
| Check a local deploy package | `compressionkit golden validate-deploy <deploy-dir>` | No |
| Stage a package for review | `compressionkit golden run <id> --skip-train --publish --dry-run` | No, if `deploy/` already exists |
| Reproduce metrics and artifacts | `compressionkit golden run <id>` | Yes |
| Change the experiment | Copy a config or ready-made flow, then export and validate artifacts | Usually |
| Promote a result to v1 golden status | Add registry entry, frozen config, scorecard, docs, and publication target | Yes |

The registry is the release surface, not the experimentation surface. Custom work can
start as a script, notebook, or copied recipe that calls reusable blocks directly. It
only needs a golden registry entry when the output should become a supported release
artifact.

## v1 Registry

!!! info "Publication status"
    **Single-stage AI (RVQ) baselines are published** and downloadable from
    HuggingFace today — those links resolve. **DSP (SPIHT), hybrid, and entropy-prior
    variants are registered and reproducible** from their configs via
    `compressionkit golden run <id>`, but their HuggingFace repos are **not yet
    published**; the slugs shown are reserved targets.

### AI Baselines

| Experiment | Modality | Family | CR | Parent | Dataset | HuggingFace |
|------------|----------|--------|----|--------|---------|-------------|
| [`ppg-rvq-2x`](ppg-rvq-2x.md) | PPG | codec | 2× | — | `mesa` | [`Ambiq/compressionkit-ppg-2x`](https://huggingface.co/Ambiq/compressionkit-ppg-2x) |
| [`ppg-rvq-4x`](ppg-rvq-4x.md) | PPG | codec | 4× | — | `mesa` | [`Ambiq/compressionkit-ppg-4x`](https://huggingface.co/Ambiq/compressionkit-ppg-4x) |
| [`ppg-rvq-8x`](ppg-rvq-8x.md) | PPG | codec | 8× | — | `mesa` | [`Ambiq/compressionkit-ppg-8x`](https://huggingface.co/Ambiq/compressionkit-ppg-8x) |
| [`ppg-rvq-16x`](ppg-rvq-16x.md) | PPG | codec | 16× | — | `mesa` | [`Ambiq/compressionkit-ppg-16x`](https://huggingface.co/Ambiq/compressionkit-ppg-16x) |
| [`ppg-rvq-32x`](ppg-rvq-32x.md) | PPG | codec | 32× | — | `mesa` | [`Ambiq/compressionkit-ppg-32x`](https://huggingface.co/Ambiq/compressionkit-ppg-32x) |
| [`ppg-rvq-4x-prior`](ppg-rvq-4x-prior.md) | PPG | two_stage | 4× | ppg-rvq-4x | `mesa` | pending (`Ambiq/compressionkit-ppg-4x`) |
| [`ppg-rvq-8x-prior`](ppg-rvq-8x-prior.md) | PPG | two_stage | 8× | ppg-rvq-8x | `mesa` | pending (`Ambiq/compressionkit-ppg-8x`) |
| [`ecg-rvq-2x`](ecg-rvq-2x.md) | ECG | codec | 2× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-2x`](https://huggingface.co/Ambiq/compressionkit-ecg-2x) |
| [`ecg-rvq-4x`](ecg-rvq-4x.md) | ECG | codec | 4× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-4x`](https://huggingface.co/Ambiq/compressionkit-ecg-4x) |
| [`ecg-rvq-8x`](ecg-rvq-8x.md) | ECG | codec | 8× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-8x`](https://huggingface.co/Ambiq/compressionkit-ecg-8x) |
| [`ecg-rvq-16x`](ecg-rvq-16x.md) | ECG | codec | 16× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-16x`](https://huggingface.co/Ambiq/compressionkit-ecg-16x) |
| [`ecg-rvq-32x`](ecg-rvq-32x.md) | ECG | codec | 32× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-32x`](https://huggingface.co/Ambiq/compressionkit-ecg-32x) |
| [`ecg-rvq-64x`](ecg-rvq-64x.md) | ECG | codec | 64× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-64x`](https://huggingface.co/Ambiq/compressionkit-ecg-64x) |
| [`ecg-rvq-4x-prior`](ecg-rvq-4x-prior.md) | ECG | two_stage | 4× | ecg-rvq-4x | `ptb-xl` | pending (`Ambiq/compressionkit-ecg-4x`) |
| [`ecg-rvq-8x-prior`](ecg-rvq-8x-prior.md) | ECG | two_stage | 8× | ecg-rvq-8x | `ptb-xl` | pending (`Ambiq/compressionkit-ecg-8x`) |

### DSP Baselines

These operating points are registered and reproducible today; their HuggingFace
repos are **not yet published** (slugs reserved).

| Experiment | Modality | Family | CR | Parent | Dataset | HuggingFace (pending) |
|------------|----------|--------|----|--------|---------|-----------------------|
| `ppg-spiht-2x` | PPG | codec | 2× | — | `mesa` | `Ambiq/compressionkit-ppg-spiht-2x` |
| `ppg-spiht-4x` | PPG | codec | 4× | — | `mesa` | `Ambiq/compressionkit-ppg-spiht-4x` |
| `ppg-spiht-8x` | PPG | codec | 8× | — | `mesa` | `Ambiq/compressionkit-ppg-spiht-8x` |
| `ppg-spiht-16x` | PPG | codec | 16× | — | `mesa` | `Ambiq/compressionkit-ppg-spiht-16x` |
| `ppg-spiht-32x` | PPG | codec | 32× | — | `mesa` | `Ambiq/compressionkit-ppg-spiht-32x` |
| `ecg-spiht-2x` | ECG | codec | 2× | — | `ptb-xl` | `Ambiq/compressionkit-ecg-spiht-2x` |
| `ecg-spiht-4x` | ECG | codec | 4× | — | `ptb-xl` | `Ambiq/compressionkit-ecg-spiht-4x` |
| `ecg-spiht-8x` | ECG | codec | 8× | — | `ptb-xl` | `Ambiq/compressionkit-ecg-spiht-8x` |
| `ecg-spiht-16x` | ECG | codec | 16× | — | `ptb-xl` | `Ambiq/compressionkit-ecg-spiht-16x` |
| `ecg-spiht-32x` | ECG | codec | 32× | — | `ptb-xl` | `Ambiq/compressionkit-ecg-spiht-32x` |
| `ecg-spiht-64x` | ECG | codec | 64× | — | `ptb-xl` | `Ambiq/compressionkit-ecg-spiht-64x` |

## Validate an existing deploy package

```bash
# Fast, dataset-free check of a local release bundle.
uv run compressionkit golden validate-deploy results/ppg_rvq_64hz_08x_golden/deploy --max-vectors 1

# Stage the HuggingFace package without uploading (requires an existing deploy/ dir).
uv run compressionkit golden run ppg-rvq-4x --skip-train --publish --dry-run
```

`--skip-train` and `validate-deploy` are for existing artifacts. They do not regenerate
metrics or weights.

## Reproduce one experiment

```bash
# 1. Fetch or mount the dataset first (MESA requires NSRR_TOKEN; PTB-XL is open).
uv run compressionkit golden run ppg-rvq-4x

# DSP goldens use the same lifecycle entry point.
uv run compressionkit golden run ppg-spiht-4x

# 2. Real run, publish to HuggingFace (set HF_TOKEN first).
uv run compressionkit golden run ppg-rvq-4x --publish
```

Use `--skip-dataset-check` only when you intentionally bypass the pre-flight dataset
availability check, for example in a custom environment where the loader resolves data
through another path. Training and evaluation still need the underlying data.

## Extend an experiment

For exploration, prefer copying a small working path over adding another framework hook.
A typical extension flow is:

1. Copy the closest config from `configs/` or the closest ready-made experiment.
2. Change one decision at a time: compression ratio, augmentation, loss weighting,
   architecture width, quantization target, or scorecard policy.
3. Train and evaluate the modified experiment with the same dataset split policy.
4. Export a deploy package with `export_for_deployment(...)` or the matching DSP export
   helper.
5. Validate the package with `validate_deploy_package(...)` or
   `compressionkit golden validate-deploy <deploy-dir>`.
6. Promote it to a golden only after the artifact contract, scorecard, docs, and
   publication target are stable.

This keeps defaults easy to run while leaving advanced work flexible. The golden runner
is useful when you want release-grade repeatability; it is not required for every
custom experiment.

## Reproduce every experiment in a modality

```bash
uv run compressionkit golden run-all --modality ppg --method rvq
uv run compressionkit golden run-all --modality ppg --method spiht
uv run compressionkit golden run-all --modality ecg --method rvq
uv run compressionkit golden run-all --modality ecg --method spiht
```

See also:

- [HuggingFace testing guide](../huggingface.md) — load any AmbiqAI model in five minutes.
- [Deployment guide](../deployment.md) — exporting the artifacts to an Ambiq-class device.
- [Experiment architecture](../experiment-architecture.md) — how reusable blocks, recipes, and goldens fit together.
- [Methods · RVQ Autoencoder](../methods/rvq.md) — the architecture every entry uses.
