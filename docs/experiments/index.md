---
icon: lucide/flask-conical
---

# Golden Experiments

A **golden experiment** is a release-grade run that ships with compressionKIT for v1.
Each entry is fully declarative: a YAML config under `configs/`, a registered training
recipe, and a fixed dataset + HuggingFace repo target. The lifecycle runner
(`compressionkit golden run <id>`) reproduces it end-to-end from a clean checkout.

Two families exist:

- **`codec`** — single-stage RVQ autoencoder (encoder → RVQ → decoder).
- **`two_stage`** — paired entropy prior on top of a parent codec. Trains a small
  causal-transformer prior over the codec's token stream and bundles `prior_int8.tflite`
  alongside the codec artifacts in the same HuggingFace repo.

## v1 Registry

| Experiment | Modality | Structure | CR | Parent | Dataset | HuggingFace |
|------------|----------|--------|----|--------|---------|-------------|
| [`ppg-rvq-2x`](ppg-rvq-2x.md) | PPG | codec | 2× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-2x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-2x-v1.0) |
| [`ppg-spiht-2x`](ppg-spiht-2x.md) | PPG | codec | 2× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-spiht-2x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-spiht-2x-v1.0) |
| [`ppg-hybrid-2x`](ppg-hybrid-2x.md) | PPG | codec | 2× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-hybrid-2x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-hybrid-2x-v1.0) |
| [`ppg-rvq-4x`](ppg-rvq-4x.md) | PPG | codec | 4× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-4x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-4x-v1.0) |
| [`ppg-spiht-4x`](ppg-spiht-4x.md) | PPG | codec | 4× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-spiht-4x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-spiht-4x-v1.0) |
| [`ppg-hybrid-4x`](ppg-hybrid-4x.md) | PPG | codec | 4× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-hybrid-4x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-hybrid-4x-v1.0) |
| [`ppg-rvq-8x`](ppg-rvq-8x.md) | PPG | codec | 8× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-8x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-8x-v1.0) |
| [`ppg-spiht-8x`](ppg-spiht-8x.md) | PPG | codec | 8× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-spiht-8x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-spiht-8x-v1.0) |
| [`ppg-hybrid-8x`](ppg-hybrid-8x.md) | PPG | codec | 8× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-hybrid-8x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-hybrid-8x-v1.0) |
| [`ppg-rvq-16x`](ppg-rvq-16x.md) | PPG | codec | 16× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-16x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-16x-v1.0) |
| [`ppg-spiht-16x`](ppg-spiht-16x.md) | PPG | codec | 16× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-spiht-16x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-spiht-16x-v1.0) |
| [`ppg-hybrid-16x`](ppg-hybrid-16x.md) | PPG | codec | 16× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-hybrid-16x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-hybrid-16x-v1.0) |
| [`ppg-rvq-32x`](ppg-rvq-32x.md) | PPG | codec | 32× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-32x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-32x-v1.0) |
| [`ppg-spiht-32x`](ppg-spiht-32x.md) | PPG | codec | 32× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-spiht-32x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-spiht-32x-v1.0) |
| [`ppg-hybrid-32x`](ppg-hybrid-32x.md) | PPG | codec | 32× | — | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-hybrid-32x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-hybrid-32x-v1.0) |
| [`ppg-rvq-4x-prior`](ppg-rvq-4x-prior.md) | PPG | two_stage | 4× | ppg-rvq-4x | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-4x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-4x-v1.0) |
| [`ppg-rvq-8x-prior`](ppg-rvq-8x-prior.md) | PPG | two_stage | 8× | ppg-rvq-8x | `ppg-unified-strict-sanitize-v1` | [`Ambiq/compressionkit-ppg-8x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ppg-8x-v1.0) |
| [`ecg-rvq-2x`](ecg-rvq-2x.md) | ECG | codec | 2× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-2x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-2x-v1.0) |
| [`ecg-spiht-2x`](ecg-spiht-2x.md) | ECG | codec | 2× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-spiht-2x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-spiht-2x-v1.0) |
| [`ecg-hybrid-2x`](ecg-hybrid-2x.md) | ECG | codec | 2× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-hybrid-2x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-hybrid-2x-v1.0) |
| [`ecg-rvq-4x`](ecg-rvq-4x.md) | ECG | codec | 4× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-4x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-4x-v1.0) |
| [`ecg-spiht-4x`](ecg-spiht-4x.md) | ECG | codec | 4× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-spiht-4x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-spiht-4x-v1.0) |
| [`ecg-hybrid-4x`](ecg-hybrid-4x.md) | ECG | codec | 4× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-hybrid-4x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-hybrid-4x-v1.0) |
| [`ecg-rvq-8x`](ecg-rvq-8x.md) | ECG | codec | 8× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-8x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-8x-v1.0) |
| [`ecg-spiht-8x`](ecg-spiht-8x.md) | ECG | codec | 8× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-spiht-8x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-spiht-8x-v1.0) |
| [`ecg-hybrid-8x`](ecg-hybrid-8x.md) | ECG | codec | 8× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-hybrid-8x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-hybrid-8x-v1.0) |
| [`ecg-rvq-16x`](ecg-rvq-16x.md) | ECG | codec | 16× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-16x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-16x-v1.0) |
| [`ecg-spiht-16x`](ecg-spiht-16x.md) | ECG | codec | 16× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-spiht-16x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-spiht-16x-v1.0) |
| [`ecg-hybrid-16x`](ecg-hybrid-16x.md) | ECG | codec | 16× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-hybrid-16x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-hybrid-16x-v1.0) |
| [`ecg-rvq-32x`](ecg-rvq-32x.md) | ECG | codec | 32× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-32x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-32x-v1.0) |
| [`ecg-spiht-32x`](ecg-spiht-32x.md) | ECG | codec | 32× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-spiht-32x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-spiht-32x-v1.0) |
| [`ecg-hybrid-32x`](ecg-hybrid-32x.md) | ECG | codec | 32× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-hybrid-32x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-hybrid-32x-v1.0) |
| [`ecg-rvq-64x`](ecg-rvq-64x.md) | ECG | codec | 64× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-64x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-64x-v1.0) |
| [`ecg-spiht-64x`](ecg-spiht-64x.md) | ECG | codec | 64× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-spiht-64x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-spiht-64x-v1.0) |
| [`ecg-hybrid-64x`](ecg-hybrid-64x.md) | ECG | codec | 64× | — | `ptb-xl` | [`Ambiq/compressionkit-ecg-hybrid-64x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-hybrid-64x-v1.0) |
| [`ecg-rvq-4x-prior`](ecg-rvq-4x-prior.md) | ECG | two_stage | 4× | ecg-rvq-4x | `ptb-xl` | [`Ambiq/compressionkit-ecg-4x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-4x-v1.0) |
| [`ecg-rvq-8x-prior`](ecg-rvq-8x-prior.md) | ECG | two_stage | 8× | ecg-rvq-8x | `ptb-xl` | [`Ambiq/compressionkit-ecg-8x-v1.0`](https://huggingface.co/Ambiq/compressionkit-ecg-8x-v1.0) |

## Reproduce one experiment

```bash
# 1. Fetch the dataset (MESA requires NSRR_TOKEN; PTB-XL is open).
uv run compressionkit golden run ppg-rvq-4x --skip-dataset-check  # smoke

# 2. Real run, publish to HuggingFace (set HF_TOKEN first).
uv run compressionkit golden run ppg-rvq-4x --publish
```

## Reproduce every experiment in a modality

```bash
uv run compressionkit golden run-all --modality ppg
uv run compressionkit golden run-all --modality ecg
```

See also:

- [HuggingFace testing guide](../huggingface.md) — load any AmbiqAI model in five minutes.
- [Deployment guide](../deployment.md) — exporting the artifacts to an Ambiq-class device.
- [Methods · RVQ Autoencoder](../methods/rvq.md) — the architecture every entry uses.
