# RVQ export repair audit — issue #74

2026-10-05. Source checkout: `7fb3663`; repair branch: `issue-74-rvq-codebook-export`.

## Affected artifacts

All 11 distinct registered single-stage RVQ runs contain exported EMA accumulators. Four two-stage registry entries reuse those run directories; their entropy priors were not requalified. The codebook payloads at all 11 published revisions pinned in compressionkit-dev's `experiments/mvp_v02/inventory.json` match the faulty local codebooks exactly. This establishes codebook identity, not identity of every file in those bundles.

| Published model | Pinned revision |
|---|---|
| `Ambiq/compressionkit-ecg-2x-v1.0` | `4d07900f24978ee3830ee2a6a4e2becaa0880645` |
| `Ambiq/compressionkit-ecg-4x-v1.0` | `f8be1ba5454d0eb5c645393ec3261e74d0bf26a8` |
| `Ambiq/compressionkit-ecg-8x-v1.0` | `548a82529fc10075c31f19ab1ed8c09dfb2e58f4` |
| `Ambiq/compressionkit-ecg-16x-v1.0` | `e1c78d7227fec77887bd6a619d3463e238c3e01d` |
| `Ambiq/compressionkit-ecg-32x-v1.0` | `13c9be7e94b80191385f6be3aad32d2effb59e0b` |
| `Ambiq/compressionkit-ecg-64x-v1.0` | `49c2dd0ddb04b23d6817e584c9865046ecd15424` |
| `Ambiq/compressionkit-ppg-2x-v1.0` | `af00d62fcdd8e8c12a2929bae9d22911ab518966` |
| `Ambiq/compressionkit-ppg-4x-v1.0` | `8e3b465983808d50d99799df6135aaffe68029ff` |
| `Ambiq/compressionkit-ppg-8x-v1.0` | `7e2b0cea808d1b1f168c4131ea5c00ffe50e9cce` |
| `Ambiq/compressionkit-ppg-16x-v1.0` | `928d302ac822712c4513488f6d4e0fef962bde37` |
| `Ambiq/compressionkit-ppg-32x-v1.0` | `bff7e89fbe3c6b35a2db829d7da6022eb26fcf34` |

## Before and after

Ten saved reference frames per modality were compared with the saved Keras encoder/decoder and restored training quantizer. These measurements assess deployment fidelity, not physiological accuracy or clean-reference denoising quality.

| Measurement | ECG 8x | PPG 8x |
|---|---:|---:|
| Exported levels | 4 → 2 | 4 → 2 |
| Codebook value bytes | 65,536 → 32,768 | 65,536 → 32,768 |
| Complete packet bytes per frame | 260 → 132 | 164 → 84 |
| Median reconstruction PRD against trained discrete path (%) | 6.36077 → 0.00002241 | 1.62395 → 0.00001964 |

Both corrected packages have exact token and quantized-latent parity on identical source latents. Keras versus FP32 LiteRT encoder tokens agree on every audited frame. Maximum encoder/decoder differences are below 2.4e-6 in normalized units.

Packet checks reused `RVQFrames` from the sibling checkout at `80d774d`, supplied with the compressionkit runtime. All 40 old/new packet round trips passed. Counts include the 4-byte float16 mean/scale header and fixed-width tokens. These checks used FP32 encoders and do not establish bit-exactness of MCU INT8 kernels.

## Validation and provenance

- Reproduced the original extractor returning four tables from an actual two-level EMA layer.
- Regression coverage includes EMA/non-EMA, warm-start state, numeric checkpoint and level ordering, malformed layouts, independent training references, and C99 compilation with float32 table parity.
- After review fixes, 50 focused tests pass under normal GPU devcontainer settings. Float references use a local CPU context to avoid TF32 rounding; INT8-only decoders retain both float-companion training checks and quantized deployed-reference checks. A fresh independent review found no remaining actionable findings and verified 19 focused tests.
- All 11 regenerated single-stage candidates pass strict deploy validation on ten independent training-reference frames each, with no errors or warnings.
- Recomputed INT8, FP16, and INT16x8 model-card reports on real disjoint validation partitions of 2,048 frames per run. Historical physiological scorecards remain historical results; no physiological evaluation or retraining was performed.
- Original run/deploy artifacts remain unchanged. Corrected candidates and evidence are in the repair worktree's git-ignored `results/issue-74/`: `audit.json`, `source-checksums.json`, `strict-validation.json`, `environment.txt`, test output, and investigation scripts.

The repair requires declared EMA layout and independent trained-quantizer references for strict RVQ validation. Existing packets must retain their original bundle identity because corrected level counts change packet interpretation. Corrected model artifacts remain unpublished; this audit does not authorize a release or merge.
