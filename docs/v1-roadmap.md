---
icon: lucide/map
---

# V1 Roadmap

This page turns the v1 release goals into an execution plan.

The immediate v1 target is not only a strong ECG or PPG model. The target is a
clean, modular release system for **ECG** and **PPG** golden experiments with
reproducible results, comparable scorecards, deployable artifacts, and simple
rerun paths for users.

For the artifact-level requirements, see [V1 Release Contract](release-contract.md).
For the code-organization rules behind the plan, see
[Experiment Architecture](experiment-architecture.md).
For the current ECG and PPG release-status audit, see
[V1 Golden Gap Audit](v1-golden-gap-audit.md).

## Frozen V1 Cut

For the current v1 push, the release surface is intentionally narrower than the
full space of possible experiments. The goal is to ship one defensible,
artifact-complete comparison surface for ECG and PPG, not to keep expanding the
 matrix while packaging and scorecards are still settling.

The frozen v1 cut is:

- faithful PPG RVQ goldens at 2x, 4x, 8x, 16x, 32x
- faithful PPG SPIHT goldens at 2x, 4x, 8x, 16x, 32x
- faithful ECG RVQ goldens at 2x, 4x, 8x, 16x, 32x, 64x
- faithful ECG SPIHT goldens at 2x, 4x, 8x, 16x, 32x, 64x
- one PPG denoise comparison lane at 8x: wearable RVQ candidate plus SPIHT reference
- one ECG denoise comparison lane at 8x: RVQ denoise candidate plus SPIHT reference

Explicitly not blocking this v1 cut:

- expanding denoise to a full CR ladder
- promoting additional prior-stage or hybrid variants into the public release surface
- adding IMU or other modalities

## Current Release Tracker

This is the remaining work tracker for the frozen v1 cut.

### In Progress

- Capture ECG denoise evidence.
	The evidence is now measured, but the current 8x denoise candidate does not yet support a
	release-grade claim across clean and low-noise conditions.

### Remaining

- Capture ECG denoise evidence.
	Decide whether to retrain or narrow the claim surface. The current 8x RVQ denoise run only
	outperforms the noisy-input baseline at heavier synthetic corruption and badly degrades clean
	windows, so it is not yet a safe release-facing denoise lane.
- Standardize v1 scorecards.
	Emit one canonical release-facing shape across faithful and denoise lanes, including
	dataset provenance, clean-reference deltas, and dropped-window accounting where relevant.
- Validate deploy packaging parity.
	Every promoted RVQ and SPIHT run must ship the same release-contract surface:
	`deploy_manifest.json`, `codec_spec.json`, `checksums.json`, `reference_vectors.npz`,
	scorecard artifacts, and publishable metadata.
- Write the v1 release summary.
	Publish one concise release-facing summary describing the frozen matrix, datasets,
	reproduction path, and the specific claims supported by faithful and denoise scorecards.

### Already Landed

- The exact frozen v1 release matrix is now documented in this roadmap and aligned to the
	canonical registry surface.
- Faithful PPG RVQ v1 uses the frozen `ppg-unified-strict-sanitize-v1` dataset contract.
- Faithful PPG and ECG SPIHT ladders have canonical evaluator paths and frozen expected metrics.
- Faithful ECG RVQ goldens at 2x, 4x, 8x, 16x, 32x, and 64x now have refreshed
	`n=1000` scorecards on the 256 Hz release-facing surface, including the newer
	adversarial and local-imprinting report sections.
- The full ECG 8x denoise training run completed at
	`results/ecg_rvq_256hz_08x_denoise` from `configs/ecg_rvq_256hz_08x_denoise.yaml`, with best
	validation MSE `0.081954` at epoch `209`.
- `scripts/sweep_codec_noise_ecg.py` now accepts RVQ run-dir overrides, configurable
	noise severities, larger sample counts, and separate output stems, so the ECG
	denoise candidate can be compared against SPIHT without further script edits.
- The PPG 8x wearable denoise candidate already has a completed run directory.
- The first synthetic denoise comparison confirms that RVQ materially improves noisy PPG
	windows over the SPIHT reference at 8x.

### ECG Denoise Evidence Snapshot

The release candidate is the clean-target 8x run defined in
`configs/ecg_rvq_256hz_08x_denoise.yaml`.

Release-grade sweep at `n_windows=500`:

| Noise pct | Baseline PRD vs clean | SPIHT 8x PRD vs clean | SPIHT 8x denoise delta PRD | RVQ 8x PRD vs clean | RVQ 8x denoise delta PRD | RVQ relative PRD reduction |
|-----------|------------------------|------------------------|----------------------------|---------------------|--------------------------|----------------------------|
| 0.0 | 0.00 | 0.59 | -0.59 | 35.97 | -35.97 | n/a |
| 0.4 | 37.69 | 36.28 | 1.41 | 42.18 | -4.49 | -11.9% |
| 0.8 | 66.03 | 65.38 | 0.65 | 55.29 | 10.74 | 16.3% |
| 1.2 | 84.67 | 83.67 | 1.00 | 68.86 | 15.81 | 18.7% |

Dialed-back retrain (`configs/ecg_rvq_256hz_08x_denoise_v2.yaml`) — softer target filter
(0.5–70 Hz, order 2), lighter `filtered_loss` (weight 0.3, cutoff 40 Hz) and milder input
corruption. Best `val_mse=0.067827` at epoch 138 (vs v1 0.081954), but the release-grade sweep
at `n_windows=500` is a regression on both axes:

| Noise pct | Baseline PRD vs clean | RVQ 8x v1 PRD / delta | RVQ 8x v2 PRD / delta |
|-----------|------------------------|------------------------|------------------------|
| 0.0 | 0.00 | 35.97 / -35.97 | 37.98 / -37.98 |
| 0.2 | 19.65 | — | 40.29 / -20.64 |
| 0.4 | 37.69 | 42.18 / -4.49 | 46.91 / -9.22 |
| 0.8 | 66.03 | 55.29 / +10.74 | 62.80 / +3.23 |
| 1.2 | 84.67 | 68.86 / +15.81 | 77.43 / +7.24 |

Decisive control: the **non-denoise** RVQ 8x golden scores **3.23% clean PRD** on this same
synthetic sweep (`summary_ecg.json`). The ~36–38% clean PRD on both denoise models is therefore
*entirely denoise-induced distortion*, not a compression/out-of-domain floor.

Comparison artifacts:

- `results/_codec_noise_sweep/summary_ecg_denoise_8x.json` (v1)
- `results/_codec_noise_sweep/summary_ecg_denoise_8x_v2.json` (v2)
- `results/_codec_noise_sweep/summary_ecg.json` (non-denoise control, RVQ 8x clean PRD 3.23)
- `results/ecg_rvq_256hz_08x_denoise/quality_scorecard.json`

Current interpretation:

- SPIHT remains essentially faithful on clean ECG and provides little true denoising benefit.
- The 8x ECG RVQ denoise candidate does denoise heavier synthetic corruption, but only once the
	input is already badly degraded.
- Both denoise configs catastrophically damage clean windows (3.23% → 36–38% PRD) and lose to the
	noisy-input baseline at low corruption, so neither is suitable as a release-facing denoise lane.
- Dialing back the denoising (v2) did **not** recover clean fidelity — it slightly worsened clean
	PRD *and* roughly halved the high-noise denoising benefit (delta +15.81 → +7.24 at 1.2). The
	`val_mse` improvement is against the milder filtered target and does not transfer to the
	clean-truth sweep. The clean-vs-denoise tradeoff is intrinsic to the denoise objective, not a
	tunable-aggressiveness knob.
- For the current v1 cut, ECG denoise remains the main open blocker unless the release claim is
	narrowed to high-noise-only behavior or a fundamentally different approach is used.

### PPG Denoise Evidence Snapshot

The current PPG denoise lane uses the completed 8x wearable RVQ candidate run
`results/ppg_rvq_64hz_08x_unified_bridge_strict_sanitize_balanced_denoise_tier1_wearable_gpu`
and compares it against the SPIHT 8x reference on the same synthetic-truth
clean/noisy bundle.

Release-grade sweep at `n_windows=500`:

| Noise pct | Baseline PRD vs clean | SPIHT recon PRD vs clean | SPIHT denoise delta PRD | RVQ recon PRD vs clean | RVQ denoise delta PRD | RVQ relative PRD reduction |
|-----------|------------------------|---------------------------|-------------------------|------------------------|-----------------------|----------------------------|
| 0.4 | 37.76 | 36.75 | 1.01 | 20.87 | 16.89 | 44.7% |
| 0.8 | 66.19 | 65.11 | 1.08 | 34.99 | 31.20 | 47.1% |
| 1.2 | 84.88 | 83.98 | 0.90 | 49.20 | 35.67 | 42.0% |

Comparison artifacts:

- `results/_synthetic_scorecards/ppg_08x_noise040/comparison.json`
- `results/_synthetic_scorecards/ppg_08x_noise080/comparison.json`
- `results/_synthetic_scorecards/ppg_08x_noise120/comparison.json`

Current interpretation:

- SPIHT remains a fidelity reference at fixed bitrate, but it does not provide meaningful denoising on this noisy-input evaluation surface.
- The 8x wearable RVQ candidate consistently denoises moderate-to-heavy synthetic corruption while preserving strong clean-reference similarity.
- For the current v1 cut, one 8x PPG denoise lane is sufficient to support the claim that RVQ offers a material advantage over SPIHT in noisy wearable conditions.

## V1 Outcome

compressionKIT v1 should let a user:

- run a documented golden release from a clean checkout
- modify a config and rerun without changing framework internals
- add a new experiment by composing blocks and registering it cleanly
- compare DSP and AI methods with the same release-grade scorecard surface
- download a validated release package from HuggingFace and hydrate it from packaged artifacts

compressionKIT v1 should let the team ship:

- golden registries for PPG and ECG
- both DSP and AI method families where they are promoted to release grade
- frozen deploy packages with scorecards, manifests, checksums, and reference vectors
- website and HuggingFace pages rendered from packaged release artifacts

IMU remains important, but it is explicitly deferred until the ECG and PPG v1
release system is stable.

## Product Axes

The repo should treat these as first-class axes, not as special cases:

| Axis | Required for v1 |
|------|------------------|
| Modality | PPG, ECG |
| Method family | DSP, AI where promoted to release-grade |
| Release path | local run, validated deploy package, HuggingFace publication |
| Experiment mode | golden release, tweak-and-rerun, new custom experiment |

Hybrid methods remain supported, but they should not block v1 unless a specific operating point is explicitly promoted into the release registry.

## Immediate V1 Focus

For the first v1 cut, priority goes to:

- reproducing the current ECG and PPG golden results cleanly
- generating proper release-grade scorecards for those goldens
- exporting and publishing validated release assets to HuggingFace
- making the codebase easier to reason about, with clearer logical ordering and cleaner boundaries
- keeping golden experiments close to normal user experiments, with the golden path adding registered hooks, release metadata, publication, and docs

This means the golden flow should feel like a thin release layer on top of the
same reusable blocks that ordinary users call from configs and scripts.

## Guiding Principles

- Keep the codebase modular: blocks first, ready-made experiments second, golden release automation at the boundary.
- Keep the release contract artifact-driven: validation should target packaged outputs, not training control flow.
- Keep DSP and AI symmetric at the release boundary: same scorecard shape, same validation contract, same publication story.
- Keep modality-specific rules local to modality blocks: preprocessing, scorecard extensions, dataset contracts, and reference metrics.
- Keep user workflows simple: one command to run a golden, one command to publish, one obvious way to fork a config.

## Epic 1: Clean Experiment And Release Architecture

Goal: make the codebase easier to reason about and keep golden flows close to normal user flows.

Deliverables:

- shared experiment contract for dataset, preprocessing, train, eval, export, validate, publish
- block-level APIs that can be used from a custom script without joining the golden runner
- golden registry entries that map cleanly to configs, docs, HuggingFace targets, and release pages
- clearer logical ordering across blocks, ready-made experiments, and golden release automation
- golden-specific hooks kept at the release boundary rather than spread across training code

Exit criteria:

- user experiments and golden experiments share the same reusable building blocks
- golden automation reads like a thin release wrapper rather than a separate framework
- code organization makes it obvious where to look for datasets, preprocessing, models, evaluation, export, and release logic

## Epic 2: Golden Reproducibility And Experiment UX

Goal: make goldens easy to rerun, easy to tweak, and easy to extend.

Deliverables:

- versioned golden configs for every released operating point
- config inheritance or override patterns for small experiment changes
- CLI override support for common knobs such as CR, frame size, augmentation, export flags, and evaluation settings
- experiment templates for new goldens, ablations, and new modality-method combinations
- docs showing the shortest path for "rerun", "tweak", "publish", and "add a new experiment"
- published reproduction instructions for every golden release

Exit criteria:

- a user can rerun a golden from config only
- a user can fork a golden by editing config values instead of patching framework code
- a contributor can add a new experiment with a config and registry entry plus minimal glue
- every published golden tells the user exactly how it was reproduced

## Epic 3: Release-Grade Scorecards

Goal: define one release decision surface for promoted ECG and PPG goldens.

Deliverables:

- shared scorecard schema across method families
- modality-specific sections for ECG and PPG
- frozen release metrics for fidelity, physiology or task preservation, stitching or continuity, and deployment context
- hallucination and imprinting sections where applicable
- threshold-based release checks for required scorecard metrics

Exit criteria:

- DSP and AI codecs for the same modality and CR can be compared from the same scorecard schema
- release candidates can be accepted or rejected from packaged scorecards without manual plot review

## Epic 4: Deploy Packaging, Validation, And HuggingFace Export

Goal: make packaged artifacts the source of truth for v1.

Deliverables:

- unified top-level package contract across DSP and AI
- required release files such as `deploy_manifest.json`, `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, `scorecard.json`, and `model_card.json`
- strict deploy validation driven by packaged artifacts only
- cross-runtime reference-vector checks for Python and embedded-facing paths
- package metadata sufficient for browser demos and website rendering
- one obvious export and publish path for HuggingFace releases

Exit criteria:

- a runtime can hydrate from the package without reading the original YAML config
- release validation fails on missing artifacts or broken reference vectors
- a validated golden package can be exported and published without custom per-run glue

## Epic 5: Golden Runner And Publication UX

Goal: make golden runs feel like user experiments plus registered release hooks.

Deliverables:

- one standard run, validate, export, and publish path for all goldens
- resumable publish flow that does not require retraining
- model-card generation from packaged metadata and scorecards
- HuggingFace naming and staging rules for ECG and PPG releases
- docs that explain how to pull and run a released codec in minutes

Exit criteria:

- a validated golden package can be published with one command
- the HF repo contents match the local validated package

## Epic 6: PPG And ECG Golden Completion

Goal: finish the ECG and PPG golden set before broadening scope.

Deliverables:

- reproducible ECG and PPG golden configs for the chosen operating points
- frozen scorecards for each promoted golden
- validated deploy packages for each promoted golden
- published HuggingFace assets and docs pages for each promoted golden

Exit criteria:

- ECG and PPG goldens can be rerun, validated, exported, published, and documented end to end

## Epic 7: Post-V1 Expansion Hooks

Goal: leave clean hooks for DSP parity and later IMU work without blocking the first v1 cut.

Deliverables:

- release boundaries that do not assume RVQ-only goldens forever
- scorecard and package contracts that can absorb DSP and IMU later without being rewritten
- registry and docs patterns that support later expansion cleanly

Exit criteria:

- broadening to DSP-heavy or IMU-heavy work does not require undoing the first v1 cleanup

## Milestones

### Milestone 0: Freeze The V1 Contract

- freeze the release artifact schema
- freeze the scorecard schema
- freeze naming conventions for run IDs, HuggingFace repos, and deploy bundles
- freeze the initial ECG and PPG faithful and denoise set for v1

### Milestone 1: Unify The Release Boundary

- normalize DSP and AI deploy manifests
- make `codec_spec.json` authoritative for runtime hydration
- unify scorecard generation and release validation

### Milestone 2: Make Golden Runs Easy

- improve `compressionkit golden run <id>` and `run-all`
- add docs for rerun, tweak, and publish workflows
- add config override ergonomics and templates
- keep the golden path close to the ordinary user path, with release hooks layered on top

### Milestone 3: Complete PPG And ECG Release Flows

- make the current PPG and ECG goldens fully artifact-complete
- ensure scorecards include the frozen release and safety sections
- ensure HuggingFace publication and docs are package-derived

### Milestone 4: Harden CI And Publication

- add CI checks for package completeness, reference vectors, scorecard schema, and publish readiness
- reject incomplete release candidates automatically

### Milestone 5: Expand Beyond The First V1 Cut

- deepen DSP parity beyond the initial ECG and PPG golden matrix where additional operating points or variants are justified
- add IMU once the ECG and PPG release flow is stable and clean

## V1 Golden Matrix

This is the frozen v1 matrix for the current release push.

| Lane | Modality | DSP Baseline | AI Baseline | Operating Points | Notes |
|------|----------|--------------|-------------|------------------|-------|
| Faithful | PPG | SPIHT | RVQ | 2x, 4x, 8x, 16x, 32x | Registry-backed goldens on the frozen unified strict-sanitized PPG dataset |
| Faithful | ECG | SPIHT | RVQ | 2x, 4x, 8x, 16x, 32x, 64x | Registry-backed goldens on PTB-XL |
| Denoise | PPG | SPIHT reference | RVQ wearable candidate | 8x | Synthetic-truth clean/noisy/reconstruction evaluation to demonstrate wearable denoising value |
| Denoise | ECG | SPIHT reference | RVQ denoise candidate | 8x | Same release-facing evaluation shape as PPG denoise |

IMU is intentionally out of scope for the first v1 cut.

Minimum release expectation per matrix cell:

- one canonical experiment ID
- one source-controlled config
- one reproducible command
- one frozen deploy package
- one frozen scorecard
- one HuggingFace destination
- one rendered docs page

## Backlog Seeds

These should become tracked issues or epics:

1. Unify DSP and AI deploy manifests under the v1 package contract.
2. Clean up code organization so user experiments and golden experiments share the same blocks with thin release hooks.
3. Keep the golden registry family-symmetric across ECG and PPG operating points, with method-aware tooling and docs.
4. Add config overlay or override UX for quick golden variations.
5. Add scorecard thresholds for release gating, including safety metrics.
6. Add CI checks for package completeness and reference-vector validation.
7. Generate docs and model cards from packaged artifacts rather than hand-authored tables.
8. Add user-facing templates for new experiments and new golden candidates.

## Definition Of Done

compressionKIT is ready for v1 when all of the following are true:

- ECG and PPG each have documented v1 golden releases at the chosen operating points.
- ECG and PPG each have documented DSP and AI golden releases at the chosen operating points.
- each promoted golden release is reproducible from config and one documented command.
- each golden release emits a validated package and frozen scorecard.
- HuggingFace repos are populated from validated release packages.
- users can tweak a golden config and rerun without changing core framework code.
- contributors can add a new experiment without introducing a new orchestration silo.
- release decisions are driven by packaged scorecards and validation checks, not by ad hoc notebook outputs.
- the golden path remains logically close to the normal user path, with release and publication concerns added through registered hooks and reusable helpers.

## Non-goals For The First V1 Cut

- making every historical experiment conform to the new golden contract
- requiring one mandatory training base class or one mandatory runner for all experiments
- blocking the first v1 cut on IMU or on every post-v1 DSP variant beyond the chosen ECG and PPG golden matrix
- blocking v1 on hybrid methods unless a hybrid entry is intentionally promoted to release-grade
- solving every post-v1 downstream analytics task inside compressionKIT itself
