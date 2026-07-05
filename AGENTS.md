# AGENTS.md

These are the project guidelines for agentic AI models working in this repository.

## Repository Focus
- Build a compression toolkit for physiological signals (ECG/PPG) targeting edge and wearable devices.
- Emphasize efficient on-device inference, compact models, and deployable artifacts (e.g., LiteRT, headers).

## Core Principles
- Lean heavily on Keras 3.
- Create reusable, focused components and functions.
- Before writing new extraction/training/eval logic, check whether an existing block already does it (`compressionkit/trainers/`, `compressionkit/datasets/`, `compressionkit/evaluation/`, `compressionkit/pipeline/`) and reuse or extend it rather than writing a parallel implementation in a script.
- Prioritize readability and maintainability.
- The focus is porting toward Edge AI.
- Utilize HeliaEdge (python package) where applicable; if something fits better in HeliaEdge, we can create a PR there later.
- Every algorithm and approach must be portable to embedded C; assume no portable C implementation exists today.
- Prefer fixed memory layouts and avoid dynamic allocation (no malloc/free) when designing algorithms.
- Prefer modern Python 3 type hints (use `list[str]` over `List[str]`, `str | None` over `Optional[str]`, etc.).
- Prefer Pydantic models/objects over dictionaries where it makes sense.
- For Keras 3 / TF preprocessing, prefer using dicts with data and labels.
- Use Keras 3 built-in ops rather than backend-specific ops where possible.
- For AI models, only use operators that can be ported to LiteRT (TFLite); assume deployment will use INT8 or INT16x8 quantization.
- Create highly efficient code but avoid premature optimization.
- No notebooks and no one-off "data science slop" in the shipped codebase: research scripts under `scripts/` are fine for one-off exploration, but logic used more than once (or by more than one script) belongs in a tested `compressionkit/` package module. Don't let duplicate implementations of the same idea accumulate across scripts.

## Training Recipes
- Use modular building blocks that can be easily swapped out.
- Use config to pass arguments to blocks; do not use configs to decide major building blocks.
- Major building blocks include datasets, dataloaders, models, optimizers, loss functions, metrics, callbacks, and training loops.
- For logging and metrics, prefer TensorBoard and Keras callbacks where possible.

## Experiment Architecture
Three layers, kept deliberately separate (see `docs/experiment-architecture.md` for the full rationale):
1. **Blocks** — small, importable capabilities (datasets/cache builders, preprocessing/augmentation, model builders, losses/metrics/callbacks, scorecard builders, exporters/validators). No hidden global state, no mandatory registries.
2. **Ready-made experiments** — readable end-to-end flows built from blocks. Opinionated but never the only path.
3. **Golden releases** — the release boundary: stable IDs, reproducible configs, scorecards, deploy packages. An experiment becomes golden by passing the artifact contract, not by being rewritten around a mandatory base class.

- Prefer composition over orchestration: experiments should pull in reusable blocks, not be forced into one framework-shaped entry point.
- `BaseRVQTrainer` and `compressionkit golden` are convenience paths for shipped/golden flows, not mandatory APIs for every experiment.
- Release validation should target artifacts (`deploy_manifest.json`, `codec_spec.json`, `checksums.json`, `reference_vectors.npz`, scorecards), not how an experiment was written.
- When a shared trainer or runner starts accumulating export, validation, or scorecard policy, extract that behavior into standalone helpers before adding new abstract hooks.
- Consider moving generic Keras 3 edge-model training/deployment blocks to HeliaEdge once they are no longer compressionKIT-specific. Good candidates include RVQ/VQ layers, reusable RVQ architectures, generic losses/metrics/callbacks, LiteRT export helpers, and reference-vector utilities.
- Keep modality-specific datasets, physiological scorecards, signal preprocessing policy, golden registry entries, HuggingFace naming, and v1 release policy in compressionKIT.
- Entropy-coding research (new priors/entropy coders as a second stage on top of a frozen codec) follows this same pattern; see `docs/entropy-coding-research.md` for the current blocks, the promotion path, and known lessons (don't re-derive these).

## Style
- Docstrings should use Google style.

## Tooling
- We primarily develop in a dev container and use `uv` for project and package management.

## Project Management
- GitHub Project board: https://github.com/orgs/AmbiqAI/projects/36
- All work items should have a corresponding GitHub Issue with appropriate labels.
- Labels: `evaluation`, `deployment`, `12-lead`, `documentation`, `infrastructure`, `runtime`.
- Link issues to the project board via `gh project item-add 36 --owner AmbiqAI --url <issue_url>`.

### Issue & PR Workflow
- Branch naming: `issue-<number>-<short-description>` (e.g., `issue-13-agents-md-update`).
- PRs should reference the issue they resolve (e.g., `Closes #13`).
- Use draft PRs for work-in-progress.
- Request Copilot review on all PRs before merging.

## Release & Deployment
- This project is **pre-v1**; breaking changes are permitted on major updates until v1.
- Customers will see this codebase — keep public-facing code, configs, and docs clean.
- Golden run naming: `{modality}_rvq_{sample_rate}hz_{cr}x_golden/` (e.g., `ppg_rvq_64hz_04x_golden/`). Runs live under `results/` (git-ignored) — commit only configs, scripts, and code; never model weights or large artifacts.
- Deploy artifacts are exported via `compressionkit.export.deploy.export_for_deployment()` into each run's `deploy/` subdirectory (`.tflite`, `.h`, `.npz`, `.json`) — the only publishable outputs.
- Codec-family dispatch (loader class, required files, model card, default license) is centralized in `compressionkit.export.family_registry.FAMILY_REGISTRY` — adding a new codec family (`rvq`/`spiht`/`hybrid` today) means adding one entry there, not editing the runtime loader, validator, and publisher independently. See `docs/adding-a-codec-family.md` for the full concept-to-release lifecycle.
- Before cutting a release, run `compressionkit golden validate-all --strict-release` to audit every registered golden's existing local deploy package for schema drift (gitignored `results/` can silently go stale relative to code changes).

### HuggingFace Releases
- Model repos follow the naming convention `Ambiq/compressionkit-{modality}-{cr}x-{version}` (RVQ) or `Ambiq/compressionkit-{modality}-{method}-{cr}x-{version}` (SPIHT/hybrid), e.g. `Ambiq/compressionkit-ppg-4x-v1.0`.
- Use `HF_TOKEN` env var for authentication — **never commit tokens**.
- Publish via `scripts/publish_to_huggingface.py` (when available).
- Release checklist: export deploy artifacts → generate scorecard → build model card → publish to HF.

### Dataset Licensing
- PTB-XL (ECG): CC BY 4.0 — sample data may be redistributed.
- MESA (PPG): NSRR restricted — **do not redistribute**; use synthetic physiokit data for examples.
