# AGENTS.md

These are the project guidelines for agentic AI models working in this repository.

## Repository Focus
- Build a compression toolkit for physiological signals (ECG/PPG) targeting edge and wearable devices.
- Emphasize efficient on-device inference, compact models, and deployable artifacts (e.g., LiteRT, headers).

## Core Principles
- Lean heavily on Keras 3.
- Create reusable, focused components and functions.
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

## Training Recipes
- Use modular building blocks that can be easily swapped out.
- Use config to pass arguments to blocks; do not use configs to decide major building blocks.
- Major building blocks include datasets, dataloaders, models, optimizers, loss functions, metrics, callbacks, and training loops.
- For logging and metrics, prefer TensorBoard and Keras callbacks where possible.

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
- Golden run naming convention: `{modality}_rvq_{sample_rate}hz_{cr}x_golden/` (e.g., `ppg_rvq_64hz_04x_golden/`).
- Deploy artifacts are exported via `compressionkit.export.deploy.export_for_deployment()`.

### HuggingFace Releases
- Model repos follow the naming convention `Ambiq/compressionkit-{modality}-{cr}x`.
- Use `HF_TOKEN` env var for authentication — **never commit tokens**.
- Publish via `scripts/publish_to_huggingface.py` (when available).
- Release checklist: export deploy artifacts → generate scorecard → build model card → publish to HF.

### Dataset Licensing
- PTB-XL (ECG): CC BY 4.0 — sample data may be redistributed.
- MESA (PPG): NSRR restricted — **do not redistribute**; use synthetic physiokit data for examples.

### Golden Run Conventions
- Naming: `{modality}_rvq_{sample_rate}hz_{cr}x_golden/` (e.g., `ppg_rvq_64hz_04x_golden/`).
- Golden runs live under `results/` which is git-ignored; do **not** commit model weights or large artifacts.
- Deploy artifacts (`deploy/` subdirectory) are the publishable outputs: `.tflite`, `.h`, `.npz`, `.json`.
- Commit only configs, scripts, and code — golden results are reproduced from configs or published to HuggingFace.
