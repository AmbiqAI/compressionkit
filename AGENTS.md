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
