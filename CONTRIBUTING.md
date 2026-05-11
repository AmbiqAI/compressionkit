# Contributing to CompressionKIT

Thanks for your interest in contributing! This guide covers the local dev
workflow, code style, and how to add new signals, models, or configs.

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for project and package management
- (Optional) NVIDIA GPU + CUDA for training — a dev container with CUDA is provided under `.devcontainer/`

## Getting set up

```bash
git clone https://github.com/AmbiqAI/compressionkit.git
cd compressionkit
uv sync --group dev            # runtime + dev deps
uv sync --group docs           # (optional) docs build deps
uv run pre-commit install      # install git hooks
```

After this, every commit will be linted and formatted automatically.

## Daily workflow

```bash
# Run tests
uv run pytest

# Lint + auto-format
uv run ruff check --fix .
uv run ruff format .

# Run the whole pre-commit suite manually
uv run pre-commit run --all-files

# Build the docs locally
uv run --group docs zensical serve   # live-reload at http://localhost:8000
uv run --group docs zensical build   # static site in site/
```

## Code style

- Google-style docstrings.
- Modern Python 3 type hints: `list[str]`, `str | None` — never `List[str]` or `Optional[str]`.
- Prefer Pydantic models over untyped dicts for configuration surfaces.
- For Keras layers / models, use `keras.ops` instead of backend-specific ops.
- Only use operators portable to LiteRT (TFLite) for model code; assume INT8 / INT16×8 quantization at deployment.
- Avoid dynamic allocation (`malloc`-style growth) in algorithm designs that are meant to port to embedded C.
- Keep comments and docstrings focused; don't add them to code you didn't touch.

Ruff enforces the rest. The config lives in [`pyproject.toml`](pyproject.toml).

## Adding things

### A new compression ratio

1. Copy an existing YAML in [`configs/`](configs/) (e.g., `ppg_rvq_64hz_08x_golden.yaml`).
2. Adjust `downsample_factor` and `num_levels` to match your target CR.
3. Run `uv run train-ppg-rvq --config configs/your_new_config.yaml`.
4. Artifacts land in `results/<run_name>/`.

### A new signal type

1. Add a dataset class under `compressionkit/datasets/` that can load individual signals and build a `tf.data.Dataset` pipeline.
2. Add a preprocessing module under `compressionkit/preprocessing/`.
3. Add a trainer under `compressionkit/trainers/` wiring dataset + model + loss + export.
4. Add a recipe under `compressionkit/recipes/` that exposes `train(cfg)` and `main()`, then register its `main` in `pyproject.toml` under `[project.scripts]`.
5. Add a few tests under `tests/` for the new dataset / trainer.

### A new layer or model variant

1. Drop a Keras layer into `compressionkit/layers/` (with `@keras.saving.register_keras_serializable`).
2. Add a model builder in `compressionkit/models/` that wires it up.
3. Expose both via the relevant package `__init__.py` / top-level `compressionkit/__init__.py`.
4. Add the new knobs to the relevant config Pydantic model under `compressionkit/configs/`.

## Testing policy

- All new public functions should have at least a smoke test (importable, callable on a small synthetic input).
- Tests live under [`tests/`](tests/) and follow the `test_*.py` naming convention.
- Mark long-running or GPU-bound tests with `@pytest.mark.slow` or `@pytest.mark.gpu` so they can be filtered:

  ```bash
  uv run pytest -m "not slow"
  ```

## Documentation

- Public docs live under [`docs/`](docs/) and are served via Zensical / MkDocs Material.
- If you add a new CLI command, config, or model family, add a page (or extend an existing one) and update the nav in [`zensical.toml`](zensical.toml).
- API docs are generated via `mkdocstrings` — a well-formed docstring is enough, no extra wiring needed.

## Pull requests

Keep PRs focused: one logical change per PR, with tests and docs updated in the same commit when relevant.
CI runs ruff + pytest + shell-script syntax checks on every PR; all three must pass before review.

Thanks!
