# Active recipes

This folder is the home of **in-progress, experimental training recipes**.
Unlike the golden recipes in `compressionkit/recipes/` (which ship with
the package), the files here are source-controlled but not importable as
part of the `compressionkit` package. They are meant to be copied,
modified, and iterated on without ceremony.

## How to use

1. Copy a golden recipe as your starting point:

   ```bash
   cp compressionkit/recipes/train_ppg_rvq.py recipes/train_ppg_rvq_myidea.py
   ```

2. Edit freely. Add new loss terms, swap the model, try a different
   dataset — the recipe file is a plain Python script. Because each step
   delegates to a building block in `compressionkit.trainers`, you can
   override any piece without re-implementing the whole flow.

3. Give the recipe a CLI name with the `@recipe` decorator so it can be
   invoked via `python <file>.py --config …` or through the
   `compressionkit` multiplexer once imported:

   ```python
   from compressionkit.recipes import recipe
   from compressionkit.configs.ppg_rvq import PpgRvqConfig

   @recipe("train-ppg-rvq-myidea", config_cls=PpgRvqConfig)
   def train(cfg): ...
   ```

4. Run it directly:

   ```bash
   uv run python recipes/train_ppg_rvq_myidea.py --config configs/myidea.yaml
   ```

5. If a recipe here turns out to be broadly useful, promote it into
   `compressionkit/recipes/` and add a dedicated console script in
   `pyproject.toml`.

## Balance: config vs. code

The YAML config should capture **what varies across runs** (dataset
paths, model width, augmentation strength, loss weights). The recipe
file captures **how the pieces fit together**. When a setting only
makes sense for one recipe, prefer hard-coding it in the recipe rather
than widening the config schema.
