"""Export documentation facts without importing the training package."""

import importlib.util
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("docs_experiment_registry", repo / "compressionkit/experiments/registry.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
output = Path("src/data/experiments.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps([entry.model_dump(mode="json") for entry in module.GOLDEN_REGISTRY], indent=2) + "\n")
