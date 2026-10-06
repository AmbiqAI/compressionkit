"""Architecture guardrails for composition-first experiments."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_experiment_architecture_doc_keeps_contract_at_artifacts() -> None:
    text = (ROOT / "astro-site" / "src" / "content" / "docs" / "experiment-architecture.md").read_text()
    assert "experiments pull in capabilities" in text
    assert "The v1 contract applies to artifacts" in text
    assert "one base class" in text
    assert "one orchestration framework" in text


def test_export_blocks_do_not_depend_on_golden_orchestration() -> None:
    """Reusable export blocks must stay usable outside golden runners."""
    export_files = [
        ROOT / "compressionkit" / "export" / "deploy.py",
        ROOT / "compressionkit" / "export" / "release.py",
        ROOT / "compressionkit" / "export" / "spiht_deploy.py",
        ROOT / "compressionkit" / "export" / "validate.py",
    ]
    forbidden_prefixes = ("compressionkit.experiments", "compressionkit.recipes")
    for path in export_files:
        imported = _imported_modules(path)
        forbidden = sorted(
            module for module in imported if any(module.startswith(prefix) for prefix in forbidden_prefixes)
        )
        assert not forbidden, f"{path.relative_to(ROOT)} imports orchestration modules: {forbidden}"


def test_base_rvq_trainer_is_documented_as_optional_convenience() -> None:
    text = re.sub(r"\s+", " ", (ROOT / "compressionkit" / "recipes" / "base_rvq.py").read_text())
    assert "not a required framework entry point" in text
    assert "Use this when it helps; bypass it" in text
