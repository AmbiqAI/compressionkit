"""Writers for per-run safety artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from compressionkit.evaluation.adversarial import run_adversarial_battery
from compressionkit.evaluation.rvq_codec import RvqCodec


def write_adversarial_metrics_report(
    run_dir: Path,
    *,
    modality: str,
    output_path: Path | None = None,
    n_frames: int = 16,
    seed: int = 0,
) -> Path:
    """Run the adversarial battery for a trained codec and persist the rows."""
    run_dir = Path(run_dir)
    codec = RvqCodec.from_run_dir(run_dir, modality=modality)
    rows = [result.to_dict() for result in run_adversarial_battery(codec, n_frames=n_frames, seed=seed)]
    out = output_path or (run_dir / "adversarial_metrics.json")
    out.write_text(json.dumps(rows, indent=2))
    return out


__all__ = ["write_adversarial_metrics_report"]
