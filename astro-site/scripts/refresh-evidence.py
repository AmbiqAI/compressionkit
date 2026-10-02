"""Export scorecard tables without replacing authored documentation."""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("evidence_export", repo / "scripts/build_customer_evidence_summary.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    snapshot = json.loads(Path("content-data/customer-evidence.json").read_text())
    rows = {}
    digest = hashlib.sha256()
    for modality, runs in module.DEFAULT_RUNS.items():
        for run in runs:
            row = module.build_row(args.results_dir, run, modality)
            if row is None:
                raise SystemExit(f"Missing scorecard for {run}; refusing partial evidence refresh")
            rows[(modality, row.cr_label)] = row
            for path in sorted((args.results_dir / run).rglob("*")):
                if path.is_file() and (path.name == "quality_scorecard.json" or path.parent.name == "deploy"):
                    digest.update(str(path.relative_to(args.results_dir)).encode())
                    digest.update(hashlib.sha256(path.read_bytes()).digest())

    columns = {
        "quality": [("n_samples", 0, ""), ("truth_prd_clean", 2, ""), ("faithful_prd", 2, ""), ("hr_mae_bpm", 2, "")],
        "files": [("edge_payload_kib", 0, " KiB"), ("encoder_kib", 0, " KiB"), ("decoder_kib", 0, " KiB"), ("codebook_kib", 0, " KiB")],
        "detail": [("prdn_noise", 2, ""), ("band_error", 4, ""), ("coherence", 4, ""), ("seam_ratio", 3, "")],
    }
    for table in snapshot["tables"]:
        modality, kind = table["id"].split("-")
        for cells in table["rows"]:
            row = rows[(modality, cells[0])]
            for index, (field, precision, suffix) in enumerate(columns[kind], 1):
                value = getattr(row, field)
                if field == "edge_payload_kib" and any(part is None for part in (row.encoder_kib, row.decoder_kib, row.codebook_kib)):
                    value = None
                if value is not None and not math.isfinite(float(value)):
                    raise SystemExit(f"Nonfinite {field} in {row.run_name}")
                cells[index] = "-" if value is None else f"{value:.{precision}f}{suffix}"
    snapshot["provenance"] = {"kind": "scorecards", "source": "Golden quality scorecards and deploy artifacts", "sha256": digest.hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
