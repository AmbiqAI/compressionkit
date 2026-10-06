"""Export complete fidelity tables without rewriting documentation prose."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def number(value: object, precision: int = 2) -> str:
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return "—"
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid fidelity measurement: {value}")
    if precision == 0 and not value.is_integer():
        raise ValueError(f"Sample count must be an integer: {value}")
    return f"{value:.{precision}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("fidelity_export", repo / "scripts/build_cr_vs_fidelity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outputs = {}
    for modality, runs in (("ppg", module.PPG_GOLDEN_RUNS), ("ecg", module.ECG_GOLDEN_RUNS)):
        data = json.loads(Path(f"content-data/fidelity-{modality}.json").read_text())
        rows = {table["id"]: [] for table in data["tables"]}
        digest = hashlib.sha256()
        for run in runs:
            path = args.results_dir / run / "quality_scorecard.json"
            if not path.is_file():
                raise SystemExit(f"Missing {path}; refusing partial fidelity refresh")
            raw = path.read_bytes()
            digest.update(run.encode())
            digest.update(hashlib.sha256(raw).digest())
            score = json.loads(raw)
            item = module.build_headline_row(score, run, modality)
            cr = item["cr_label"]
            rows["headline"].append([cr, number(item["codec_cr"]), number(item["effective_cr"]), number(item["bits_per_token"]), number(item["n"], 0), number(item["prd_percent"])])
            rows["physiology"].append([cr, number(item["truth_prd_clean"]), number(item["prdn_noise_percent"]), number(item["hr_mae_bpm"]), number(item["qrs_band_err"], 4), number(item["coherence"], 4)])
            long_recording = score.get("long_recording") or {}
            seam = long_recording.get("seam_ratio")
            if seam is None:
                seam = (long_recording.get("stitching") or {}).get("seam_ratio")
            rows["seams"].append([cr, number(seam, 3)])
            for bucket in module.build_tertile_rows(score, run, modality):
                keys = [cr, bucket["tertile"]]
                rows["noise"].append(keys + [number(bucket["n"], 0), number(bucket["prd_percent"]), number(bucket["prdn_noise_percent"]), number(bucket["hr_mae_bpm"])])
                rows["noise-spectral"].append(keys + [number(bucket["qrs_band_err"], 4), number(bucket["coherence"], 4)])
        for table in data["tables"]:
            table["rows"] = rows[table["id"]]
        data["provenance"] = {"kind": "scorecards", "source": "Golden quality scorecards", "sha256": digest.hexdigest()}
        outputs[f"fidelity-{modality}.json"] = json.dumps(data, indent=2, allow_nan=False) + "\n"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (args.output_dir / name).write_text(content)


if __name__ == "__main__":
    main()
