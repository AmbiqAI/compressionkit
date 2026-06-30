"""Batch-evaluate all primary golden runs and produce a manifest.

Iterates over the naming-convention golden runs:
    results/{modality}_rvq_{sample_rate}hz_{cr}x_golden/

For each, runs the full eval harness (fidelity, adversarial, stitching, qos)
via ``scripts/eval_codec.py``, then collects top-line metrics into a single
``results/golden_eval_manifest.json`` that summarises the fleet.

The individual ``report.json`` + ``report.md`` files live under each golden
run's directory (git-ignored); only the manifest is small enough to commit
and acts as the "evaluation badge" for the model fleet.

Usage::

    source .venv/bin/activate
    python scripts/generate_golden_scorecards.py

The script is idempotent — re-running overwrites existing reports.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

# Load eval_codec.py as a module so we can call its run() programmatically.
_SCRIPTS = Path(__file__).resolve().parent
_CLI_PATH = _SCRIPTS / "eval_codec.py"
_spec = importlib.util.spec_from_file_location("eval_codec", _CLI_PATH)
assert _spec is not None and _spec.loader is not None
_cli = importlib.util.module_from_spec(_spec)
sys.modules["eval_codec"] = _cli
_spec.loader.exec_module(_cli)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

# Primary golden runs per the AGENTS.md naming convention.
GOLDEN_RUNS: list[dict[str, str | float]] = [
    {"modality": "ppg", "cr": 2, "dir": "ppg_rvq_64hz_02x_golden"},
    {"modality": "ppg", "cr": 4, "dir": "ppg_rvq_64hz_04x_golden"},
    {"modality": "ppg", "cr": 8, "dir": "ppg_rvq_64hz_08x_golden"},
    {"modality": "ppg", "cr": 16, "dir": "ppg_rvq_64hz_16x_golden"},
    {"modality": "ppg", "cr": 32, "dir": "ppg_rvq_64hz_32x_golden"},
    {"modality": "ecg", "cr": 2, "dir": "ecg_rvq_256hz_02x_golden"},
    {"modality": "ecg", "cr": 4, "dir": "ecg_rvq_256hz_04x_golden"},
    {"modality": "ecg", "cr": 8, "dir": "ecg_rvq_256hz_08x_golden_empirical_midpoint"},
    {"modality": "ecg", "cr": 16, "dir": "ecg_rvq_256hz_16x_golden"},
    {"modality": "ecg", "cr": 32, "dir": "ecg_rvq_256hz_32x_golden"},
    {"modality": "ecg", "cr": 64, "dir": "ecg_rvq_256hz_64x_golden"},
]

TIERS = ["fidelity", "adversarial", "stitching", "qos"]


def _extract_topline(report: dict) -> dict:
    """Pull the customer-facing numbers from a full report."""
    fid = report.get("tiers", {}).get("fidelity", {}).get("metrics", {})
    adv = report.get("tiers", {}).get("adversarial", {})
    qos = report.get("tiers", {}).get("qos", {})

    topline: dict = {
        "codec": report["codec"]["name"],
        "modality": report["codec"]["modality"],
        "cr": report["codec"]["target_cr"],
        "frame_size": report["codec"]["frame_size"],
        "sample_rate": report["codec"]["sample_rate"],
    }

    # Fidelity
    if fid:
        topline["prd_percent_mean"] = fid.get("prd_percent", {}).get("mean")
        topline["prd_percent_p90"] = fid.get("prd_percent", {}).get("p90")
        topline["cosine_similarity_mean"] = fid.get("cosine_similarity", {}).get("mean")

    # Adversarial — zero-input hallucination
    zero = adv.get("zero_input", {})
    if zero:
        topline["zero_input_l2"] = zero.get("output_l2_when_input_zero")
        topline["zero_input_peaks"] = zero.get("hallucinated_peaks")

    # QoS
    if qos and not qos.get("skipped"):
        conf = qos.get("confidence", {})
        topline["qos_confidence_mean"] = conf.get("mean")
        topline["qos_confidence_p10"] = conf.get("p10")

    return topline


def main() -> int:
    manifest: list[dict] = []
    t0 = time.time()
    n_ok, n_fail = 0, 0

    for golden in GOLDEN_RUNS:
        run_dir = RESULTS_DIR / golden["dir"]
        if not (run_dir / "best_model.weights.h5").exists():
            print(f"SKIP {golden['dir']} (missing weights)")
            continue

        out_dir = run_dir / "eval"
        print(f"\n{'='*60}")
        print(f"Evaluating: {golden['dir']}")
        print(f"{'='*60}")

        try:
            args = _cli.build_parser().parse_args(
                [
                    "--codec", "rvq",
                    "--modality", str(golden["modality"]),
                    "--rvq-run", str(run_dir),
                    "--tiers", *TIERS,
                    "--n-frames", "16",
                    "--seed", "42",
                    "--out", str(out_dir),
                ]
            )
            report = _cli.run(args)
            topline = _extract_topline(report)
            topline["status"] = "ok"
            topline["elapsed_s"] = sum(report.get("elapsed_s", {}).values())
            manifest.append(topline)
            n_ok += 1
            print(f"  PRD={topline.get('prd_percent_mean', '?'):.2f}%  "
                  f"zero_l2={topline.get('zero_input_l2', '?')}  "
                  f"qos={topline.get('qos_confidence_mean', '?')}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            manifest.append({
                "codec": golden["dir"],
                "modality": golden["modality"],
                "cr": golden["cr"],
                "status": "error",
                "error": str(exc),
            })
            n_fail += 1

    elapsed = time.time() - t0
    manifest_path = RESULTS_DIR / "golden_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n{'='*60}")
    print(f"Done: {n_ok} ok, {n_fail} failed in {elapsed:.1f}s")
    print(f"Manifest: {manifest_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
