"""Render the per-experiment docs tree from ``GOLDEN_REGISTRY`` (#28).

Run from the repo root::

    uv run python scripts/render_experiment_docs.py

Writes ``docs/experiments/index.md`` plus one page per registered
:class:`~compressionkit.experiments.registry.GoldenExperiment`. Pages
are idempotent so re-running after a registry edit refreshes them
in-place.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

from compressionkit.experiments.registry import (
    GOLDEN_REGISTRY,
    GoldenExperiment,
    list_two_stage_children,
)

_DATASET_BLURBS: dict[str, dict[str, str]] = {
    "ptb-xl": {
        "name": "PTB-XL",
        "license": "CC BY 4.0 (open)",
        "source": "https://physionet.org/content/ptb-xl/",
        "notes": "Auto-downloaded on first use.",
    },
    "mesa": {
        "name": "MESA (NSRR)",
        "license": "NSRR Data Use Agreement (restricted)",
        "source": "https://sleepdata.org/datasets/mesa",
        "notes": "Requires an NSRR token. Set ``NSRR_TOKEN`` and call ``MesaDataset(...).download()``.",
    },
    "ppg-unified-strict-sanitize-v1": {
        "name": "Open unified PPG v1",
        "license": "Open (BIDMC, BUT PPG, PPG-DaLiA, WESAD — mixed open licenses, no restricted-access dependency)",
        "source": "../datasets.md",
        "notes": (
            "Sources: BIDMC, BUT PPG, PPG-DaLiA, and WESAD. Published v1 PPG goldens are "
            "MESA-free. Build the cache with `scripts/build_ppg_cache.py`."
        ),
    },
}


def _render_index() -> str:
    by_modality: dict[str, list[GoldenExperiment]] = defaultdict(list)
    for exp in GOLDEN_REGISTRY:
        by_modality[exp.modality].append(exp)

    def _row(exp: GoldenExperiment) -> str:
        page = f"[`{exp.experiment_id}`]({exp.experiment_id}.md)"
        structure = exp.structure
        parent = exp.parent or "—"
        return (
            f"| {page} | {exp.modality.upper()} | {structure} | {exp.compression_ratio}× | "
            f"{parent} | `{exp.dataset_id}` | [`{exp.hf_repo_id}`](https://huggingface.co/{exp.hf_repo_id}) |"
        )

    lines = [
        "---",
        "icon: lucide/flask-conical",
        "---",
        "",
        "# Golden Experiments",
        "",
        "A **golden experiment** is a release-grade run that ships with compressionKIT for v1.",
        "Each entry is fully declarative: a YAML config under `configs/`, a registered training",
        "recipe, and a fixed dataset + HuggingFace repo target. The lifecycle runner",
        "(`compressionkit golden run <id>`) reproduces it end-to-end from a clean checkout.",
        "",
        "Two families exist:",
        "",
        "- **`codec`** — single-stage RVQ autoencoder (encoder → RVQ → decoder).",
        "- **`two_stage`** — paired entropy prior on top of a parent codec. Trains a small",
        "  causal-transformer prior over the codec's token stream and bundles `prior_int8.tflite`",
        "  alongside the codec artifacts in the same HuggingFace repo.",
        "",
        "## v1 Registry",
        "",
        "| Experiment | Modality | Structure | CR | Parent | Dataset | HuggingFace |",
        "|------------|----------|--------|----|--------|---------|-------------|",
    ]
    for modality in ("ppg", "ecg"):
        for exp in sorted(by_modality[modality], key=lambda e: (e.structure != "codec", e.compression_ratio)):
            lines.append(_row(exp))

    lines += [
        "",
        "## Reproduce one experiment",
        "",
        "```bash",
        "# 1. Fetch the dataset (MESA requires NSRR_TOKEN; PTB-XL is open).",
        "uv run compressionkit golden run ppg-rvq-4x --skip-dataset-check  # smoke",
        "",
        "# 2. Real run, publish to HuggingFace (set HF_TOKEN first).",
        "uv run compressionkit golden run ppg-rvq-4x --publish",
        "```",
        "",
        "## Reproduce every experiment in a modality",
        "",
        "```bash",
        "uv run compressionkit golden run-all --modality ppg",
        "uv run compressionkit golden run-all --modality ecg",
        "```",
        "",
        "See also:",
        "",
        "- [HuggingFace testing guide](../huggingface.md) — load any AmbiqAI model in five minutes.",
        "- [Deployment guide](../deployment.md) — exporting the artifacts to an Ambiq-class device.",
        "- [Methods · RVQ Autoencoder](../methods/rvq.md) — the architecture every entry uses.",
        "",
    ]
    return "\n".join(lines)


def _render_experiment(exp: GoldenExperiment) -> str:
    ds = _DATASET_BLURBS.get(exp.dataset_id, {"name": exp.dataset_id, "license": "—", "source": "", "notes": ""})
    icon = "lucide/heart-pulse" if exp.modality == "ppg" else "lucide/activity"
    children = list_two_stage_children(exp.experiment_id) if exp.structure == "codec" else []
    parent_block = ""
    if exp.structure == "two_stage":
        parent_block = dedent(
            f"""
            ## Parent codec

            This entry is the entropy-prior stage paired with [`{exp.parent}`]({exp.parent}.md).
            Codec and prior artifacts publish to the same HuggingFace repo
            ([`{exp.hf_repo_id}`](https://huggingface.co/{exp.hf_repo_id})).
            """
        ).strip()
    two_stage_block = ""
    if children:
        ids = ", ".join(f"[`{c.experiment_id}`]({c.experiment_id}.md)" for c in children)
        two_stage_block = dedent(
            f"""
            ## Two-Stage Variant

            Paired entropy prior: {ids}. Run via the lifecycle runner to chain codec → prior
            and bundle `prior_int8.tflite` into this experiment's `deploy/`.
            """
        ).strip()

    lines = [
        "---",
        f"icon: {icon}",
        "---",
        "",
        f"# `{exp.experiment_id}`",
        "",
        "## Overview",
        "",
        f"- **Modality**: {exp.modality.upper()}",
        f"- **Structure**: `{exp.structure}`",
        f"- **Compression ratio**: {exp.compression_ratio}×",
        f"- **Sample rate**: {exp.sample_rate} Hz",
    ]
    if exp.recipe is not None:
        lines.append(f"- **Recipe**: `{exp.recipe}`")
    if exp.config_path is not None:
        lines.append(
            f"- **Config**: [`{exp.config_path}`]"
            f"(https://github.com/AmbiqAI/compressionkit/blob/main/{exp.config_path})"
        )
    else:
        lines.append("- **Config**: — (operating point is fully declared in the registry; no training config)")
    # All three families (RVQ, SPIHT, hybrid) publish under the registry's
    # own hf_repo_id, which already includes the release-track suffix
    # (see GoldenExperiment.hf_version).
    hf_repo_id = exp.hf_repo_id
    lines += [
        f"- **Run name**: `{exp.run_name}`",
        f"- **HuggingFace**: [`{hf_repo_id}`](https://huggingface.co/{hf_repo_id})",
        "",
        "## Dataset & License",
        "",
        f"- **Dataset**: [{ds['name']}]({ds['source']}) (`dataset_id: {exp.dataset_id}`)",
        f"- **License**: {ds['license']}",
    ]
    if ds["notes"]:
        lines.append(f"- **Notes**: {ds['notes']}")
    lines += [
        "",
        "The lifecycle runner pre-flights dataset availability before training (see #26 and the",
        "[dataset contract](../api/datasets.md)).",
        "",
        "## Reproduction",
        "",
        "```bash",
        "# Single command, end-to-end.",
        f"uv run compressionkit golden run {exp.experiment_id}",
        "",
        "# Publish the deploy package to HuggingFace (requires HF_TOKEN).",
        f"uv run compressionkit golden run {exp.experiment_id} --publish",
        "```",
        "",
        f"Results land under `results/{exp.run_name}/`; deploy artifacts under `results/{exp.run_name}/deploy/`.",
        "",
    ]
    if parent_block:
        lines.append(parent_block + "\n")
    if two_stage_block:
        lines.append(two_stage_block + "\n")
    lines += [
        "## Evaluation Metrics",
        "",
        "See the modality model zoo for the full metrics table:",
        "",
        f"- [{exp.modality.upper()} models]({{path}})".replace("{path}", f"../models/{exp.modality}.md"),
        "",
        "Each run writes `quality_scorecard.json` and `summary.json` under its `results/<run>/`.",
        "",
        "## Deploy Artifacts",
        "",
        "Every successful run produces the canonical edge deploy package:",
        "",
    ]
    if exp.method == "rvq":
        lines += [
            "- `encoder.tflite` / `encoder.h` — INT8 encoder.",
            "- `encoder.keras` — float32 Python reference encoder.",
            "- `decoder.tflite` / `decoder.h` — decoder (float32 + optional INT8).",
            "- `decoder.keras` — float32 Python reference decoder.",
            "- `codebook.npz` / `codebook.h` — RVQ codebook tables.",
            "- `sample_data.npz` — license-safe input/target/reconstruction reference frames "
            "(published to HuggingFace as `sample_stimulus.npz`).",
            "- `model_card.json`, `deploy_manifest.json` — metadata.",
        ]
    else:
        lines += [
            "- `spiht_config.json` / `spiht_app_config.h` — codec parameters (language-neutral + C header).",
            "- `c_sources/spiht.[ch]` — portable C99 SPIHT reference.",
            "- `sample_stimulus.npz` / `reference_vectors.npz` — license-safe test frames and known-good encode/decode vectors.",
            "- `model_card.json`, `deploy_manifest.json` — metadata.",
        ]
        if exp.method == "hybrid":
            lines += [
                "- `denoiser_gain_model.tflite` / `.h` — INT8 wavelet-gain denoiser (embeddable, LiteRT).",
                "- `denoiser_gain_model.keras`, `hybrid_manifest.json` — float32 Python reference denoiser and pipeline stage order.",
            ]
    if exp.structure == "two_stage" or children:
        lines += [
            "- `prior_int8.tflite` / `prior_int8.h` / `prior_manifest.json` — entropy prior (two-stage only).",
        ]
    lines += [
        "",
        "## Customization Notes",
        "",
    ]
    if exp.config_path is not None:
        lines.append("- Tweak the YAML to explore neighbouring operating points; copy the file before editing.")
        lines.append("- For new recipes, prefer the `compressionkit/recipes/` package recipes as a starting point.")
    else:
        lines.append(
            "- This operating point is declared directly in `compressionkit/experiments/registry.py` "
            "(no training YAML) \u2014 add a new registry entry to explore a neighbouring operating point."
        )
    lines += [
        f"- To resume publishing without retraining, pass `--skip-train` to `compressionkit golden run {exp.experiment_id} --publish`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/experiments"))
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "index.md").write_text(_render_index())
    for exp in GOLDEN_REGISTRY:
        (out_dir / f"{exp.experiment_id}.md").write_text(_render_experiment(exp))

    print(f"Wrote {1 + len(GOLDEN_REGISTRY)} pages under {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
