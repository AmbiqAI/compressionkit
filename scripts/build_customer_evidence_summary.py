"""Build a customer-facing evidence summary from golden scorecards.

The generated page is intentionally compact: it answers the release questions a
customer is likely to ask before diving into modality-specific scorecards.

Usage:
    uv run python scripts/build_customer_evidence_summary.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RUNS = {
    "ppg": [f"ppg_rvq_64hz_{cr}_golden" for cr in ("02x", "04x", "08x", "16x", "32x")],
    "ecg": [f"ecg_rvq_256hz_{cr}_golden" for cr in ("02x", "04x", "08x", "16x", "32x", "64x")],
}

BAND_KEYS = {
    "ppg": ("band_0.5_3_rel_error", "Pulse-band error"),
    "ecg": ("band_5_15_rel_error", "QRS-band error"),
}


@dataclass(frozen=True)
class EvidenceRow:
    modality: str
    cr_label: str
    run_name: str
    sample_rate_hz: float | None
    frame_ms: float | None
    codec_cr: float | None
    effective_cr: float | None
    n_samples: int | None
    truth_prd_clean: float | None
    faithful_prd: float | None
    prdn_noise: float | None
    hr_mae_bpm: float | None
    band_error: float | None
    coherence: float | None
    seam_ratio: float | None
    encoder_kib: float | None
    decoder_kib: float | None
    codebook_kib: float | None
    edge_payload_kib: float | None
    codebook_entries: int | None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(data: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
        if data is None:
            return None
    return data


def _mean(data: Any, *keys: str) -> float | None:
    value = _safe_get(data, *keys)
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | int | None, precision: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{precision}f}"


def _fmt_size(kib: float | None) -> str:
    if kib is None:
        return "-"
    if kib >= 1024:
        return f"{kib / 1024:.2f} MiB"
    return f"{kib:.0f} KiB"


def _cr_label(run_name: str, fallback: float | None) -> str:
    for part in run_name.split("_"):
        if part.endswith("x") and part[:-1].isdigit():
            return f"{int(part[:-1])}x"
    if fallback is not None:
        return f"{int(fallback)}x"
    return run_name


def _artifact_size_kib(deploy_dir: Path, manifest: dict[str, Any] | None, key: str) -> float | None:
    artifact = _safe_get(manifest, "artifacts", key)
    if not artifact:
        return None
    path = deploy_dir / str(artifact)
    if not path.exists():
        return None
    return path.stat().st_size / 1024


def _edge_payload_kib(*parts: float | None) -> float | None:
    values = [part for part in parts if part is not None]
    if not values:
        return None
    return sum(values)


def _frame_ms(manifest: dict[str, Any] | None, sample_rate_hz: float | None) -> float | None:
    if sample_rate_hz is None:
        return None
    input_shape = _safe_get(manifest, "encoder", "input_shape")
    if not isinstance(input_shape, list):
        return None
    numeric_dims = [dim for dim in input_shape if isinstance(dim, int)]
    if not numeric_dims:
        return None
    frame_samples = max(numeric_dims)
    return 1000.0 * frame_samples / sample_rate_hz


def _physiology(scorecard: dict[str, Any]) -> dict[str, Any]:
    phys = scorecard.get("physiology") or {}
    if isinstance(phys, dict) and isinstance(phys.get("vs_raw_original"), dict):
        return phys["vs_raw_original"]
    return phys if isinstance(phys, dict) else {}


def _seam_ratio(scorecard: dict[str, Any]) -> float | None:
    stability = scorecard.get("stability")
    if isinstance(stability, dict):
        preferred = (
            stability.get("tukey_overlap_add") or stability.get("linear_crossfade") or stability.get("overlap_add")
        )
        value = _safe_get(preferred, "seam_ratio_mean")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return _mean(scorecard, "long_recording", "seam_ratio")


def build_row(results_dir: Path, run_name: str, modality: str) -> EvidenceRow | None:
    run_dir = results_dir / run_name
    scorecard = _load_json(run_dir / "quality_scorecard.json")
    if scorecard is None:
        return None
    deploy_dir = run_dir / "deploy"
    manifest = _load_json(deploy_dir / "deploy_manifest.json")
    physiology = _physiology(scorecard)
    band_key, _ = BAND_KEYS[modality]
    codec_cr = _mean(scorecard, "bitrate", "cr_codec_uniform") or _mean(scorecard, "bitrate", "codec_compression_ratio")
    sample_rate_hz = scorecard.get("sample_rate")
    try:
        sample_rate_hz = None if sample_rate_hz is None else float(sample_rate_hz)
    except (TypeError, ValueError):
        sample_rate_hz = None
    codebook = _safe_get(manifest, "codebook") or {}
    codebook_entries = None
    if isinstance(codebook, dict):
        levels = codebook.get("num_levels")
        embeddings = codebook.get("num_embeddings")
        try:
            codebook_entries = int(levels) * int(embeddings)
        except (TypeError, ValueError):
            codebook_entries = None
    encoder_kib = _artifact_size_kib(deploy_dir, manifest, "encoder_tflite")
    decoder_kib = _artifact_size_kib(deploy_dir, manifest, "decoder_int8_tflite")
    codebook_kib = _artifact_size_kib(deploy_dir, manifest, "codebook_npz")
    return EvidenceRow(
        modality=modality.upper(),
        cr_label=_cr_label(run_name, codec_cr),
        run_name=run_name,
        sample_rate_hz=sample_rate_hz,
        frame_ms=_frame_ms(manifest, sample_rate_hz),
        codec_cr=codec_cr,
        effective_cr=_mean(scorecard, "bitrate", "cr_codec_learned"),
        n_samples=scorecard.get("num_samples"),
        truth_prd_clean=_mean(scorecard, "headline", "truth_prd_vs_clean_pct"),
        faithful_prd=_mean(scorecard, "time_domain", "prd_percent"),
        prdn_noise=_mean(scorecard, "time_domain", "prdn_noise_percent"),
        hr_mae_bpm=_mean(physiology, "hr_mae_bpm"),
        band_error=_mean(scorecard, "spectral", "per_band_rel_error", band_key),
        coherence=_mean(scorecard, "spectral", "coherence"),
        seam_ratio=_seam_ratio(scorecard),
        encoder_kib=encoder_kib,
        decoder_kib=decoder_kib,
        codebook_kib=codebook_kib,
        edge_payload_kib=_edge_payload_kib(encoder_kib, decoder_kib, codebook_kib),
        codebook_entries=codebook_entries,
    )


def _recommendation(modality: str) -> str:
    if modality == "PPG":
        return "2x-8x for tight HR/HRV preservation; 16x-32x when storage or radio budget dominates."
    return "4x-16x for morphology-focused use; 32x-64x for bandwidth-limited capture paths."


def render_overview(rows: list[EvidenceRow]) -> str:
    by_modality: dict[str, list[EvidenceRow]] = {}
    for row in rows:
        by_modality.setdefault(row.modality, []).append(row)
    lines = [
        "| Signal | Published CR range | Recommended operating read | Evidence surfaces |",
        "|---|---:|---|---|",
    ]
    for modality in ("PPG", "ECG"):
        modality_rows = by_modality.get(modality, [])
        if not modality_rows:
            continue
        cr_range = f"{modality_rows[0].cr_label}-{modality_rows[-1].cr_label}"
        surfaces = "truth PRD, PRDN-noise, HR/peak timing, band error, stitching, deploy footprint"
        lines.append(f"| {modality} | {cr_range} | {_recommendation(modality)} | {surfaces} |")
    return "\n".join(lines)


def render_detail_table(rows: list[EvidenceRow], modality: str) -> str:
    _, band_label = BAND_KEYS[modality.lower()]
    lines = [
        f"### {modality} release ladder",
        "",
        "| CR | Frame | N | Truth PRD% | Faithful PRD% | PRDN-noise% | HR MAE bpm | "
        f"{band_label} | Coherence | Seam ratio | Edge payload | Encoder | Decoder | Codebook |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        frame = f"{_fmt(row.frame_ms, 0)} ms" if row.frame_ms is not None else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    row.cr_label,
                    frame,
                    str(row.n_samples) if row.n_samples is not None else "-",
                    _fmt(row.truth_prd_clean),
                    _fmt(row.faithful_prd),
                    _fmt(row.prdn_noise),
                    _fmt(row.hr_mae_bpm),
                    _fmt(row.band_error, 4),
                    _fmt(row.coherence, 4),
                    _fmt(row.seam_ratio, 3),
                    _fmt_size(row.edge_payload_kib),
                    _fmt_size(row.encoder_kib),
                    _fmt_size(row.decoder_kib),
                    _fmt_size(row.codebook_kib),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_document(rows: list[EvidenceRow]) -> str:
    by_modality: dict[str, list[EvidenceRow]] = {}
    for row in rows:
        by_modality.setdefault(row.modality, []).append(row)
    sections = [
        "---",
        "icon: lucide/clipboard-check",
        "---",
        "# Customer Evidence Summary",
        "",
        "This generated page summarizes the v1 release evidence in the terms a product or engineering team usually needs before selecting an operating point. It intentionally combines compression level, fidelity, physiological utility, stitching behavior, and deploy footprint instead of leading with one waveform metric.",
        "",
        "## Release decision overview",
        "",
        render_overview(rows),
        "",
        "## Compression and deploy ladder",
        "",
        "The tables below are generated from local golden `quality_scorecard.json` files and deploy manifests. Truth PRD is measured against the clean reference when available; faithful PRD is measured against the recorded input; PRDN-noise tracks residual noise-normalized distortion. Edge payload is the encoder TFLite, decoder TFLite, and codebook NPZ size, excluding validation samples and documentation files.",
    ]
    for modality in ("PPG", "ECG"):
        modality_rows = by_modality.get(modality, [])
        if modality_rows:
            sections.extend(["", render_detail_table(modality_rows, modality)])
    sections.extend(
        [
            "",
            "## How to use this page",
            "",
            "- Start with the CR ladder to identify feasible compression levels for memory, radio, or storage targets.",
            "- Use Truth PRD and PRDN-noise together when judging noisy recordings; faithful PRD alone can penalize codecs that suppress artifact energy.",
            "- Use HR MAE and band error as utility checks before moving to waveform examples and full scorecards.",
            "- Use seam ratio as the first-pass long-recording stitching check; model pages provide the visual and method-specific context.",
            "- Use edge-payload columns for target budgeting; parameter and MAC summaries can be added when those are reliably exported for every bundle.",
            "",
            "Detailed modality pages remain the source for waveform examples, robustness plots, and reproduction commands: [PPG models](https://ambiqai.github.io/compressionkit/models/ppg/), [ECG models](https://ambiqai.github.io/compressionkit/models/ecg/), [PPG CR vs fidelity](https://ambiqai.github.io/compressionkit/methods/cr_vs_fidelity_ppg/), and [ECG CR vs fidelity](https://ambiqai.github.io/compressionkit/methods/cr_vs_fidelity_ecg/).",
            "",
        ]
    )
    return "\n".join(sections)


def collect_rows(results_dir: Path, runs: dict[str, list[str]]) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    missing: list[str] = []
    for modality, run_names in runs.items():
        for run_name in run_names:
            row = build_row(results_dir, run_name, modality)
            if row is None:
                missing.append(run_name)
            else:
                rows.append(row)
    if missing:
        print("Skipped missing scorecards: " + ", ".join(missing))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/reports/customer-evidence.md"))
    args = parser.parse_args()

    rows = collect_rows(args.results_dir, DEFAULT_RUNS)
    if not rows:
        raise SystemExit(f"No scorecards found under {args.results_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_document(rows), encoding="utf-8")
    print(f"Wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
