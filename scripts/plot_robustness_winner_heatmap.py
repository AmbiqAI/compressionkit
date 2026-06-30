#!/usr/bin/env python3
"""Render winner heatmaps from the scorecard robustness blocks.

This is the robustness counterpart to ``plot_codec_winner_heatmap.py``. It reads
``quality_scorecard.json`` for every golden in a modality, extracts the merged
``robustness`` block, and produces:

* an empirical-SNR winner heatmap across ``clean`` / ``native`` / SNR ladder
* an additive-artifact winner heatmap across artifact families (mid severity)
* compact CSV tables with the winning lane and winning PRD for each cell

The intent is to make the wearable-noise story visible at a glance: DSP / AI on
clean inputs, Hybrid on native and heavy-noise inputs, with modality-specific
behavior in the high-CR corner.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from compressionkit.experiments.registry import list_goldens

LANES = [
    ("spiht", "SPIHT", "S", "#d9d9d9"),
    ("hybrid", "Hybrid", "H", "#4c956c"),
    ("rvq", "RVQ", "R", "#2f6690"),
]
LANE_ORDER = [k for k, _l, _s, _c in LANES]
LANE_LABEL = {k: l for k, l, _s, _c in LANES}
LANE_SHORT = {k: s for k, _l, s, _c in LANES}
LANE_COLOR = {k: c for k, _l, _s, c in LANES}
WHITE_TEXT = {"hybrid", "rvq"}


def _load_scorecard(path: Path) -> dict:
    return json.loads(path.read_text())


def _lane_runs(modality: str, results_dir: Path) -> dict[int, dict[str, Path]]:
    out: dict[int, dict[str, Path]] = {}
    for exp in list_goldens(modality=modality):
        if exp.family != "codec":
            continue
        out.setdefault(exp.compression_ratio, {})[exp.method] = results_dir / exp.run_name / "quality_scorecard.json"
    return out


def _snr_rows(scorecard: dict) -> tuple[list[str], dict[str, float]]:
    rob = scorecard.get("robustness", {})
    labels = ["clean", "native"]
    values = {
        "clean": float(rob.get("reference", {}).get("clean", float("nan"))),
        "native": float(rob.get("reference", {}).get("native", float("nan"))),
    }
    for row in rob.get("empirical_snr_curve", []):
        snr = row.get("snr_db")
        if snr is None:
            continue
        label = f"{snr:g}dB"
        labels.append(label)
        values[label] = float(row.get("prd_mean", float("nan")))
    return labels, values


def _artifact_rows(scorecard: dict) -> tuple[list[str], dict[str, float]]:
    rob = scorecard.get("robustness", {})
    out: dict[str, float] = {}
    for family, rows in rob.get("artifacts", {}).items():
        if not rows:
            continue
        mid = min(rows, key=lambda r: abs(float(r.get("severity", 0.0)) - 0.5))
        out[family] = float(mid.get("prd_mean", float("nan")))
    labels = sorted(out)
    return labels, out


def _winner_grid(crs: list[int], columns: list[str], per_lane: dict[int, dict[str, dict[str, float]]]) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    idx = np.full((len(crs), len(columns)), np.nan)
    val = np.full((len(crs), len(columns)), np.nan)
    winners: list[list[str]] = [["" for _ in columns] for _ in crs]
    for r, cr in enumerate(crs):
        lane_map = per_lane[cr]
        for c, col in enumerate(columns):
            best_lane = None
            best_value = float("inf")
            for lane in LANE_ORDER:
                lane_vals = lane_map.get(lane, {})
                v = lane_vals.get(col, float("nan"))
                if np.isfinite(v) and v < best_value:
                    best_lane, best_value = lane, v
            if best_lane is None:
                continue
            idx[r, c] = LANE_ORDER.index(best_lane)
            val[r, c] = best_value
            winners[r][c] = best_lane
    return idx, val, winners


def _plot_heatmap(grid: np.ndarray, vals: np.ndarray, crs: list[int], columns: list[str], out_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(1.0 * len(columns) + 2.8, 0.7 * len(crs) + 2.8))
    cmap = ListedColormap([LANE_COLOR[k] for k in LANE_ORDER])
    masked = np.ma.masked_invalid(grid)
    ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=len(LANE_ORDER) - 1)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(crs)))
    ax.set_yticklabels([f"{cr}x" for cr in crs])
    ax.set_xlabel("Condition")
    ax.set_ylabel("Compression ratio")
    ax.set_title(title, fontsize=10)
    for r in range(len(crs)):
        for c in range(len(columns)):
            if np.isnan(grid[r, c]):
                continue
            lane = LANE_ORDER[int(grid[r, c])]
            text_color = "white" if lane in WHITE_TEXT else "black"
            ax.text(c, r, f"{LANE_SHORT[lane]}\n{vals[r, c]:.1f}", ha="center", va="center", fontsize=7, color=text_color)
    legend = [Patch(facecolor=LANE_COLOR[k], edgecolor="none", label=LANE_LABEL[k]) for k in LANE_ORDER]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _write_csv(path: Path, crs: list[int], columns: list[str], winners: list[list[str]], values: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cr"] + columns)
        for r, cr in enumerate(crs):
            row = [f"{cr}x"]
            for c, _ in enumerate(columns):
                if winners[r][c]:
                    row.append(f"{winners[r][c]}:{values[r, c]:.4f}")
                else:
                    row.append("")
            writer.writerow(row)
    print(f"Wrote {path}")


def _build_metric_map(modality: str, results_dir: Path) -> tuple[list[int], dict[int, dict[str, dict[str, float]]], dict[int, dict[str, dict[str, float]]], list[str], list[str]]:
    run_map = _lane_runs(modality, results_dir)
    crs = sorted(run_map)
    snr_map: dict[int, dict[str, dict[str, float]]] = {}
    art_map: dict[int, dict[str, dict[str, float]]] = {}
    snr_cols: list[str] | None = None
    art_cols: set[str] = set()
    for cr in crs:
        snr_map[cr] = {}
        art_map[cr] = {}
        for lane, sc_path in run_map[cr].items():
            if not sc_path.exists():
                continue
            scorecard = _load_scorecard(sc_path)
            cols, vals = _snr_rows(scorecard)
            snr_map[cr][lane] = vals
            if snr_cols is None:
                snr_cols = cols
            acols, avals = _artifact_rows(scorecard)
            art_map[cr][lane] = avals
            art_cols.update(acols)
    return crs, snr_map, art_map, (snr_cols or []), sorted(art_cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=["ecg", "ppg"], required=True)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--output-stem", type=str, default=None)
    args = ap.parse_args()

    crs, snr_map, art_map, snr_cols, art_cols = _build_metric_map(args.modality, args.results_dir)
    if not crs:
        raise SystemExit(f"No codec scorecards found for modality={args.modality!r}")
    stem = args.output_stem or f"{args.modality}_robustness_winners"
    out_base = args.results_dir / stem

    snr_grid, snr_vals, snr_winners = _winner_grid(crs, snr_cols, snr_map)
    _plot_heatmap(
        snr_grid,
        snr_vals,
        crs,
        snr_cols,
        out_base.with_name(out_base.name + "_snr.png"),
        title=f"{args.modality.upper()} robustness winners: clean / native / empirical SNR",
    )
    _write_csv(out_base.with_name(out_base.name + "_snr.csv"), crs, snr_cols, snr_winners, snr_vals)

    if art_cols:
        art_grid, art_vals, art_winners = _winner_grid(crs, art_cols, art_map)
        _plot_heatmap(
            art_grid,
            art_vals,
            crs,
            art_cols,
            out_base.with_name(out_base.name + "_artifacts.png"),
            title=f"{args.modality.upper()} robustness winners: additive artifacts (mid severity)",
        )
        _write_csv(out_base.with_name(out_base.name + "_artifacts.csv"), crs, art_cols, art_winners, art_vals)


if __name__ == "__main__":
    main()
