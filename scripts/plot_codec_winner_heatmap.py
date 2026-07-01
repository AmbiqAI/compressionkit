#!/usr/bin/env python3
"""Build a winner table and categorical heatmap from codec comparison JSON.

Handles both result schemas in this repo:

* float-valued blocks (artifact regimes), where ``by_cr[crX][codec]`` is a list
  of PRD floats aligned to ``columns``.
* dict-valued blocks (empirical regimes), where each entry is a dict such as
  ``{"prd_truth": ..., "prd_faithful": ...}``; the metric is selected via
  ``--metric``.

Codec lanes are detected per block. ``learned`` and ``hybrid`` map to the same
"Hybrid" lane, so ECG (filter/learned) and PPG (hybrid) JSONs both work. Columns
that are entirely non-finite (e.g. ``null_frame``) are dropped from the figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

LANE_SPECS: list[tuple[str, str, str, str]] = [
    ("spiht", "SPIHT", "S", "#d9d9d9"),
    ("filter", "Bandpass+SPIHT", "B", "#e9c46a"),
    ("hybrid", "Hybrid (AI+DSP)", "H", "#4c956c"),
    ("rvq", "RVQ", "R", "#2f6690"),
]
LANE_ORDER = [lane for lane, _label, _short, _color in LANE_SPECS]
LANE_LABEL = {lane: label for lane, label, _short, _color in LANE_SPECS}
LANE_SHORT = {lane: short for lane, _label, short, _color in LANE_SPECS}
LANE_COLOR = {lane: color for lane, _label, _short, color in LANE_SPECS}
WHITE_TEXT_LANES = {"hybrid", "rvq"}
DEFAULT_METRIC_KEYS = ["prd_truth", "prd_target", "prd_clean", "prd"]

# Taxonomy: "Bandpass+SPIHT" is the pure-DSP path (bandpass filter + DWT + SPIHT,
# or a classical wavelet shrink). "Hybrid (AI+DSP)" is an AI denoiser fused with
# SPIHT. The ambiguous "hybrid" block key is resolved against the JSON's
# ``hybrid_method`` so filter_spiht/bayes_shrink land in the DSP lane and only a
# learned denoiser lands in the Hybrid lane. The lane retains the ``filter``
# block key for backward compatibility; only its display label changed.
# Note: Bandpass+SPIHT is proxy-favoured wherever its passband matches the
# clean-truth proxy band (see eval_contact_artifact_regime_ecg.py).
DSP_HYBRID_METHODS = {"filter_spiht", "filter", "bayes_shrink_spiht", "dsp"}
AI_HYBRID_METHODS = {"learned_shrink_spiht", "learned_shrink_spiht_v2", "learned", "ai"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path", type=Path, help="Codec comparison JSON with a 'by_cr' block.")
    ap.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Output stem without extension; defaults beside the JSON using '<stem>_winners'",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default=",".join(DEFAULT_METRIC_KEYS),
        help="Comma-separated priority list of dict keys to score on (first finite wins).",
    )
    ap.add_argument("--title", type=str, default=None, help="Override the figure title.")
    return ap.parse_args()


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def _extract(entry: object, metric_keys: list[str]) -> float:
    """Return the scalar PRD for one codec/column entry under either schema."""
    if isinstance(entry, dict):
        for key in metric_keys:
            if key in entry:
                value = _safe_float(entry[key])
                if math.isfinite(value):
                    return value
        return math.nan
    return _safe_float(entry)


def _key_to_lane(data: dict) -> dict[str, str]:
    """Map raw block keys to display lanes, honoring the AI/DSP taxonomy."""
    method = str(data.get("hybrid_method", "")).strip().lower()
    if method in AI_HYBRID_METHODS:
        hybrid_lane = "hybrid"
    elif method in DSP_HYBRID_METHODS:
        hybrid_lane = "filter"
    else:
        hybrid_lane = "hybrid"
    return {"spiht": "spiht", "filter": "filter", "learned": "hybrid", "hybrid": hybrid_lane, "rvq": "rvq"}


def _block_lane_series(block: dict, key_to_lane: dict[str, str]) -> dict[str, list]:
    out: dict[str, list] = {}
    for raw_key, series in block.items():
        lane = key_to_lane.get(raw_key)
        if lane is not None and isinstance(series, list) and lane not in out:
            out[lane] = series
    return out


def _best_two(values: dict[str, float]) -> tuple[str | None, float, float]:
    finite = [(codec, score) for codec, score in values.items() if math.isfinite(score)]
    if not finite:
        return None, math.nan, math.nan
    finite.sort(key=lambda item: item[1])
    winner, best = finite[0]
    margin = finite[1][1] - best if len(finite) > 1 else math.nan
    return winner, best, margin


def _load_rows(
    data: dict, metric_keys: list[str]
) -> tuple[list[int], list[str], list[list[dict[str, object]]], list[str]]:
    crs = [int(cr) for cr in data["crs"]]
    columns = list(data["columns"])
    key_to_lane = _key_to_lane(data)
    rows: list[list[dict[str, object]]] = []
    column_has_data = [False] * len(columns)
    for cr in crs:
        block = data["by_cr"][f"{cr}x"]
        lane_series = _block_lane_series(block, key_to_lane)
        lanes = [lane for lane in LANE_ORDER if lane in lane_series]
        cr_rows: list[dict[str, object]] = []
        for index, label in enumerate(columns):
            values = {
                lane: _extract(lane_series[lane][index] if index < len(lane_series[lane]) else math.nan, metric_keys)
                for lane in lanes
            }
            if any(math.isfinite(v) for v in values.values()):
                column_has_data[index] = True
            winner, best_prd, margin = _best_two(values)
            cr_rows.append(
                {
                    "cr": cr,
                    "label": label,
                    "winner": winner,
                    "best_prd": best_prd,
                    "margin": margin,
                    "values": values,
                }
            )
        rows.append(cr_rows)
    keep = [i for i, has in enumerate(column_has_data) if has]
    columns = [columns[i] for i in keep]
    rows = [[cr_rows[i] for i in keep] for cr_rows in rows]
    lanes_used = sorted(
        {cell["winner"] for cr_rows in rows for cell in cr_rows if cell["winner"]}, key=LANE_ORDER.index
    )
    return crs, columns, rows, lanes_used


def _write_csv(csv_path: Path, rows: list[list[dict[str, object]]]) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cr", "condition", "winner", "best_prd", "margin_to_runner_up", *LANE_ORDER])
        for cr_rows in rows:
            for cell in cr_rows:
                values = cell["values"]
                writer.writerow(
                    [
                        f"{cell['cr']}x",
                        cell["label"],
                        LANE_LABEL.get(cell["winner"], "NA"),
                        _fmt(cell["best_prd"]),
                        _fmt(cell["margin"]),
                        *[_fmt(values.get(lane, math.nan)) for lane in LANE_ORDER],
                    ]
                )


def _write_markdown(md_path: Path, columns: list[str], rows: list[list[dict[str, object]]]) -> None:
    with md_path.open("w") as handle:
        header = ["CR", *columns]
        handle.write("| " + " | ".join(header) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for cr_rows in rows:
            values = [f"{cr_rows[0]['cr']}x"]
            for cell in cr_rows:
                label = LANE_LABEL.get(cell["winner"], "NA")
                values.append(f"{label} ({_fmt(cell['best_prd'])})")
            handle.write("| " + " | ".join(values) + " |\n")


def _plot_heatmap(
    png_path: Path,
    crs: list[int],
    columns: list[str],
    rows: list[list[dict[str, object]]],
    lanes_used: list[str],
    title: str,
) -> None:
    lane_to_idx = {lane: idx for idx, lane in enumerate(LANE_ORDER)}
    grid = np.full((len(crs), len(columns)), np.nan, dtype=np.float32)
    for row_index, cr_rows in enumerate(rows):
        for col_index, cell in enumerate(cr_rows):
            winner = cell["winner"]
            if winner is not None:
                grid[row_index, col_index] = lane_to_idx[winner]

    fig, ax = plt.subplots(figsize=(1.1 * len(columns) + 3.0, 0.8 * len(crs) + 2.8))
    cmap = ListedColormap([LANE_COLOR[lane] for lane in LANE_ORDER])
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(LANE_ORDER) - 0.5, origin="upper")
    im.cmap.set_bad(color="#ffffff")

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(range(len(crs)))
    ax.set_yticklabels([f"{cr}x" for cr in crs])
    ax.set_xlabel("Input condition / SNR / artifact")
    ax.set_ylabel("Compression ratio")
    ax.set_title(title, fontsize=10)

    for row_index, cr_rows in enumerate(rows):
        for col_index, cell in enumerate(cr_rows):
            winner = cell["winner"]
            if winner is None:
                continue
            ax.text(
                col_index,
                row_index,
                f"{LANE_SHORT[winner]}\n{_fmt(cell['best_prd'])}\n+{_fmt(cell['margin'])}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if winner in WHITE_TEXT_LANES else "black",
            )

    legend_lanes = lanes_used or LANE_ORDER
    legend = [Patch(facecolor=LANE_COLOR[lane], label=LANE_LABEL[lane]) for lane in legend_lanes]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, title="Winner")
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


def _fmt(value: object) -> str:
    num = _safe_float(value)
    return "NA" if not math.isfinite(num) else f"{num:.2f}"


def main() -> None:
    args = parse_args()
    metric_keys = [part.strip() for part in args.metric.split(",") if part.strip()] or DEFAULT_METRIC_KEYS
    data = json.loads(args.json_path.read_text())
    crs, columns, rows, lanes_used = _load_rows(data, metric_keys)

    output_dir = args.json_path.parent
    output_stem = args.output_stem or f"{args.json_path.stem}_winners"
    csv_path = output_dir / f"{output_stem}.csv"
    md_path = output_dir / f"{output_stem}.md"
    png_path = output_dir / f"{output_stem}.png"

    title = (
        args.title or "Best codec by PRD (lower is better)\ncell shows winner, PRD, and margin to runner-up"
    ).replace("\\n", "\n")

    _write_csv(csv_path, rows)
    _write_markdown(md_path, columns, rows)
    _plot_heatmap(png_path, crs, columns, rows, lanes_used, title)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
