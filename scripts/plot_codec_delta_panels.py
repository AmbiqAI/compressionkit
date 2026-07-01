#!/usr/bin/env python3
"""Delta-focused views of the 3-lane codec crossover JSON.

The winner-takes-all heatmap (``plot_codec_winner_heatmap``) shows *who* wins but
hides *by how much*: a 0.02 PRD edge and a 26-point blowout look identical. This
script renders two complementary views that expose the margins:

1. **PRD-vs-SNR line panels** (one panel per CR): three curves (SPIHT / Hybrid /
   RVQ) over the noise ladder, with the real ``native`` column drawn as a distinct
   marker. Crossover points and separation magnitude are read off directly.
2. **Signed pairwise delta heatmaps**: ``Hybrid - SPIHT``, ``RVQ - SPIHT`` and
   ``Hybrid - RVQ`` on a diverging colormap centered at 0 (negative => the first
   lane wins). Annotated with the PRD gap so near-ties vs blowouts are obvious.

Consumes the ``by_cr`` schema written by ``eval_learned_shrink_{ecg,ppg}`` where
``by_cr[crX][lane]`` is a list of PRD floats aligned to top-level ``columns`` and
lanes are ``spiht`` / ``learned`` (Hybrid) / ``rvq``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LANE_STYLE = {
    "spiht": ("SPIHT (DSP)", "#b08900", "o", "-"),
    "hybrid": ("Hybrid (AI+DSP)", "#2a7f5f", "s", "-"),
    "rvq": ("RVQ (AI)", "#2f6690", "^", "-"),
}


def _hybrid_key(block: dict) -> str | None:
    for key in ("hybrid", "learned"):
        if key in block:
            return key
    return None


def _lane_series(block: dict) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    if "spiht" in block:
        out["spiht"] = [float(x) for x in block["spiht"]]
    hk = _hybrid_key(block)
    if hk is not None:
        out["hybrid"] = [float(x) for x in block[hk]]
    if "rvq" in block:
        out["rvq"] = [float(x) for x in block["rvq"]]
    return out


def _split_columns(columns: list[str]) -> tuple[list[int], list[int]]:
    """Return (ladder_indices, special_indices). Special = native (non-SNR)."""
    ladder, special = [], []
    for i, col in enumerate(columns):
        if col.lower() == "native":
            special.append(i)
        else:
            ladder.append(i)
    return ladder, special


def plot_line_panels(
    crs: list[str],
    columns: list[str],
    by_cr: dict[str, dict],
    out_path: Path,
    title: str,
) -> None:
    ladder_idx, special_idx = _split_columns(columns)
    ladder_labels = [columns[i] for i in ladder_idx]
    x = np.arange(len(ladder_idx))

    n = len(crs)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)

    for ax_i, cr in enumerate(crs):
        ax = axes[ax_i // ncol][ax_i % ncol]
        series = _lane_series(by_cr[cr])
        for lane, (label, color, marker, ls) in LANE_STYLE.items():
            if lane not in series:
                continue
            vals = series[lane]
            ax.plot(
                x,
                [vals[i] for i in ladder_idx],
                color=color,
                marker=marker,
                linestyle=ls,
                markersize=4,
                linewidth=1.6,
                label=label,
            )
            # native point as a hollow off-ladder marker at x=-1
            for si in special_idx:
                ax.plot(
                    -1.0,
                    vals[si],
                    color=color,
                    marker=marker,
                    markersize=7,
                    markerfacecolor="none",
                    markeredgewidth=1.6,
                )
        ax.set_title(f"CR {cr}")
        ax.set_xticks(np.concatenate(([-1.0], x)))
        ax.set_xticklabels(["native", *ladder_labels], rotation=45, ha="right", fontsize=7)
        ax.axvline(-0.5, color="0.7", linestyle=":", linewidth=0.8)
        ax.set_ylabel("PRD% vs clean truth")
        ax.grid(True, alpha=0.25)
        if ax_i == 0:
            ax.legend(fontsize=7, loc="upper left")

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _delta_matrix(crs: list[str], columns: list[str], by_cr: dict[str, dict], a: str, b: str) -> np.ndarray:
    """PRD[a] - PRD[b]; negative => lane a wins."""
    mat = np.full((len(crs), len(columns)), np.nan, dtype=float)
    for r, cr in enumerate(crs):
        series = _lane_series(by_cr[cr])
        if a not in series or b not in series:
            continue
        va, vb = series[a], series[b]
        for c in range(len(columns)):
            mat[r, c] = va[c] - vb[c]
    return mat


def plot_delta_heatmaps(
    crs: list[str],
    columns: list[str],
    by_cr: dict[str, dict],
    out_path: Path,
    title: str,
) -> None:
    pairs = [
        ("hybrid", "spiht", "Hybrid - SPIHT"),
        ("rvq", "spiht", "RVQ - SPIHT"),
        ("hybrid", "rvq", "Hybrid - RVQ"),
    ]
    mats = [(_delta_matrix(crs, columns, by_cr, a, b), lbl) for a, b, lbl in pairs]
    vmax = max(np.nanmax(np.abs(m)) for m, _ in mats)

    fig, axes = plt.subplots(1, 3, figsize=(6.0 * 3, 0.55 * len(crs) + 2.2), squeeze=False)
    for k, (mat, lbl) in enumerate(mats):
        ax = axes[0][k]
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(crs)))
        ax.set_yticklabels(crs, fontsize=8)
        ax.set_title(lbl + "\n(blue = first lane wins)", fontsize=9)
        for r in range(len(crs)):
            for c in range(len(columns)):
                if np.isnan(mat[r, c]):
                    continue
                ax.text(
                    c,
                    r,
                    f"{mat[r, c]:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black" if abs(mat[r, c]) < 0.55 * vmax else "white",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ΔPRD%")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path", type=Path, help="Crossover JSON with a 'by_cr' block.")
    ap.add_argument("--output-stem", type=str, default=None, help="Stem beside the JSON.")
    ap.add_argument("--title", type=str, default="3-lane codec deltas")
    args = ap.parse_args()

    data = json.loads(args.json_path.read_text())
    columns: list[str] = data["columns"]
    by_cr: dict[str, dict] = data["by_cr"]
    crs = list(by_cr.keys())

    stem = args.output_stem or (args.json_path.stem + "_deltas")
    base = args.json_path.parent / stem

    plot_line_panels(crs, columns, by_cr, base.with_name(base.name + "_lines.png"), args.title)
    plot_delta_heatmaps(crs, columns, by_cr, base.with_name(base.name + "_heat.png"), args.title)


if __name__ == "__main__":
    main()
