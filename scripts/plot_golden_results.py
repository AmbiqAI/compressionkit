"""Generate static plots for golden model results.

Usage:
    python scripts/plot_golden_results.py [--input results/golden_summary.csv] [--output docs/assets/plots]

Reads the CSV produced by collect_golden_results.py and generates individual
PNG images — one metric per chart, separate PPG and ECG, light + dark variants.

Output naming: ``{signal}_{metric}_{light|dark}.png``
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Brand palette ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    fg: str
    fg_secondary: str
    grid: str
    ppg: str
    ppg_light: str
    ecg: str
    ecg_light: str


LIGHT = Theme(
    name="light",
    bg="#FFFFFF",
    fg="#1E1E1E",
    fg_secondary="#555555",
    grid="#CCCCCC",
    ppg="#0D7377",
    ppg_light="#4DB6AC",
    ecg="#B45309",
    ecg_light="#D4A76A",
)

DARK = Theme(
    name="dark",
    bg="#1E1E1E",
    fg="#E0E0E0",
    fg_secondary="#9E9E9E",
    grid="#444444",
    ppg="#26A69A",
    ppg_light="#80CBC4",
    ecg="#F9C80E",
    ecg_light="#FDE68A",
)

THEMES = [LIGHT, DARK]

CR_ORDER = ["02x", "04x", "08x", "16x", "32x"]
CR_LABELS = ["2×", "4×", "8×", "16×", "32×"]
MARKERS = {"PPG": "o", "ECG": "s"}
SAMPLE_RATES = {"PPG": 64, "ECG": 256}


def _color(theme: Theme, signal: str) -> str:
    return theme.ppg if signal == "PPG" else theme.ecg


def _color_light(theme: Theme, signal: str) -> str:
    return theme.ppg_light if signal == "PPG" else theme.ecg_light


# ── Theme application ──────────────────────────────────────────────────────────


def _apply_theme(theme: Theme) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "figure.facecolor": theme.bg,
            "axes.facecolor": theme.bg,
            "axes.edgecolor": theme.grid,
            "axes.labelcolor": theme.fg,
            "axes.titlecolor": theme.fg,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.color": theme.grid,
            "grid.alpha": 0.45,
            "grid.linewidth": 0.5,
            "text.color": theme.fg,
            "xtick.color": theme.fg_secondary,
            "ytick.color": theme.fg_secondary,
            "legend.facecolor": theme.bg,
            "legend.edgecolor": theme.grid,
            "legend.labelcolor": theme.fg,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "600",
            "axes.labelsize": 11.5,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
            "savefig.facecolor": theme.bg,
            "savefig.edgecolor": "none",
            "savefig.transparent": False,
        }
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _signal_df(df: pd.DataFrame, signal: str) -> pd.DataFrame:
    return df[df["signal"] == signal].sort_values("compression_ratio")


def _cr_ticks(ax: plt.Axes, sdf: pd.DataFrame) -> None:
    crs = sorted(sdf["compression_ratio"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(crs)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}×" if v == int(v) else f"{v:.1f}×"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Compression Ratio")


def _annotate(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    ax.annotate(
        text, (x, y), textcoords="offset points", xytext=(0, 9), fontsize=8, ha="center", color=color, fontweight="500"
    )


def _save(fig: plt.Figure, output_dir: Path, name: str, theme: Theme) -> Path:
    path = output_dir / f"{name}_{theme.name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ── Plot generators (one signal, one metric each) ─────────────────────────────


def _line_plot(
    sdf: pd.DataFrame,
    signal: str,
    y_col: str,
    y_label: str,
    title: str,
    theme: Theme,
    fmt: str = ".4f",
    fill: bool = False,
) -> plt.Figure:
    """Single-signal line chart over compression ratios."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    c = _color(theme, signal)

    ax.plot(sdf["compression_ratio"], sdf[y_col], color=c, marker=MARKERS[signal], zorder=3)
    if fill:
        ax.fill_between(sdf["compression_ratio"], 0, sdf[y_col], color=c, alpha=0.08)
    for _, row in sdf.iterrows():
        _annotate(ax, row["compression_ratio"], row[y_col], f"{row[y_col]:{fmt}}", c)

    _cr_ticks(ax, sdf)
    ax.set_ylabel(y_label)
    ax.set_title(f"{signal} — {title}")
    fig.tight_layout()
    return fig


def plot_quality_metrics(df: pd.DataFrame, output_dir: Path, theme: Theme) -> list[Path]:
    paths = []
    specs = [
        ("val_prd", "PRD (%)", "Percent Root-Mean-Square Difference", ".1f"),
        ("val_mse", "MSE", "Mean Squared Error", ".4f"),
        ("val_cos", "Cosine Similarity", "Cosine Similarity", ".4f"),
    ]
    for signal in ["PPG", "ECG"]:
        sdf = _signal_df(df, signal)
        for y_col, y_label, title, fmt in specs:
            fig = _line_plot(sdf, signal, y_col, y_label, title, theme, fmt)
            slug = y_col.replace("val_", "")
            paths.append(_save(fig, output_dir, f"{signal.lower()}_{slug}", theme))
    return paths


def plot_effective_rate(df: pd.DataFrame, output_dir: Path, theme: Theme) -> list[Path]:
    paths = []
    for signal in ["PPG", "ECG"]:
        sdf = _signal_df(df, signal)
        c = _color(theme, signal)
        cl = _color_light(theme, signal)
        native_rate = SAMPLE_RATES[signal]

        fig, ax = plt.subplots(figsize=(7, 3.8))
        x = np.arange(len(sdf))
        bars = ax.bar(x, sdf["effective_sample_rate"], color=c, alpha=0.85, zorder=3)
        for bar, rate in zip(bars, sdf["effective_sample_rate"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.0f} Hz",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="600",
                color=c,
            )

        ax.axhline(native_rate, color=cl, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(
            len(sdf) - 0.6, native_rate + 2, f"Native {native_rate} Hz", fontsize=8, color=cl, alpha=0.7, ha="right"
        )

        ax.set_xticks(x)
        ax.set_xticklabels([CR_LABELS[CR_ORDER.index(cr)] for cr in sdf["cr_label"]])
        ax.set_xlabel("Compression Ratio")
        ax.set_ylabel("Effective Sample Rate (Hz)")
        ax.set_title(f"{signal} — Effective Latent Sample Rate")
        fig.tight_layout()
        paths.append(_save(fig, output_dir, f"{signal.lower()}_effective_rate", theme))
    return paths


def plot_hr_hrv(df: pd.DataFrame, output_dir: Path, theme: Theme) -> list[Path]:
    ppg = _signal_df(df, "PPG")
    ppg = ppg[ppg["hr_mae_bpm"].notna()]
    if ppg.empty:
        logger.warning("No HR/HRV data — skipping")
        return []

    paths = []
    specs = [
        ("hr_mae_bpm", "HR MAE (bpm)", "Heart Rate Error", ".2f"),
        ("sdnn_mae_ms", "SDNN MAE (ms)", "SDNN Error", ".1f"),
        ("rmssd_mae_ms", "RMSSD MAE (ms)", "RMSSD Error", ".1f"),
    ]
    for y_col, y_label, title, fmt in specs:
        fig = _line_plot(ppg, "PPG", y_col, y_label, title, theme, fmt, fill=True)
        slug = y_col.replace("_mae_bpm", "").replace("_mae_ms", "")
        paths.append(_save(fig, output_dir, f"ppg_{slug}", theme))
    return paths


def plot_bits_budget(df: pd.DataFrame, output_dir: Path, theme: Theme) -> list[Path]:
    paths = []
    for signal in ["PPG", "ECG"]:
        sdf = _signal_df(df, signal)
        c = _color(theme, signal)
        cl = _color_light(theme, signal)

        fig, ax = plt.subplots(figsize=(7, 3.8))
        x = np.arange(len(sdf))
        raw = sdf["raw_bits_per_frame"].values
        comp = sdf["bits_per_frame"].values

        ax.bar(x, raw, color=cl, alpha=0.30, label="Raw", zorder=2)
        ax.bar(x, comp, color=c, alpha=0.85, label="Compressed", zorder=3)
        for j, (r, cv) in enumerate(zip(raw, comp)):
            savings = (1 - cv / r) * 100
            ax.text(
                x[j],
                cv + 80,
                f"{savings:.0f}%\nsaved",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="500",
                color=c,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([CR_LABELS[CR_ORDER.index(cr)] for cr in sdf["cr_label"]])
        ax.set_xlabel("Compression Ratio")
        ax.set_ylabel("Bits per Frame")
        ax.set_title(f"{signal} — Bit Budget per Frame")
        ax.legend(fontsize=9, framealpha=0.7)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        fig.tight_layout()
        paths.append(_save(fig, output_dir, f"{signal.lower()}_bits_budget", theme))
    return paths


def plot_loss_and_rvq(df: pd.DataFrame, output_dir: Path, theme: Theme) -> list[Path]:
    paths = []
    for signal in ["PPG", "ECG"]:
        sdf = _signal_df(df, signal)
        c = _color(theme, signal)

        # Validation loss
        fig = _line_plot(sdf, signal, "val_loss", "Validation Loss", "Total Loss vs Compression", theme, ".4f")
        paths.append(_save(fig, output_dir, f"{signal.lower()}_val_loss", theme))

        # RVQ usage
        fig, ax = plt.subplots(figsize=(7, 3.8))
        usage = sdf["val_rvq_usage"] * 100
        ax.plot(sdf["compression_ratio"], usage, color=c, marker=MARKERS[signal], zorder=3)
        for _, row in sdf.iterrows():
            if pd.notna(row["val_rvq_usage"]):
                _annotate(
                    ax, row["compression_ratio"], row["val_rvq_usage"] * 100, f"{row['val_rvq_usage'] * 100:.0f}%", c
                )
        _cr_ticks(ax, sdf)
        ax.set_ylabel("Codebook Usage (%)")
        ax.set_title(f"{signal} — RVQ Codebook Utilization")
        ax.set_ylim(0, 105)
        fig.tight_layout()
        paths.append(_save(fig, output_dir, f"{signal.lower()}_rvq_usage", theme))
    return paths


# ── Main ───────────────────────────────────────────────────────────────────────

PLOT_FNS = [
    plot_quality_metrics,
    plot_effective_rate,
    plot_hr_hrv,
    plot_bits_budget,
    plot_loss_and_rvq,
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden result plots.")
    parser.add_argument("--input", type=Path, default=Path("results/golden_summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/plots"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_csv(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)
    args.output.mkdir(parents=True, exist_ok=True)

    all_paths: list[Path] = []
    for theme in THEMES:
        _apply_theme(theme)
        for fn in PLOT_FNS:
            all_paths.extend(fn(df, args.output, theme))

    logger.info("Generated %d plots in %s", len(all_paths), args.output)


if __name__ == "__main__":
    main()
