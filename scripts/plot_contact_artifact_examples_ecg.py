#!/usr/bin/env python3
"""Render ECG contact-artifact examples across families and severities."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compressionkit.preprocessing.ecg import build_noise_bank_from_h5
from compressionkit.synthetic.ecg_contact_artifacts import (
    DEFAULT_CONTACT_FAMILIES,
    simulate_contact_artifact_batch,
)
from scripts.sweep_rvq_vs_spiht_crossover_ecg import build_real_windows


def _parse_families(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_severities(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _family_title(name: str) -> str:
    return name.replace("_", " ")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--frame-size", type=int, default=512)
    ap.add_argument("--sample-rate", type=int, default=256)
    ap.add_argument("--source-sample-rate", type=int, default=500)
    ap.add_argument("--lead-index", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--noise-bank-files", type=int, default=300)
    ap.add_argument("--data-dir", type=Path, default=Path("datasets/ptbxl"))
    ap.add_argument("--glob-pattern", type=str, default="*.h5")
    ap.add_argument("--families", type=_parse_families, default=list(DEFAULT_CONTACT_FAMILIES))
    ap.add_argument("--severities", type=_parse_severities, default=[0.25, 0.5, 1.0])
    ap.add_argument("--include-pure-artifact", action="store_true")
    ap.add_argument("--output-stem", type=str, default="ecg_contact_artifact_examples")
    return ap.parse_args()


def _plot_example_sheet(
    output_path: Path,
    *,
    time_axis: np.ndarray,
    example_index: int,
    native: np.ndarray,
    clean: np.ndarray,
    native_snr_db: float,
    families: list[str],
    severities: list[float],
    rendered: dict[tuple[str, float], np.ndarray],
) -> None:
    n_rows = len(families)
    n_cols = 2 + len(severities)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.5 * n_cols + 1.2, 1.7 * n_rows + 1.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    row_arrays = [native, clean] + [rendered[(family, severity)] for family in families for severity in severities]
    ylim = max(float(np.max(np.abs(arr))) for arr in row_arrays) * 1.12
    ylim = max(ylim, 1.5)

    for row, family in enumerate(families):
        ax_native = axes[row, 0]
        ax_native.plot(time_axis, native, color="#222222", lw=1.0)
        ax_native.set_ylim(-ylim, ylim)
        ax_native.grid(alpha=0.15, linewidth=0.5)
        ax_native.set_ylabel(_family_title(family))
        if row == 0:
            ax_native.set_title(f"native\n~{native_snr_db:.1f} dB")

        ax_clean = axes[row, 1]
        ax_clean.plot(time_axis, clean, color="#1f77b4", lw=1.0)
        ax_clean.set_ylim(-ylim, ylim)
        ax_clean.grid(alpha=0.15, linewidth=0.5)
        if row == 0:
            ax_clean.set_title("clean")

        for col, severity in enumerate(severities, start=2):
            ax = axes[row, col]
            ax.plot(time_axis, rendered[(family, severity)], color="#b04a1f", lw=1.0)
            ax.set_ylim(-ylim, ylim)
            ax.grid(alpha=0.15, linewidth=0.5)
            if row == 0:
                ax.set_title(f"sev={severity:.2f}")

    for ax in axes[-1, :]:
        ax.set_xlabel("seconds")

    fig.suptitle(f"ECG contact artifacts: example {example_index + 1}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    families = list(args.families)
    if args.include_pure_artifact and "pure_artifact" not in families:
        families.append("pure_artifact")

    filtered, native, native_snr_db = build_real_windows(
        n_windows=args.n_examples,
        frame_size=args.frame_size,
        sample_rate=float(args.sample_rate),
        data_dir=args.data_dir,
        glob_pattern=args.glob_pattern,
        lead_index=args.lead_index,
        source_sample_rate=args.source_sample_rate,
        seed=args.seed,
    )
    bank_files = sorted(args.data_dir.glob(args.glob_pattern))[: args.noise_bank_files]
    noise_bank = build_noise_bank_from_h5(
        bank_files,
        source_sample_rate=args.source_sample_rate,
        target_sample_rate=args.sample_rate,
        window_size=args.frame_size,
        lead_index=args.lead_index,
    )
    if noise_bank is None or len(noise_bank) == 0:
        raise RuntimeError("Failed to build an empirical residual bank from the requested PTB-XL files")

    rendered_batches: dict[tuple[str, float], np.ndarray] = {}
    for family_index, family in enumerate(families):
        for severity_index, severity in enumerate(args.severities):
            seed = args.seed + 1000 + 100 * family_index + severity_index
            rendered_batches[(family, severity)] = simulate_contact_artifact_batch(
                filtered,
                family=family,
                severity=severity,
                sample_rate=float(args.sample_rate),
                seed=seed,
                noise_bank=noise_bank,
            )

    out_dir = Path("results") / "_rvq_vs_spiht_crossover"
    out_dir.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(args.frame_size, dtype=np.float32) / float(args.sample_rate)

    manifest: dict[str, object] = {
        "families": families,
        "severities": args.severities,
        "files": [],
    }
    for index in range(args.n_examples):
        output_path = out_dir / f"{args.output_stem}_ex{index + 1}.png"
        rendered = {
            (family, severity): rendered_batches[(family, severity)][index]
            for family in families
            for severity in args.severities
        }
        _plot_example_sheet(
            output_path,
            time_axis=time_axis,
            example_index=index,
            native=native[index],
            clean=filtered[index],
            native_snr_db=float(native_snr_db[index]),
            families=families,
            severities=args.severities,
            rendered=rendered,
        )
        manifest["files"].append(output_path.name)
        print(f"Wrote {output_path}")

    manifest_path = out_dir / f"{args.output_stem}_manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
