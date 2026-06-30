"""PPG empirical-noise regime map: SPIHT vs RVQ on real windows.

This is the PPG counterpart to ``sweep_empirical_regime_ecg``. It loads real
PPG validation windows, builds a bandpass-filtered clean-truth proxy, injects
empirical wrist noise at exact SNR, and compares SPIHT / hybrid / RVQ using a
dual reference:

* truth fidelity: reconstruction vs filtered proxy
* faithfulness: reconstruction vs the actual noisy input

Outputs a JSON summary plus PRD-gap heatmaps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import physiokit as pk

from compressionkit.configs.ppg_rvq import PpgRvqConfig
from compressionkit.datasets.ppg import collect_random_samples
from compressionkit.evaluation.codec import BayesShrinkSpihtCodec, SpihtAcCodec
from compressionkit.evaluation.noise import estimate_ppg_noise_floor
from compressionkit.evaluation.rvq_codec import RvqCodec
from compressionkit.playbook.catalog import get_method
from compressionkit.preprocessing.augmentations import build_noise_bank_from_h5
from compressionkit.preprocessing.ppg import build_augmenter, build_preprocessor
from compressionkit.trainers.ppg_rvq import build_datasets as build_ppg_datasets

from scripts.sweep_codec_noise_ppg import _corr, _prd, encode_decode_batch

from compressionkit.evaluation.empirical_regime import (
    DEFAULT_SNR_DB,
    add_empirical_noise,
    autocorr_peak,
    normalize_signal,
    sample_noise_segment,
    snr_label,
)


DEFAULT_RVQ_RUN_DIRS: dict[int, Path] = {
    2: Path("results/ppg_rvq_64hz_02x_golden"),
    4: Path("results/ppg_rvq_64hz_04x_golden"),
    8: Path("results/ppg_rvq_64hz_08x_golden"),
    16: Path("results/ppg_rvq_64hz_16x_golden"),
    32: Path("results/ppg_rvq_64hz_32x_golden"),
}

HYBRID_METHOD_IDS = {
    "bayes_shrink_spiht",
    "filter_spiht",
    "learned_shrink_spiht",
}

# Backward-compatible aliases — these names are imported by release eval tools.
_normalize = normalize_signal
_sample_noise_segment = sample_noise_segment
_snr_label = snr_label


def _pulse_autocorr_peak(recon: np.ndarray, sample_rate: float) -> np.ndarray:
    """Max normalized autocorrelation in the 40-180 bpm pulse band, per window."""
    return autocorr_peak(recon, sample_rate, lo_bpm=40.0, hi_bpm=180.0)


def _parse_snr_db(text: str) -> list[float | None]:
    out: list[float | None] = []
    for part in text.split(","):
        token = part.strip().lower()
        if not token:
            continue
        out.append(None if token in {"clean", "inf", "none"} else float(token))
    if not out:
        raise ValueError("--snr-db must provide at least one value")
    return out


def _parse_crs(text: str) -> list[int]:
    out = [int(p.strip()) for p in text.split(",") if p.strip()]
    if not out:
        raise ValueError("--crs must provide at least one value")
    return out


def _parse_sources(text: str) -> list[str]:
    out = [p.strip() for p in text.split(",") if p.strip()]
    if not out:
        raise ValueError("--noise-bank-sources must provide at least one source")
    return out


def _build_hybrid_codec(method_id: str, *, sample_rate: int, frame_size: int, target_cr: float):
    if method_id == "bayes_shrink_spiht":
        return BayesShrinkSpihtCodec(
            name=f"bayes_spiht_{target_cr:g}x",
            modality="ppg",
            sample_rate=sample_rate,
            frame_size=frame_size,
            target_cr=target_cr,
        )
    card = get_method(method_id)
    return card.builder(  # type: ignore[misc]
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
        modality="ppg",
    )


def build_real_windows(
    n_windows: int,
    *,
    reference_run: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PpgRvqConfig]:
    cfg = PpgRvqConfig.model_validate_json((reference_run / "config.json").read_text())
    preprocessor = build_preprocessor(frame_size=cfg.data.frame_size, epsilon=cfg.data.epsilon)
    augmenter = build_augmenter(tuple(cfg.data.gaussian_noise), aug_cfg=cfg.data.augmentation)
    _train_ds, val_ds, _validation_steps, _info = build_ppg_datasets(cfg, preprocessor, augmenter)
    rng = np.random.default_rng(cfg.data.shuffle_seed)
    _inputs, targets = collect_random_samples(val_ds, n_windows, rng)
    raw_native = targets[:, 0, :, 0].astype(np.float32)
    filtered = np.empty_like(raw_native)
    native_snr = np.empty(raw_native.shape[0], dtype=np.float32)
    for i, x in enumerate(raw_native):
        clean = pk.signal.filter_signal(
            x.astype(np.float32),
            sample_rate=cfg.data.sampling_rate,
            lowcut=0.5,
            highcut=8.0,
            order=3,
            forward_backward=True,
        ).astype(np.float32)
        filtered[i] = _normalize(clean)
        nf = estimate_ppg_noise_floor(x, cfg.data.sampling_rate)
        sig_power = float(nf.get("bp_signal_power", 0.0)) + 1e-12
        noise_power = float(nf.get("bp_noise_power", 0.0)) + 1e-12
        native_snr[i] = float(10.0 * np.log10(sig_power / noise_power))
    return filtered, raw_native, native_snr, cfg


def _plot_heatmap(grid: np.ndarray, crs: list[int], col_labels: list[str], png_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(1.1 * len(col_labels) + 3, 0.7 * len(crs) + 2.8))
    vmax = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(crs)))
    ax.set_yticklabels([f"{cr}x" for cr in crs])
    ax.set_xlabel("Input condition / true SNR")
    ax.set_ylabel("Compression ratio")
    ax.set_title(f"{title}\nblue = RVQ better   red = SPIHT better", fontsize=9)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if abs(v) < 0.6 * vmax else "white",
                )
    fig.colorbar(im, ax=ax, label="PRD gap (%)  (negative = RVQ better)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-run", type=Path, default=Path("results/ppg_rvq_64hz_08x_golden"))
    ap.add_argument("--n-windows", type=int, default=300)
    ap.add_argument("--crs", type=_parse_crs, default=[2, 4, 8, 16, 32])
    ap.add_argument("--snr-db", type=_parse_snr_db, default=DEFAULT_SNR_DB)
    ap.add_argument("--run-dirs", type=str, default=None)
    ap.add_argument("--noise-bank-root", type=Path, default=Path("/home/vscode/datasets"))
    ap.add_argument("--noise-bank-sources", type=_parse_sources, default=["ppg_dalia", "wesad"])
    ap.add_argument("--noise-bank-files", type=int, default=400)
    ap.add_argument("--output-stem", type=str, default="ppg_empirical_regime")
    ap.add_argument(
        "--hybrid-method",
        type=str,
        default="bayes_shrink_spiht",
        choices=sorted(HYBRID_METHOD_IDS),
        help="SPIHT-front-end denoise variant to compare against plain SPIHT and RVQ.",
    )
    args = ap.parse_args()

    run_dirs: dict[int, Path] = dict(DEFAULT_RVQ_RUN_DIRS)
    if args.run_dirs:
        for item in args.run_dirs.split(","):
            cr_str, path_str = item.split("=", 1)
            run_dirs[int(cr_str.strip())] = Path(path_str.strip())

    out_dir = Path("results/_rvq_vs_spiht_crossover")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Loading {args.n_windows} real PPG windows for references ...")
    filtered, raw_native, native_snr, ref_cfg = build_real_windows(
        args.n_windows,
        reference_run=args.reference_run,
    )
    sample_rate = float(ref_cfg.data.sampling_rate)
    frame_size = int(ref_cfg.data.frame_size)
    native_snr_summary = {
        "median_db": float(np.median(native_snr)),
        "mean_db": float(np.mean(native_snr)),
        "p10_db": float(np.percentile(native_snr, 10)),
        "p90_db": float(np.percentile(native_snr, 90)),
    }
    print(
        "[native-SNR] real input (signal vs 0.5-8Hz residual): "
        f"median={native_snr_summary['median_db']:.1f} dB  "
        f"p10..p90=[{native_snr_summary['p10_db']:.1f}, {native_snr_summary['p90_db']:.1f}] dB"
    )

    source_files: list[str] = []
    for source in args.noise_bank_sources:
        source_files.extend(str(p) for p in sorted((args.noise_bank_root / source).glob("*.h5")))
    source_files = source_files[: args.noise_bank_files]
    print(f"[setup] Building empirical PPG noise bank from up to {len(source_files)} files ...")
    noise_bank = build_noise_bank_from_h5(
        source_files,
        target_fs=int(sample_rate),
        window_size=frame_size,
        max_segments=5000,
    )
    if noise_bank is None or len(noise_bank) == 0:
        raise RuntimeError("Empirical PPG noise bank is empty; widen --noise-bank-sources or --noise-bank-root.")
    print(f"[setup] Noise bank: {len(noise_bank)} residual segments.")

    columns: list[tuple[str, float | None, np.ndarray, np.ndarray]] = []
    columns.append(("native", None, raw_native, raw_native))
    for snr_db in args.snr_db:
        if snr_db is None:
            inp = filtered.copy()
        else:
            inp = add_empirical_noise(filtered, noise_bank, snr_db, seed=1000 + int(round(snr_db)))
        columns.append((_snr_label(snr_db), snr_db, inp, inp))

    probe_rng = np.random.default_rng(7)
    pure_noise = np.stack(
        [_normalize(_sample_noise_segment(noise_bank, frame_size, probe_rng)) for _ in range(min(args.n_windows, 128))]
    ).astype(np.float32)

    summary: dict = {
        "source": "real-empirical",
        "modality": "ppg",
        "hybrid_method": args.hybrid_method,
        "sample_rate": sample_rate,
        "frame_size": frame_size,
        "reference_run": str(args.reference_run),
        "n_windows": args.n_windows,
        "columns": [c[0] for c in columns],
        "snr_db": [c[1] for c in columns],
        "crs": args.crs,
        "native_snr": native_snr_summary,
        "noise_bank_size": int(len(noise_bank)),
        "by_cr": {},
        "imprinting": {},
    }

    n_cols = len(columns)
    truth_gap = np.full((len(args.crs), n_cols), np.nan)
    faith_gap = np.full((len(args.crs), n_cols), np.nan)
    hybrid_truth_gap = np.full((len(args.crs), n_cols), np.nan)
    hybrid_faith_gap = np.full((len(args.crs), n_cols), np.nan)

    for ci, cr in enumerate(args.crs):
        run_dir = run_dirs.get(cr)
        if run_dir is None or not run_dir.exists():
            print(f"[skip] CR {cr}x: no RVQ run dir ({run_dir}).")
            continue

        print(f"\n=== CR {cr}x ===")
        spiht = SpihtAcCodec(
            name=f"spiht_{cr}x",
            modality="ppg",
            sample_rate=int(sample_rate),
            frame_size=frame_size,
            target_cr=float(cr),
        )
        hybrid = _build_hybrid_codec(
            args.hybrid_method,
            sample_rate=int(sample_rate),
            frame_size=frame_size,
            target_cr=float(cr),
        )
        rvq = RvqCodec.from_run_dir(run_dir, modality="ppg")

        block: dict = {"columns": [], "spiht": [], "hybrid": [], "rvq": []}
        for si, (label, snr_db, inp, faithful_ref) in enumerate(columns):
            rec_s = encode_decode_batch(spiht, inp)
            rec_h = encode_decode_batch(hybrid, inp)
            rec_r = encode_decode_batch(rvq, inp)

            ts = float(_prd(filtered, rec_s).mean())
            th = float(_prd(filtered, rec_h).mean())
            tr = float(_prd(filtered, rec_r).mean())
            fs = float(_prd(faithful_ref, rec_s).mean())
            fh = float(_prd(faithful_ref, rec_h).mean())
            fr = float(_prd(faithful_ref, rec_r).mean())
            cs = float(_corr(filtered, rec_s).mean())
            ch = float(_corr(filtered, rec_h).mean())
            crr = float(_corr(filtered, rec_r).mean())

            truth_gap[ci, si] = tr - ts
            faith_gap[ci, si] = fr - fs
            hybrid_truth_gap[ci, si] = th - ts
            hybrid_faith_gap[ci, si] = fh - fs

            block["columns"].append(label)
            block["spiht"].append({"prd_truth": ts, "prd_faithful": fs, "corr_truth": cs})
            block["hybrid"].append({"prd_truth": th, "prd_faithful": fh, "corr_truth": ch})
            block["rvq"].append({"prd_truth": tr, "prd_faithful": fr, "corr_truth": crr})

            win_t = "RVQ " if tr < ts else "SPIHT"
            win_h = "DSP" if th < ts else "SPIHT"
            print(
                f"  {label:>7s}: truth  SPIHT={ts:6.2f}  DSP={th:6.2f} ({th - ts:+5.2f} {win_h})"
                f"  RVQ={tr:6.2f} ({tr - ts:+5.2f} {win_t})"
                f"   |  faithful  SPIHT={fs:6.2f}  DSP={fh:6.2f}  RVQ={fr:6.2f}"
            )

        probe_s = encode_decode_batch(spiht, pure_noise)
        probe_h = encode_decode_batch(hybrid, pure_noise)
        probe_r = encode_decode_batch(rvq, pure_noise)
        ac_in = float(_pulse_autocorr_peak(pure_noise, sample_rate).mean())
        ac_s = float(_pulse_autocorr_peak(probe_s, sample_rate).mean())
        ac_h = float(_pulse_autocorr_peak(probe_h, sample_rate).mean())
        ac_r = float(_pulse_autocorr_peak(probe_r, sample_rate).mean())
        summary["imprinting"][f"{cr}x"] = {
            "input_rr_autocorr": ac_in,
            "spiht_rr_autocorr": ac_s,
            "hybrid_rr_autocorr": ac_h,
            "rvq_rr_autocorr": ac_r,
        }
        print(
            f"  [imprint] pure-noise pulse-autocorr: input={ac_in:.3f}  "
            f"SPIHT={ac_s:.3f}  DSP={ac_h:.3f}  RVQ={ac_r:.3f}  "
            f"(higher = more invented periodicity)"
        )

        summary["by_cr"][f"{cr}x"] = block

    json_path = out_dir / f"{args.output_stem}.json"
    json_path.write_text(
        json.dumps(summary, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o)
    )
    print(f"\nWrote {json_path}")

    col_labels = [c[0] for c in columns]
    native_note = f"native real SNR ~ {native_snr_summary['median_db']:.0f} dB (median)"
    _plot_heatmap(
        truth_gap,
        args.crs,
        col_labels,
        out_dir / f"{args.output_stem}_truth.png",
        title=f"RVQ - SPIHT PRD-vs-FILTERED gap (truth fidelity) — empirical noise\n{native_note}",
    )
    _plot_heatmap(
        faith_gap,
        args.crs,
        col_labels,
        out_dir / f"{args.output_stem}_faithful.png",
        title=f"RVQ - SPIHT PRD-vs-INPUT gap (faithfulness) — empirical noise\n{native_note}",
    )
    _plot_heatmap(
        hybrid_truth_gap,
        args.crs,
        col_labels,
        out_dir / f"{args.output_stem}_hybrid_truth.png",
        title=f"{args.hybrid_method} - SPIHT PRD-vs-FILTERED gap — empirical noise\n{native_note}",
    )
    _plot_heatmap(
        hybrid_faith_gap,
        args.crs,
        col_labels,
        out_dir / f"{args.output_stem}_hybrid_faithful.png",
        title=f"{args.hybrid_method} - SPIHT PRD-vs-INPUT gap — empirical noise\n{native_note}",
    )
    print(
        f"Wrote {out_dir / f'{args.output_stem}_truth.png'}, _faithful.png, "
        f"_hybrid_truth.png, and _hybrid_faithful.png"
    )


if __name__ == "__main__":
    main()
