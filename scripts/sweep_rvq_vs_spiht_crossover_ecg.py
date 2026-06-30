"""ECG RVQ-vs-SPIHT crossover sweep across compression ratio AND noise level.

Goal: locate the operating region where the learned RVQ codec *beats* the
classical SPIHT baseline. RVQ is trained to denoise toward a clean ECG
manifold (filtered_loss), so it is expected to win once the input is noisy,
and to lose on near-pristine inputs (SPIHT's best case). This script makes
that crossover explicit, on both synthetic and real ECG morphology.

Two reference sources (``--source``):

* ``synthetic`` — clean ECG from the McSharry generator is the ground truth.
  Inputs are that clean signal plus controlled Gaussian noise at a target SNR.

* ``real`` — real PTB-XL windows. There is no true clean reference, so a
  zero-phase bandpass-filtered copy (0.5-40 Hz) is used as a clean-truth
  *proxy*. Inputs are: (a) the raw real window ("native", carrying whatever
  noise the recording already has) and (b) the proxy plus controlled Gaussian
  noise at a target SNR. The script also crudely estimates the *native* SNR of
  the real windows (signal vs high-frequency residual) so the real data can be
  placed on the same SNR axis.

For each compression ratio we report PRD-vs-clean for both codecs at every
column, the signed gap ``delta = PRD_rvq - PRD_spiht`` (negative => RVQ wins),
and the crossover SNR: the *highest* (cleanest) input SNR at which RVQ still
beats SPIHT. Outputs a JSON summary and a CR x SNR heatmap of the gap.
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

from compressionkit.datasets.ecg import _resample, load_ecg_signal
from compressionkit.evaluation.codec import SpihtAcCodec
from compressionkit.evaluation.empirical_regime import normalize_signal, snr_label
from compressionkit.evaluation.rvq_codec import RvqCodec

# Reuse the validated helpers from the 4x/8x noise sweep so the methodology
# stays identical across both scripts.
from scripts.sweep_codec_noise_ecg import (
    _corr,
    _prd,
    add_gaussian,
    build_clean_windows,
    encode_decode_batch,
)

# CR -> default RVQ golden run dir. Each model carries its own native CR; the
# matched SPIHT codec is built with the same nominal target_cr.
DEFAULT_RVQ_RUN_DIRS: dict[int, Path] = {
    2: Path("results/ecg_rvq_256hz_02x_golden_long_gpu"),
    4: Path("results/ecg_rvq_256hz_04x_golden_long_gpu"),
    8: Path("results/ecg_rvq_256hz_08x_golden_long_gpu"),
    16: Path("results/ecg_rvq_256hz_16x_golden_long_gpu"),
    32: Path("results/ecg_rvq_256hz_32x_golden_long_gpu"),
    64: Path("results/ecg_rvq_256hz_64x_golden_long_gpu"),
}

# Default input-SNR grid in dB. ``None`` denotes the pristine (clean) input.
# Range is chosen to span pristine -> wearable-realistic -> heavy noise so the
# crossover does not sit only at an extreme.
DEFAULT_SNR_DB: list[float | None] = [None, 30.0, 24.0, 20.0, 16.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0]

# Clean-truth proxy band for real ECG (matches the noise-estimation band used
# in compressionkit.preprocessing.ecg.extract_noise_segments).
PROXY_LOW_HZ = 0.5
PROXY_HIGH_HZ = 40.0


def _snr_to_noise_pct(snr_db: float | None) -> float:
    """Convert an input SNR in dB to ``noise_std / signal_std``."""
    if snr_db is None:
        return 0.0
    return float(10.0 ** (-snr_db / 20.0))


# Backward-compatible aliases — these names are imported by release eval tools.
_snr_label = snr_label
_normalize = normalize_signal


def _bandpass_proxy(window: np.ndarray, sample_rate: float) -> np.ndarray:
    """Zero-phase bandpass (clean-truth proxy) for a single 1-D window."""
    from scipy import signal as scipy_signal

    nyq = sample_rate / 2.0
    high = min(PROXY_HIGH_HZ, nyq * 0.95)
    sos = scipy_signal.butter(3, [PROXY_LOW_HZ / nyq, high / nyq], btype="bandpass", output="sos")
    return scipy_signal.sosfiltfilt(sos, window).astype(np.float32)


def build_real_windows(
    n_windows: int,
    frame_size: int,
    sample_rate: float,
    *,
    data_dir: Path,
    glob_pattern: str,
    lead_index: int,
    source_sample_rate: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real PTB-XL windows and build a clean-truth proxy.

    Returns ``(proxy, native_raw, native_snr_db)`` where ``proxy`` is the
    normalized bandpass-filtered clean reference, ``native_raw`` is the
    normalized raw real input (carrying recording noise), and
    ``native_snr_db`` is the per-window crude SNR estimate of the raw input.
    """
    rng = np.random.default_rng(seed)
    files = sorted(data_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No real ECG files matched {data_dir / glob_pattern}")
    rng.shuffle(files)

    # Window length at the source rate that yields ``frame_size`` after resample.
    src_window = int(round(frame_size * source_sample_rate / sample_rate))

    proxy = np.empty((n_windows, frame_size), dtype=np.float32)
    native = np.empty((n_windows, frame_size), dtype=np.float32)
    snr_db = np.empty((n_windows,), dtype=np.float32)

    collected = 0
    fi = 0
    while collected < n_windows and fi < len(files):
        path = files[fi]
        fi += 1
        try:
            sig = load_ecg_signal(path, lead_index=lead_index)
        except Exception:
            continue
        if sig.ndim != 1 or sig.shape[0] < src_window:
            continue
        start = int(rng.integers(0, sig.shape[0] - src_window + 1))
        raw = sig[start : start + src_window].astype(np.float32)
        if source_sample_rate != int(sample_rate):
            raw = _resample(raw, source_sample_rate, int(sample_rate))
        raw = raw[:frame_size]
        if raw.shape[0] < frame_size or not np.isfinite(raw).all() or raw.std() < 1e-6:
            continue

        clean = _bandpass_proxy(raw, sample_rate)
        residual = raw - clean
        res_std = float(residual.std())
        sig_std = float(clean.std())
        snr_db[collected] = 20.0 * math.log10(sig_std / (res_std + 1e-9)) if res_std > 0 else 60.0

        proxy[collected] = _normalize(clean)
        native[collected] = _normalize(raw)
        collected += 1

    if collected < n_windows:
        raise RuntimeError(f"Only collected {collected}/{n_windows} valid real windows; widen the glob.")

    return proxy, native, snr_db


def _parse_snr_db(text: str) -> list[float | None]:
    out: list[float | None] = []
    for part in text.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"clean", "inf", "none"}:
            out.append(None)
        else:
            out.append(float(token))
    if not out:
        raise ValueError("--snr-db must provide at least one comma-separated value")
    return out


def _parse_crs(text: str) -> list[int]:
    out = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not out:
        raise ValueError("--crs must provide at least one comma-separated value")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--n-windows", type=int, default=64)
    ap.add_argument(
        "--crs",
        type=_parse_crs,
        default=[2, 4, 8, 16, 32, 64],
        help="Comma-separated compression ratios to evaluate (e.g. '4,8,16').",
    )
    ap.add_argument(
        "--snr-db",
        type=_parse_snr_db,
        default=DEFAULT_SNR_DB,
        help="Comma-separated input SNR levels in dB; use 'clean' for pristine.",
    )
    ap.add_argument(
        "--run-dirs",
        type=str,
        default=None,
        help="Optional override mapping 'cr=path,cr=path' for RVQ run dirs.",
    )
    # Real-source options.
    ap.add_argument("--data-dir", type=Path, default=Path("datasets/ptbxl"))
    ap.add_argument("--data-glob", type=str, default="*.h5")
    ap.add_argument("--lead-index", type=int, default=1)
    ap.add_argument("--source-sample-rate", type=int, default=500)
    ap.add_argument("--output-stem", type=str, default=None)
    args = ap.parse_args()

    run_dirs: dict[int, Path] = dict(DEFAULT_RVQ_RUN_DIRS)
    if args.run_dirs:
        for item in args.run_dirs.split(","):
            cr_str, path_str = item.split("=", 1)
            run_dirs[int(cr_str.strip())] = Path(path_str.strip())

    out_dir = Path("results/_rvq_vs_spiht_crossover")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem or f"ecg_rvq_vs_spiht_crossover_{args.source}"

    sample_rate = 256.0
    frame_size = 512  # matches ECG RVQ goldens
    snr_grid = args.snr_db
    noise_pcts = [_snr_to_noise_pct(s) for s in snr_grid]

    native_snr_summary: dict | None = None
    native_input: np.ndarray | None = None

    if args.source == "synthetic":
        print(
            f"[setup] Building {args.n_windows} synthetic clean ECG windows "
            f"({frame_size} samples @ {sample_rate:g} Hz, ~{frame_size / sample_rate:.1f} s)"
        )
        clean = build_clean_windows(args.n_windows, frame_size, sample_rate)
    else:
        print(
            f"[setup] Loading {args.n_windows} real PTB-XL windows from "
            f"{args.data_dir / args.data_glob} (lead {args.lead_index}) ..."
        )
        clean, native_input, native_snr = build_real_windows(
            args.n_windows,
            frame_size,
            sample_rate,
            data_dir=args.data_dir,
            glob_pattern=args.data_glob,
            lead_index=args.lead_index,
            source_sample_rate=args.source_sample_rate,
        )
        native_snr_summary = {
            "median_db": float(np.median(native_snr)),
            "mean_db": float(np.mean(native_snr)),
            "p10_db": float(np.percentile(native_snr, 10)),
            "p90_db": float(np.percentile(native_snr, 90)),
            "min_db": float(np.min(native_snr)),
            "max_db": float(np.max(native_snr)),
        }
        print(
            "[native-SNR] crude estimate of real input SNR (signal vs >40Hz residual): "
            f"median={native_snr_summary['median_db']:.1f} dB  "
            f"mean={native_snr_summary['mean_db']:.1f} dB  "
            f"p10..p90=[{native_snr_summary['p10_db']:.1f}, {native_snr_summary['p90_db']:.1f}] dB"
        )

    # Build the column list: (label, snr_db_or_None, input_array). The numeric
    # SNR columns add Gaussian noise to the clean reference. For real data we
    # prepend a "native" column = raw real input vs the filtered proxy truth.
    columns: list[tuple[str, float | None, np.ndarray]] = []
    if args.source == "real" and native_input is not None:
        columns.append(("native", None, native_input))
    for snr_db, npct in zip(snr_grid, noise_pcts):
        if npct == 0.0:
            inp = clean.copy()
        else:
            inp = add_gaussian(clean, npct, seed=int(round(npct * 1000)) + 17)
        columns.append((_snr_label(snr_db), snr_db, inp))

    summary: dict = {
        "source": args.source,
        "sample_rate": sample_rate,
        "frame_size": frame_size,
        "n_windows": args.n_windows,
        "columns": [c[0] for c in columns],
        "snr_db": [s for _, s, _ in columns],
        "crs": args.crs,
        "native_snr": native_snr_summary,
        "by_cr": {},
        "crossover_snr_db": {},
    }

    n_cols = len(columns)
    delta_grid = np.full((len(args.crs), n_cols), np.nan, dtype=np.float64)

    for ci, cr in enumerate(args.crs):
        run_dir = run_dirs.get(cr)
        if run_dir is None or not run_dir.exists():
            print(f"[skip] CR {cr}x: no RVQ run dir ({run_dir}); skipping.")
            continue

        print(f"\n=== CR {cr}x ===")
        spiht = SpihtAcCodec(
            name=f"spiht_{cr}x",
            modality="ecg",
            sample_rate=int(sample_rate),
            frame_size=frame_size,
            target_cr=float(cr),
        )
        print(f"[setup] Loading RVQ {cr}x from {run_dir} ...")
        rvq = RvqCodec.from_run_dir(run_dir, modality="ecg")

        cr_block: dict = {"columns": [], "spiht": [], "rvq": [], "delta_prd": []}
        crossover_snr: float | None = None

        for si, (col_label, snr_db, inp) in enumerate(columns):
            recon_spiht = encode_decode_batch(spiht, inp)
            recon_rvq = encode_decode_batch(rvq, inp)

            prd_spiht = float(_prd(clean, recon_spiht).mean())
            prd_rvq = float(_prd(clean, recon_rvq).mean())
            corr_spiht = float(_corr(clean, recon_spiht).mean())
            corr_rvq = float(_corr(clean, recon_rvq).mean())
            delta = prd_rvq - prd_spiht
            delta_grid[ci, si] = delta

            cr_block["columns"].append(col_label)
            cr_block["spiht"].append({"prd_vs_clean": prd_spiht, "corr_vs_clean": corr_spiht})
            cr_block["rvq"].append({"prd_vs_clean": prd_rvq, "corr_vs_clean": corr_rvq})
            cr_block["delta_prd"].append(delta)

            # Crossover = cleanest (highest) numeric SNR at which RVQ still wins.
            if crossover_snr is None and delta < 0.0 and snr_db is not None:
                crossover_snr = snr_db

            winner = "RVQ " if delta < 0 else "SPIHT"
            print(
                f"  {col_label:>7s}: "
                f"SPIHT PRDc={prd_spiht:6.2f}%  RVQ PRDc={prd_rvq:6.2f}%  "
                f"delta={delta:+6.2f}  -> {winner}"
            )

        summary["by_cr"][f"{cr}x"] = cr_block
        summary["crossover_snr_db"][f"{cr}x"] = crossover_snr
        if crossover_snr is None:
            print(f"  [crossover] CR {cr}x: RVQ never beats SPIHT on the numeric-SNR grid.")
        else:
            print(f"  [crossover] CR {cr}x: RVQ overtakes SPIHT at <= {crossover_snr:g} dB input SNR.")

    json_path = out_dir / f"{output_stem}.json"
    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=lambda o: None if isinstance(o, float) and math.isnan(o) else o,
        )
    )
    print(f"\nWrote {json_path}")

    col_labels = [c[0] for c in columns]
    title_src = "synthetic McSharry truth" if args.source == "synthetic" else "real PTB-XL (filtered proxy truth)"
    native_note = ""
    if native_snr_summary is not None:
        native_note = f"\nnative real input SNR ~ {native_snr_summary['median_db']:.0f} dB (median)"
    _plot_heatmap(
        delta_grid,
        args.crs,
        col_labels,
        out_dir / f"{output_stem}.png",
        title=f"RVQ - SPIHT  PRD-vs-clean gap (%) — {title_src}{native_note}",
    )
    print(f"Wrote {out_dir / f'{output_stem}.png'}")


def _plot_heatmap(
    delta_grid: np.ndarray,
    crs: list[int],
    col_labels: list[str],
    png_path: Path,
    *,
    title: str,
) -> None:
    """Render the CR x column PRD gap (RVQ - SPIHT); negative => RVQ wins."""
    fig, ax = plt.subplots(figsize=(1.1 * len(col_labels) + 3, 0.7 * len(crs) + 2.8))
    vmax = float(np.nanmax(np.abs(delta_grid))) if np.isfinite(delta_grid).any() else 1.0
    im = ax.imshow(
        delta_grid,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
    )
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(crs)))
    ax.set_yticklabels([f"{cr}x" for cr in crs])
    ax.set_xlabel("Input condition / SNR")
    ax.set_ylabel("Compression ratio")
    ax.set_title(
        f"{title}\nblue = RVQ wins (denoises better)   red = SPIHT wins",
        fontsize=10,
    )
    for i in range(delta_grid.shape[0]):
        for j in range(delta_grid.shape[1]):
            val = delta_grid[i, j]
            if not np.isfinite(val):
                continue
            ax.text(
                j,
                i,
                f"{val:+.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if abs(val) < 0.6 * vmax else "white",
            )
    fig.colorbar(im, ax=ax, label="PRD gap (%)  (negative = RVQ better)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
