"""PPG 3-lane crossover: SPIHT (DSP) vs Learned+SPIHT (Hybrid) vs RVQ (AI).

Mirror of :mod:`scripts.eval_learned_shrink_ecg` for PPG. Scores the three v1
codec lanes on the same empirical true-SNR columns so the DSP -> Hybrid -> AI
handoff can be read off a single winner heatmap as CR and noise rise.

* ``SPIHT``         — plain bior4.4 DWT + SPIHT (DSP faithfulness baseline)
* ``Learned+SPIHT``— locked wavelet-gain denoiser + SPIHT (Hybrid lane)
* ``RVQ``          — trained neural codec golden (AI lane)

Truth fidelity is PRD against the bandpassed clean proxy (0.5-8 Hz), matching
the ECG harness. Writes a ``by_cr`` JSON consumable by
``scripts/plot_codec_winner_heatmap.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np

from compressionkit.evaluation.codec import (
    BayesShrinkSpihtCodec,
    LearnedShrinkSpihtCodec,
    SpihtAcCodec,
)
from compressionkit.evaluation.rvq_codec import RvqCodec
from compressionkit.models.wavelet_denoiser import as_coeff_denoiser
from compressionkit.preprocessing.augmentations import build_noise_bank_from_h5

from scripts.sweep_codec_noise_ppg import _prd, encode_decode_batch
from scripts.sweep_empirical_regime_ppg import (
    DEFAULT_RVQ_RUN_DIRS,
    DEFAULT_SNR_DB,
    _normalize,
    _sample_noise_segment,
    _snr_label,
    add_empirical_noise,
    build_real_windows,
)


def _parse_crs(text: str) -> list[int]:
    return [int(p.strip()) for p in text.split(",") if p.strip()]


def _parse_snr_db(text: str) -> list[float | None]:
    out: list[float | None] = []
    for part in text.split(","):
        token = part.strip().lower()
        if not token:
            continue
        out.append(None if token in {"clean", "inf", "none"} else float(token))
    return out


def _parse_run_dirs(text: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        cr_str, path_str = item.split("=", 1)
        out[int(cr_str.strip())] = Path(path_str.strip())
    return out


def _parse_sources(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gain-model", type=Path, default=Path("results/wavelet_denoiser_ppg/gain_model.keras")
    )
    ap.add_argument("--reference-run", type=Path, default=Path("results/ppg_rvq_64hz_08x_golden"))
    ap.add_argument("--n-windows", type=int, default=500)
    ap.add_argument("--crs", type=_parse_crs, default=[2, 4, 8, 16, 32])
    ap.add_argument("--snr-db", type=_parse_snr_db, default=DEFAULT_SNR_DB)
    ap.add_argument("--noise-bank-root", type=Path, default=Path("/home/vscode/datasets"))
    ap.add_argument("--noise-bank-sources", type=_parse_sources, default=["ppg_dalia", "wesad"])
    ap.add_argument("--noise-bank-files", type=int, default=400)
    ap.add_argument("--wavelet", type=str, default="bior4.4")
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--output-stem", type=str, default="ppg_learned_shrink")
    ap.add_argument(
        "--run-dirs", type=_parse_run_dirs, default=None,
        help="Optional override mapping 'cr=path,cr=path' for RVQ run dirs.",
    )
    ap.add_argument("--no-rvq", action="store_true", help="Skip the RVQ golden comparison.")
    ap.add_argument("--with-bayes", action="store_true", help="Include the BayesShrink floor.")
    args = ap.parse_args()

    run_dirs = dict(DEFAULT_RVQ_RUN_DIRS)
    if args.run_dirs:
        run_dirs.update(args.run_dirs)

    out_dir = Path("results/_rvq_vs_spiht_crossover")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Loading {args.n_windows} real PPG windows from {args.reference_run} ...")
    filtered, raw_native, native_snr, cfg = build_real_windows(
        args.n_windows, reference_run=args.reference_run
    )
    frame_size = int(cfg.data.frame_size)
    sample_rate = int(cfg.data.sampling_rate)
    print(
        f"[native-SNR] median={np.median(native_snr):.1f} dB  "
        f"p10..p90=[{np.percentile(native_snr, 10):.1f}, {np.percentile(native_snr, 90):.1f}] dB"
    )

    print(f"[setup] Loading gain model: {args.gain_model}")
    gain_model = keras.models.load_model(args.gain_model)
    coeff_denoiser = as_coeff_denoiser(
        gain_model, frame_size=frame_size, wavelet=args.wavelet, levels=args.levels
    )

    print(f"[setup] Building empirical noise bank from {args.noise_bank_sources} ...")
    bank_files: list[Path] = []
    for src in args.noise_bank_sources:
        bank_files.extend(sorted((args.noise_bank_root / src).glob("*.h5")))
    noise_bank = build_noise_bank_from_h5(
        [str(p) for p in bank_files[: args.noise_bank_files]],
        target_fs=sample_rate, window_size=frame_size, max_segments=5000,
    )
    if noise_bank is None or len(noise_bank) == 0:
        raise RuntimeError("Empirical noise bank is empty.")
    print(f"[setup] Noise bank: {len(noise_bank)} residual segments.")

    columns: list[tuple[str, float | None, np.ndarray]] = [("native", None, raw_native)]
    for snr_db in args.snr_db:
        if snr_db is None:
            columns.append(("clean", None, filtered.copy()))
        else:
            inp = add_empirical_noise(filtered, noise_bank, snr_db, seed=1000 + int(round(snr_db)))
            columns.append((_snr_label(snr_db), snr_db, inp))

    probe_rng = np.random.default_rng(7)
    pure_noise = np.stack(
        [_normalize(_sample_noise_segment(noise_bank, frame_size, probe_rng)) for _ in range(128)]
    ).astype(np.float32)

    summary: dict = {
        "n_windows": args.n_windows, "crs": args.crs,
        "columns": [c[0] for c in columns], "by_cr": {}, "imprinting": {},
    }

    from scripts.sweep_empirical_regime_ppg import _pulse_autocorr_peak

    for cr in args.crs:
        kw = dict(modality="ppg", sample_rate=sample_rate, frame_size=frame_size,
                  target_cr=float(cr), wavelet=args.wavelet, levels=args.levels)
        spiht = SpihtAcCodec(name=f"spiht_{cr}x", **kw)
        bayes = BayesShrinkSpihtCodec(name=f"bayes_{cr}x", **kw) if args.with_bayes else None
        learned = LearnedShrinkSpihtCodec(name=f"learned_{cr}x", coeff_denoiser=coeff_denoiser, **kw)

        rvq = None
        if not args.no_rvq:
            run_dir = run_dirs.get(cr)
            if run_dir is not None and run_dir.exists():
                rvq = RvqCodec.from_run_dir(run_dir, modality="ppg")
            else:
                print(f"  [skip] RVQ {cr}x: no run dir ({run_dir}).")

        print(f"\n=== CR {cr}x ===")
        block: dict = {"columns": [], "spiht": [], "bayes": [], "learned": [], "rvq": []}
        for label, _snr, inp in columns:
            rs = encode_decode_batch(spiht, inp)
            rl = encode_decode_batch(learned, inp)
            ts = float(_prd(filtered, rs).mean())
            tl = float(_prd(filtered, rl).mean())
            tb = float(_prd(filtered, encode_decode_batch(bayes, inp)).mean()) if bayes is not None else float("nan")
            tr = float(_prd(filtered, encode_decode_batch(rvq, inp)).mean()) if rvq is not None else float("nan")
            block["columns"].append(label)
            block["spiht"].append(ts)
            block["bayes"].append(tb)
            block["learned"].append(tl)
            block["rvq"].append(tr)
            rvq_str = f"  RVQ={tr:6.2f} ({tr - ts:+5.2f})" if rvq is not None else ""
            bayes_str = f"  Bayes={tb:6.2f} ({tb - ts:+5.2f})" if bayes is not None else ""
            print(
                f"  {label:>7s}: SPIHT={ts:6.2f}  Hybrid={tl:6.2f} ({tl - ts:+5.2f})"
                f"{rvq_str}{bayes_str}"
            )

        ac_in = float(_pulse_autocorr_peak(pure_noise, sample_rate).mean())
        ac_l = float(_pulse_autocorr_peak(encode_decode_batch(learned, pure_noise), sample_rate).mean())
        imp = {"input": ac_in, "learned": ac_l}
        if rvq is not None:
            imp["rvq"] = float(_pulse_autocorr_peak(encode_decode_batch(rvq, pure_noise), sample_rate).mean())
        summary["imprinting"][f"{cr}x"] = imp
        rvq_imp = f"  RVQ={imp['rvq']:.3f}" if rvq is not None else ""
        print(f"  [imprint] pure-noise pulse-autocorr: input={ac_in:.3f}  Hybrid={ac_l:.3f}{rvq_imp}")
        summary["by_cr"][f"{cr}x"] = block

    json_path = out_dir / f"{args.output_stem}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
