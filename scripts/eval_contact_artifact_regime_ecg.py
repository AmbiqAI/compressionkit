"""Evaluate codecs on synthetic ECG contact-failure artifact regimes.

Lane naming: the classical bandpass-denoise + SPIHT control (``FilterSpihtCodec``)
is labelled **Bandpass+SPIHT** in all printed output. Its JSON block key remains
``filter`` for backward compatibility with existing result files and downstream
plotters.

Proxy-favouring caveat: Bandpass+SPIHT is partly favoured by construction in this
eval. The reconstruction is scored against ``filtered`` (the bandpassed clean-truth
proxy), and when the codec's ``--filter-low-hz``/``--filter-high-hz`` match that
proxy band it is measuring the ceiling of linear denoising against its own target.
RVQ gets no equivalent prefilter, so head-to-head PRD here understates the neural
lane. Read Bandpass+SPIHT as a DSP upper bound, not a deployment-fair winner.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np

from compressionkit.evaluation.codec import FilterSpihtCodec, LearnedShrinkSpihtCodec, SpihtAcCodec
from compressionkit.evaluation.rvq_codec import RvqCodec
from compressionkit.models.wavelet_denoiser import as_coeff_denoiser
from compressionkit.preprocessing.ecg import build_noise_bank_from_h5
from compressionkit.synthetic.ecg_contact_artifacts import (
    DEFAULT_CONTACT_FAMILIES,
    DEFAULT_CONTACT_SEVERITIES,
    simulate_contact_artifact_batch,
)
from scripts.sweep_codec_noise_ecg import _prd, encode_decode_batch
from scripts.sweep_empirical_regime_ecg import _rr_autocorr_peak
from scripts.sweep_rvq_vs_spiht_crossover_ecg import DEFAULT_RVQ_RUN_DIRS, build_real_windows


def _parse_crs(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_families(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_severities(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_run_dirs(text: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        cr_str, path_str = item.split("=", 1)
        out[int(cr_str.strip())] = Path(path_str.strip())
    return out


def _artifact_label(family: str, severity: float) -> str:
    return f"{family}@{severity:.2f}"


def _flush_summary(json_path: Path, summary: dict, *, status: str) -> None:
    summary["status"] = status
    summary["completed_crs"] = list(summary["by_cr"].keys())
    json_path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gain-model", type=Path, default=Path("results/wavelet_denoiser_ecg_noharm/gain_model.keras"))
    ap.add_argument("--n-windows", type=int, default=200)
    ap.add_argument("--crs", type=_parse_crs, default=[2, 4, 8, 16, 32])
    ap.add_argument("--families", type=_parse_families, default=list(DEFAULT_CONTACT_FAMILIES))
    ap.add_argument("--severities", type=_parse_severities, default=list(DEFAULT_CONTACT_SEVERITIES))
    ap.add_argument("--data-dir", type=Path, default=Path("datasets/ptbxl"))
    ap.add_argument("--data-glob", type=str, default="*.h5")
    ap.add_argument("--lead-index", type=int, default=1)
    ap.add_argument("--source-sample-rate", type=int, default=500)
    ap.add_argument("--noise-bank-files", type=int, default=300)
    ap.add_argument("--wavelet", type=str, default="bior4.4")
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--output-stem", type=str, default="ecg_contact_artifact_regime")
    ap.add_argument("--filter-low-hz", type=float, default=0.5)
    ap.add_argument("--filter-high-hz", type=float, default=40.0)
    ap.add_argument("--filter-order", type=int, default=3)
    ap.add_argument(
        "--run-dirs",
        type=_parse_run_dirs,
        default=None,
        help="Optional override mapping 'cr=path,cr=path' for RVQ run dirs.",
    )
    ap.add_argument("--no-rvq", action="store_true")
    args = ap.parse_args()

    run_dirs = dict(DEFAULT_RVQ_RUN_DIRS)
    if args.run_dirs:
        run_dirs.update(args.run_dirs)

    sample_rate = 256.0
    frame_size = 512
    out_dir = Path("results/_rvq_vs_spiht_crossover")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Loading gain model: {args.gain_model}")
    gain_model = keras.models.load_model(args.gain_model)
    coeff_denoiser = as_coeff_denoiser(gain_model, frame_size=frame_size, wavelet=args.wavelet, levels=args.levels)

    print(f"[setup] Loading {args.n_windows} real PTB-XL windows ...")
    filtered, _raw_native, _native_snr = build_real_windows(
        args.n_windows,
        frame_size,
        sample_rate,
        data_dir=args.data_dir,
        glob_pattern=args.data_glob,
        lead_index=args.lead_index,
        source_sample_rate=args.source_sample_rate,
    )

    bank_files = sorted(args.data_dir.glob(args.data_glob))[: args.noise_bank_files]
    print(f"[setup] Building empirical residual bank from up to {len(bank_files)} files ...")
    noise_bank = build_noise_bank_from_h5(
        bank_files,
        source_sample_rate=args.source_sample_rate,
        target_sample_rate=int(sample_rate),
        window_size=frame_size,
        lead_index=args.lead_index,
    )
    if noise_bank is None or len(noise_bank) == 0:
        raise RuntimeError("Empirical noise bank is empty.")
    print(f"[setup] Residual bank: {len(noise_bank)} segments")

    columns: list[tuple[str, str, float, np.ndarray]] = []
    for family_index, family in enumerate(args.families):
        for severity_index, severity in enumerate(args.severities):
            seed = 1000 + 100 * family_index + severity_index
            inp = simulate_contact_artifact_batch(
                filtered,
                family=family,
                severity=severity,
                sample_rate=sample_rate,
                seed=seed,
                noise_bank=noise_bank,
            )
            columns.append((_artifact_label(family, severity), family, severity, inp))

    pure_probe: dict[str, np.ndarray] = {}
    for severity in (0.5, 1.0):
        pure_probe[f"pure_artifact@{severity:.2f}"] = simulate_contact_artifact_batch(
            filtered,
            family="pure_artifact",
            severity=severity,
            sample_rate=sample_rate,
            seed=9000 + round(100 * severity),
            noise_bank=noise_bank,
        )[:128]

    summary: dict = {
        "n_windows": args.n_windows,
        "crs": args.crs,
        "columns": [label for label, _family, _severity, _inp in columns],
        "families": args.families,
        "severities": args.severities,
        "by_cr": {},
        "artifact_probe": {},
    }
    json_path = out_dir / f"{args.output_stem}.json"

    try:
        for cr in args.crs:
            kw = {
                "modality": "ecg",
                "sample_rate": int(sample_rate),
                "frame_size": frame_size,
                "target_cr": float(cr),
                "wavelet": args.wavelet,
                "levels": args.levels,
            }
            spiht = SpihtAcCodec(name=f"spiht_{cr}x", **kw)
            learned = LearnedShrinkSpihtCodec(name=f"learned_{cr}x", coeff_denoiser=coeff_denoiser, **kw)
            filt = FilterSpihtCodec(
                name=f"filter_{cr}x",
                low_hz=args.filter_low_hz,
                high_hz=args.filter_high_hz,
                order=args.filter_order,
                **kw,
            )

            rvq = None
            if not args.no_rvq:
                run_dir = run_dirs.get(cr)
                if run_dir is not None and run_dir.exists():
                    rvq = RvqCodec.from_run_dir(run_dir, modality="ecg")
                else:
                    print(f"  [skip] RVQ {cr}x: no run dir ({run_dir}).")

            print(f"\n=== CR {cr}x ===")
            block: dict = {"columns": [], "spiht": [], "filter": [], "learned": [], "rvq": []}
            for label, _family, _severity, inp in columns:
                rs = encode_decode_batch(spiht, inp)
                rl = encode_decode_batch(learned, inp)
                rf = encode_decode_batch(filt, inp)
                ts = float(_prd(filtered, rs).mean())
                tl = float(_prd(filtered, rl).mean())
                tf = float(_prd(filtered, rf).mean())
                tr = float(_prd(filtered, encode_decode_batch(rvq, inp)).mean()) if rvq is not None else float("nan")
                block["columns"].append(label)
                block["spiht"].append(ts)
                block["filter"].append(tf)
                block["learned"].append(tl)
                block["rvq"].append(tr)
                rvq_str = f"  RVQ={tr:6.2f} ({tr - ts:+5.2f})" if rvq is not None else ""
                print(
                    f"  {label:>18s}: SPIHT={ts:6.2f}  Bandpass+SPIHT={tf:6.2f} ({tf - ts:+5.2f})  "
                    f"Hybrid={tl:6.2f} ({tl - ts:+5.2f}){rvq_str}"
                )
            summary["by_cr"][f"{cr}x"] = block

            probe_block: dict = {}
            for label, inp in pure_probe.items():
                ps = float(_rr_autocorr_peak(encode_decode_batch(spiht, inp), sample_rate).mean())
                pl = float(_rr_autocorr_peak(encode_decode_batch(learned, inp), sample_rate).mean())
                pf = float(_rr_autocorr_peak(encode_decode_batch(filt, inp), sample_rate).mean())
                entry = {
                    "input": float(_rr_autocorr_peak(inp, sample_rate).mean()),
                    "spiht": ps,
                    "filter": pf,
                    "learned": pl,
                }
                if rvq is not None:
                    entry["rvq"] = float(_rr_autocorr_peak(encode_decode_batch(rvq, inp), sample_rate).mean())
                probe_block[label] = entry
                rvq_probe = f"  RVQ={entry['rvq']:.3f}" if rvq is not None else ""
                print(
                    f"  [probe] {label}: input={entry['input']:.3f}  SPIHT={ps:.3f}  "
                    f"Bandpass+SPIHT={pf:.3f}  Hybrid={pl:.3f}{rvq_probe}"
                )
            summary["artifact_probe"][f"{cr}x"] = probe_block
            _flush_summary(json_path, summary, status="running")
    except KeyboardInterrupt:
        _flush_summary(json_path, summary, status="interrupted")
        print(f"\nInterrupted; partial results saved to {json_path}")
        return

    _flush_summary(json_path, summary, status="completed")
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
