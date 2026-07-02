#!/usr/bin/env python3
"""Compute a robustness block for golden codec runs under heavier (wearable) noise.

The shipped datasets (PTB-XL, BIDMC, ...) are clinical and relatively clean, but
real wearable deployment is noisier. This evaluator pushes each golden codec
(DSP / Hybrid / AI) through a deep empirical-SNR ladder *and* additive artifact
families, scoring reconstruction vs the filtered clean-truth proxy, plus a
pure-noise imprint probe. Results are written to
``<run_dir>/robustness_metrics.json`` and merged into the quality scorecard by
:func:`compressionkit.evaluation.scorecard.build_quality_scorecard`.

Codecs are rebuilt to match the *deployed* golden exactly:

* ``spiht`` — params from ``<run_dir>/deploy/spiht_config.json``.
* ``hybrid`` — denoiser + SPIHT backend from the registry ``HybridSpec``.
* ``rvq``   — :meth:`RvqCodec.from_run_dir`.

Contaminated inputs are codec-independent, so the per-modality *fixture* (clean
windows, noise bank, condition batches, imprint probes) is built once and reused
across every lane/CR of that modality.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from compressionkit.configs.paths import default_datasets_dir
from compressionkit.evaluation.codec import LearnedShrinkSpihtCodec, SpihtAcCodec
from compressionkit.evaluation.robustness import (
    ImprintProbe,
    RobustnessCondition,
    evaluate_robustness,
)
from compressionkit.evaluation.rvq_codec import RvqCodec
from compressionkit.experiments.registry import get_golden, list_goldens
from compressionkit.models.wavelet_denoiser import as_coeff_denoiser

DEFAULT_SNR_LADDER: list[float] = [12.0, 8.0, 6.0, 3.0, 0.0, -3.0, -6.0, -9.0, -12.0]
DEFAULT_SEVERITIES: list[float] = [0.25, 0.50, 0.75]
ECG_ARTIFACT_FAMILIES = ["colored", "mains", "motion", "lead_off", "weak_leak"]
PPG_ARTIFACT_FAMILIES = ["motion", "baseline_wander"]
RESULTS_ROOT = Path("results")


def _normalize_batch(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-9
    return ((x - mean) / std).astype(np.float32)


@dataclass
class ModalityFixture:
    modality: str
    sample_rate: float
    frame_size: int
    clean: np.ndarray
    conditions: list[RobustnessCondition]
    imprint: list[ImprintProbe]
    morphology_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]] | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def build_ecg_fixture(args: argparse.Namespace) -> ModalityFixture:
    from compressionkit.preprocessing.ecg import build_noise_bank_from_h5
    from compressionkit.synthetic.ecg_contact_artifacts import simulate_contact_artifact_batch
    from scripts.sweep_empirical_regime_ecg import (
        _rr_autocorr_peak,
        _sample_noise_segment,
        add_empirical_noise,
    )
    from scripts.sweep_rvq_vs_spiht_crossover_ecg import _normalize, build_real_windows

    sample_rate = 256.0
    frame_size = 512
    print(f"[ecg] Loading {args.n_windows} PTB-XL windows ...")
    filtered, native, native_snr = build_real_windows(
        args.n_windows,
        frame_size,
        sample_rate,
        data_dir=args.ecg_data_dir,
        glob_pattern=args.ecg_data_glob,
        lead_index=args.ecg_lead_index,
        source_sample_rate=args.ecg_source_sample_rate,
    )
    print(f"[ecg] native SNR median={np.median(native_snr):.1f} dB")
    bank_files = sorted(args.ecg_data_dir.glob(args.ecg_data_glob))[: args.ecg_noise_bank_files]
    noise_bank = build_noise_bank_from_h5(
        bank_files,
        source_sample_rate=args.ecg_source_sample_rate,
        target_sample_rate=int(sample_rate),
        window_size=frame_size,
        lead_index=args.ecg_lead_index,
    )
    if noise_bank is None or len(noise_bank) == 0:
        raise RuntimeError("ECG empirical noise bank is empty.")
    print(f"[ecg] noise bank: {len(noise_bank)} segments")

    conditions = _build_conditions(
        filtered,
        native,
        noise_bank,
        sample_rate,
        args,
        add_empirical_noise=add_empirical_noise,
        artifact_families=ECG_ARTIFACT_FAMILIES,
        artifact_fn=lambda fam, sev, seed: simulate_contact_artifact_batch(
            filtered, family=fam, severity=sev, sample_rate=sample_rate, seed=seed, noise_bank=noise_bank
        ),
        empirical_noise_kwargs={},
    )
    probe_rng = np.random.default_rng(7)
    pure_noise = np.stack(
        [_normalize(_sample_noise_segment(noise_bank, frame_size, probe_rng)) for _ in range(args.imprint_windows)]
    ).astype(np.float32)
    imprint = [ImprintProbe("pure_noise", pure_noise, _rr_autocorr_peak)]
    return ModalityFixture("ecg", sample_rate, frame_size, filtered, conditions, imprint)


def build_ppg_fixture(args: argparse.Namespace) -> ModalityFixture:
    from compressionkit.preprocessing.augmentations import (
        add_baseline_wander,
        add_motion_artifact,
        build_noise_bank_from_h5,
    )
    from scripts.sweep_empirical_regime_ppg import (
        _normalize,
        _pulse_autocorr_peak,
        _sample_noise_segment,
        add_empirical_noise,
        build_real_windows,
    )

    print(f"[ppg] Loading {args.n_windows} windows from {args.ppg_reference_run} ...")
    filtered, native, native_snr, cfg = build_real_windows(args.n_windows, reference_run=args.ppg_reference_run)
    frame_size = int(cfg.data.frame_size)
    sample_rate = float(cfg.data.sampling_rate)
    print(f"[ppg] native SNR median={np.median(native_snr):.1f} dB  frame={frame_size} fs={sample_rate:g}")

    bank_files: list[Path] = []
    for src in args.ppg_noise_bank_sources:
        bank_files.extend(sorted((args.ppg_noise_bank_root / src).glob("*.h5")))
    noise_bank = build_noise_bank_from_h5(
        [str(p) for p in bank_files[: args.ppg_noise_bank_files]],
        target_fs=int(sample_rate),
        window_size=frame_size,
        max_segments=5000,
    )
    if noise_bank is None or len(noise_bank) == 0:
        raise RuntimeError("PPG empirical noise bank is empty.")
    print(f"[ppg] noise bank: {len(noise_bank)} segments")

    def _ppg_artifact(fam: str, sev: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if fam == "motion":
            snr = float(np.interp(sev, [0.0, 1.0], [15.0, 0.0]))
            out = add_motion_artifact(filtered, sample_rate=int(sample_rate), snr_range=(snr, snr), rng=rng)
        elif fam == "baseline_wander":
            amp = float(np.interp(sev, [0.0, 1.0], [0.10, 0.80]))
            out = add_baseline_wander(filtered, sample_rate=int(sample_rate), amplitude_range=(amp, amp), rng=rng)
        else:  # pragma: no cover
            raise ValueError(f"Unknown PPG artifact family {fam!r}")
        return _normalize_batch(np.asarray(out, dtype=np.float32))

    conditions = _build_conditions(
        filtered,
        native,
        noise_bank,
        sample_rate,
        args,
        add_empirical_noise=add_empirical_noise,
        artifact_families=PPG_ARTIFACT_FAMILIES,
        artifact_fn=_ppg_artifact,
        empirical_noise_kwargs={},
    )
    probe_rng = np.random.default_rng(7)
    pure_noise = np.stack(
        [_normalize(_sample_noise_segment(noise_bank, frame_size, probe_rng)) for _ in range(args.imprint_windows)]
    ).astype(np.float32)
    imprint = [ImprintProbe("pure_noise", pure_noise, _pulse_autocorr_peak)]

    # SpO2 proxy: AC pulse-amplitude / shape / dicrotic-notch fidelity vs clean truth.
    from compressionkit.evaluation.ppg_morphology import evaluate_ppg_morphology

    def _ppg_morphology(clean_arr: np.ndarray, recon_arr: np.ndarray) -> dict[str, float]:
        m = evaluate_ppg_morphology(clean_arr, recon_arr, sample_rate=int(sample_rate))
        ac = m.get("ac_amplitude_norm", {})
        orig_mean = ac.get("orig", {}).get("mean")
        recon_mean = ac.get("recon", {}).get("mean")
        abs_err = ac.get("abs_delta", {}).get("mean")
        out: dict[str, float] = {}
        if orig_mean:
            out["ac_ratio"] = float(recon_mean / orig_mean)  # <1 => AC shrink (SpO2 bias risk)
            out["ac_abs_err_pct"] = float(abs_err / orig_mean * 100.0)
        shape = m.get("pulse_shape_correlation", {}).get("mean")
        if shape is not None:
            out["shape_corr"] = float(shape)
        notch = m.get("dicrotic_notch", {}).get("agreement")
        if notch is not None:
            out["notch_agreement"] = float(notch)
        out["n_pulses_matched"] = float(m.get("num_pulses_matched", 0))
        return out

    return ModalityFixture("ppg", sample_rate, frame_size, filtered, conditions, imprint, morphology_fn=_ppg_morphology)


def _build_conditions(
    filtered: np.ndarray,
    native: np.ndarray,
    noise_bank: np.ndarray,
    sample_rate: float,
    args: argparse.Namespace,
    *,
    add_empirical_noise,
    artifact_families: list[str],
    artifact_fn,
    empirical_noise_kwargs: dict,
) -> list[RobustnessCondition]:
    conditions: list[RobustnessCondition] = [
        RobustnessCondition("clean", "reference", filtered.copy(), level=None),
        RobustnessCondition("native", "reference", native.copy(), level=None),
    ]
    for snr in args.snr_ladder:
        inp = add_empirical_noise(filtered, noise_bank, snr, seed=2000 + round(snr), **empirical_noise_kwargs)
        conditions.append(RobustnessCondition(f"snr_{snr:g}db", "empirical_snr", inp, level=float(snr)))
    if not args.skip_artifacts:
        for fi, fam in enumerate(artifact_families):
            for si, sev in enumerate(args.severities):
                seed = 5000 + fi * 100 + si
                inp = artifact_fn(fam, sev, seed)
                conditions.append(
                    RobustnessCondition(f"{fam}@{sev:.2f}", "artifact", inp, level=float(sev), family=fam)
                )
    return conditions


# ---------------------------------------------------------------------------
# Codec rebuild
# ---------------------------------------------------------------------------
def _build_codec(exp, run_dir: Path, fixture: ModalityFixture):
    if exp.method == "spiht":
        cfg_path = run_dir / "deploy" / "spiht_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing SPIHT config for {exp.experiment_id}: {cfg_path}")
        cfg = json.loads(cfg_path.read_text())
        return SpihtAcCodec(
            name=f"spiht_{exp.compression_ratio}x",
            modality=exp.modality,
            sample_rate=int(cfg["sample_rate"]),
            frame_size=int(cfg["frame_size"]),
            target_cr=float(cfg["target_cr"]),
            wavelet=cfg["wavelet"],
            levels=int(cfg["levels"]),
        )
    if exp.method == "hybrid":
        import keras

        spec = exp.hybrid
        gain_model = keras.models.load_model(spec.denoiser_path)
        coeff_denoiser = as_coeff_denoiser(
            gain_model, frame_size=fixture.frame_size, wavelet=spec.wavelet, levels=spec.levels
        )
        return LearnedShrinkSpihtCodec(
            name=f"hybrid_{exp.compression_ratio}x",
            modality=exp.modality,
            sample_rate=int(fixture.sample_rate),
            frame_size=fixture.frame_size,
            target_cr=float(exp.compression_ratio),
            wavelet=spec.wavelet,
            levels=spec.levels,
            coeff_denoiser=coeff_denoiser,
        )
    if exp.method == "rvq":
        return RvqCodec.from_run_dir(run_dir, modality=exp.modality)
    raise ValueError(f"Unsupported method {exp.method!r}")


def _select_experiments(args: argparse.Namespace) -> list:
    if args.experiment_id:
        return [get_golden(args.experiment_id)]
    exps = list_goldens()
    if args.modality:
        exps = [e for e in exps if e.modality == args.modality]
    if args.methods:
        exps = [e for e in exps if e.method in args.methods]
    if args.crs:
        exps = [e for e in exps if e.compression_ratio in args.crs]
    return [e for e in exps if e.structure == "codec"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-id", type=str, default=None, help="Single golden id; overrides filters.")
    ap.add_argument("--modality", type=str, default=None, choices=["ecg", "ppg"])
    ap.add_argument("--methods", type=lambda s: [p.strip() for p in s.split(",") if p.strip()], default=None)
    ap.add_argument("--crs", type=lambda s: [int(p) for p in s.split(",") if p.strip()], default=None)
    ap.add_argument("--n-windows", type=int, default=500)
    ap.add_argument("--imprint-windows", type=int, default=128)
    ap.add_argument(
        "--snr-ladder", type=lambda s: [float(p) for p in s.split(",") if p.strip()], default=DEFAULT_SNR_LADDER
    )
    ap.add_argument(
        "--severities", type=lambda s: [float(p) for p in s.split(",") if p.strip()], default=DEFAULT_SEVERITIES
    )
    ap.add_argument("--skip-artifacts", action="store_true")
    ap.add_argument("--skip-imprint", action="store_true")
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    # ECG fixture args
    ap.add_argument("--ecg-data-dir", type=Path, default=Path("datasets/ptbxl"))
    ap.add_argument("--ecg-data-glob", type=str, default="*.h5")
    ap.add_argument("--ecg-lead-index", type=int, default=1)
    ap.add_argument("--ecg-source-sample-rate", type=int, default=500)
    ap.add_argument("--ecg-noise-bank-files", type=int, default=300)
    # PPG fixture args
    ap.add_argument("--ppg-reference-run", type=Path, default=Path("results/ppg_rvq_64hz_08x_golden"))
    ap.add_argument("--ppg-noise-bank-root", type=Path, default=Path(default_datasets_dir()))
    ap.add_argument(
        "--ppg-noise-bank-sources",
        type=lambda s: [p.strip() for p in s.split(",") if p.strip()],
        default=["ppg_dalia", "wesad"],
    )
    ap.add_argument("--ppg-noise-bank-files", type=int, default=400)
    args = ap.parse_args()

    experiments = _select_experiments(args)
    if not experiments:
        print("No matching golden experiments.")
        return
    by_modality: dict[str, list] = {}
    for exp in experiments:
        run_dir = args.results_root / exp.run_name
        if not run_dir.exists():
            print(f"[skip] {exp.experiment_id}: no run dir {run_dir}")
            continue
        by_modality.setdefault(exp.modality, []).append((exp, run_dir))

    fixtures: dict[str, ModalityFixture] = {}
    for modality, items in by_modality.items():
        print(f"\n========== {modality.upper()} fixture ({len(items)} runs) ==========")
        fixture = build_ecg_fixture(args) if modality == "ecg" else build_ppg_fixture(args)
        if args.skip_imprint:
            fixture.imprint = []
        fixtures[modality] = fixture
        for exp, run_dir in items:
            print(f"\n--- robustness: {exp.experiment_id} ({exp.method}) ---")
            try:
                codec = _build_codec(exp, run_dir, fixture)
            except Exception as err:
                print(f"  [error] cannot build codec: {err}")
                continue
            record = evaluate_robustness(
                codec,
                fixture.clean,
                fixture.conditions,
                sample_rate=fixture.sample_rate,
                imprint=fixture.imprint or None,
                morphology_fn=fixture.morphology_fn,
            )
            record["experiment_id"] = exp.experiment_id
            record["run_name"] = exp.run_name
            record["method"] = exp.method
            record["compression_ratio"] = exp.compression_ratio
            out_path = run_dir / "robustness_metrics.json"
            out_path.write_text(json.dumps(record, indent=2))
            hd = record.get("headline", {})
            es = hd.get("empirical_snr", {})
            print(
                f"  clean PRD={record['reference'].get('clean', {}).get('prd', {}).get('mean', float('nan')):.2f}"
                f"  native PRD={record['reference'].get('native', {}).get('prd', {}).get('mean', float('nan')):.2f}"
                f"  PRD@0dB={es.get('prd_at_0db', float('nan')):.2f}"
                f"  PRD@-6dB={es.get('prd_at_-6db', float('nan')):.2f}"
                f"  robustSNR@PRD10={es.get('robust_snr_db_at_prd_threshold')}"
            )
            if record.get("imprint"):
                print(f"  imprint out-autocorr={hd.get('imprint_output_autocorr_max', float('nan')):.3f}")
            morph = hd.get("morphology", {})
            if morph:
                clean_m = morph.get("clean", {})
                m0 = morph.get("at_0db", {})
                print(
                    "  SpO2-proxy AC ratio "
                    f"clean={clean_m.get('ac_ratio', float('nan')):.3f} "
                    f"@0dB={m0.get('ac_ratio', float('nan')):.3f}"
                    f"  shapeCorr clean={clean_m.get('shape_corr', float('nan')):.3f} "
                    f"@0dB={m0.get('shape_corr', float('nan')):.3f}"
                )
            print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
