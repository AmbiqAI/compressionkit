"""Evaluate codecs on structured PPG artifact regimes using role-routing semantics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from compressionkit.configs.artifact_suite import ArtifactSpec, ArtifactSuiteConfig, NoiseBudgetConfig
from compressionkit.evaluation.codec import SpihtAcCodec
from compressionkit.evaluation.rvq_codec import RvqCodec
from compressionkit.playbook.catalog import get_method
from compressionkit.preprocessing.artifact_suite import RoleRoutingAugmenter
from compressionkit.preprocessing.augmentations import build_noise_bank_from_h5

from scripts.sweep_codec_noise_ppg import _prd, encode_decode_batch
from scripts.sweep_empirical_regime_ppg import (
    DEFAULT_RVQ_RUN_DIRS,
    _pulse_autocorr_peak,
    _sample_noise_segment,
    _normalize,
    build_real_windows,
)


DEFAULT_FAMILIES = [
    "baseline_wander",
    "motion",
    "empirical_noise",
    "time_warp",
    "beat_scale",
    "cutout",
    "null_frame",
]
DEFAULT_SEVERITIES = [0.25, 0.50, 0.75, 1.00]
HYBRID_METHOD_IDS = {"bayes_shrink_spiht", "filter_spiht", "learned_shrink_spiht"}
FAMILY_SEED_OFFSETS = {name: index * 1000 for index, name in enumerate(DEFAULT_FAMILIES, start=1)}


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


def _parse_sources(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _artifact_label(family: str, severity: float) -> str:
    return f"{family}@{severity:.2f}"


def _visible_mask(target: np.ndarray) -> np.ndarray:
    return target != 0.0


def _masked_prd(target: np.ndarray, recon: np.ndarray, mask_visible: np.ndarray) -> float:
    vals: list[float] = []
    for tgt, rec, mask in zip(target, recon, mask_visible, strict=False):
        if int(mask.sum()) < 2:
            continue
        tgt_visible = tgt[mask]
        rec_visible = rec[mask]
        num = float(np.linalg.norm(tgt_visible - rec_visible))
        den = float(np.linalg.norm(tgt_visible)) + 1e-12
        vals.append(100.0 * num / den)
    return float(np.mean(vals)) if vals else float("nan")


def _masked_zero_rms(recon: np.ndarray, mask_hidden: np.ndarray) -> float:
    vals: list[float] = []
    for rec, mask in zip(recon, mask_hidden, strict=False):
        if int(mask.sum()) < 1:
            continue
        vals.append(float(np.sqrt(np.mean(rec[mask] ** 2))))
    return float(np.mean(vals)) if vals else float("nan")


def _flush_summary(json_path: Path, summary: dict, *, status: str) -> None:
    summary["status"] = status
    summary["completed_crs"] = list(summary["by_cr"].keys())
    json_path.write_text(json.dumps(summary, indent=2))


def _family_spec(name: str, severity_scale: float) -> ArtifactSpec:
    if name == "baseline_wander":
        return ArtifactSpec(
            name=name,
            role="recover",
            param="amplitude",
            prob=1.0,
            severity_min=0.05,
            severity_max=0.60,
            higher_is_worse=True,
        )
    if name == "motion":
        return ArtifactSpec(
            name=name,
            role="remove",
            param="snr_db",
            prob=1.0,
            severity_min=15.0,
            severity_max=3.0 + 12.0 * (1.0 - severity_scale),
            higher_is_worse=False,
        )
    if name == "empirical_noise":
        return ArtifactSpec(
            name=name,
            role="remove",
            param="snr_db",
            prob=1.0,
            severity_min=15.0,
            severity_max=3.0 + 12.0 * (1.0 - severity_scale),
            higher_is_worse=False,
        )
    if name == "time_warp":
        return ArtifactSpec(
            name=name,
            role="recover",
            param="max_warp_fraction",
            prob=1.0,
            severity_min=0.03,
            severity_max=0.20,
            higher_is_worse=True,
        )
    if name == "beat_scale":
        return ArtifactSpec(
            name=name,
            role="recover",
            param="scale_spread",
            prob=1.0,
            severity_min=0.08,
            severity_max=0.35,
            higher_is_worse=True,
        )
    if name == "cutout":
        return ArtifactSpec(
            name=name,
            role="abstain",
            param="fraction",
            prob=1.0,
            severity_min=0.05,
            severity_max=0.60,
            higher_is_worse=True,
        )
    if name == "null_frame":
        return ArtifactSpec(
            name=name,
            role="abstain",
            param="fraction",
            prob=1.0,
            severity_min=1.0,
            severity_max=1.0,
            higher_is_worse=True,
        )
    raise ValueError(f"Unsupported family: {name}")


def _make_suite(family: str, severity: float, *, sample_rate: int, epsilon: float, noise_bank: np.ndarray | None, seed: int) -> RoleRoutingAugmenter:
    spec = _family_spec(family, severity)
    suite_cfg = ArtifactSuiteConfig(
        enabled=True,
        faithful_all=False,
        sample_rate=sample_rate,
        epsilon=epsilon,
        normalize_after=True,
        artifacts=[spec],
        noise_budget=NoiseBudgetConfig(max_simultaneous=1, min_post_corruption_snr_db=0.0, enforce="scale_down"),
    )
    family_noise_bank = noise_bank if family == "empirical_noise" else None
    return RoleRoutingAugmenter(suite_cfg, noise_bank=family_noise_bank, seed=seed)


def _apply_suite(augmenter: RoleRoutingAugmenter, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    inputs = np.empty_like(windows)
    targets = np.empty_like(windows)
    metas: list[dict] = []
    for i in range(windows.shape[0]):
        res = augmenter.apply_pair(windows[i])
        inputs[i] = np.asarray(res["input"], dtype=np.float32)
        targets[i] = np.asarray(res["target"], dtype=np.float32)
        metas.append(dict(res["meta"]))
    return inputs, targets, metas


def _build_hybrid_codec(method_id: str, *, sample_rate: int, frame_size: int, target_cr: float):
    card = get_method(method_id)
    return card.builder(  # type: ignore[misc]
        sample_rate=sample_rate,
        frame_size=frame_size,
        target_cr=target_cr,
        modality="ppg",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-run", type=Path, default=Path("results/ppg_rvq_64hz_08x_golden"))
    ap.add_argument("--n-windows", type=int, default=200)
    ap.add_argument("--crs", type=_parse_crs, default=[2, 4, 8, 16, 32])
    ap.add_argument("--families", type=_parse_families, default=list(DEFAULT_FAMILIES))
    ap.add_argument("--severities", type=_parse_severities, default=list(DEFAULT_SEVERITIES))
    ap.add_argument("--run-dirs", type=_parse_run_dirs, default=None)
    ap.add_argument("--noise-bank-root", type=Path, default=Path("/home/vscode/datasets"))
    ap.add_argument("--noise-bank-sources", type=_parse_sources, default=["ppg_dalia", "wesad"])
    ap.add_argument("--noise-bank-files", type=int, default=400)
    ap.add_argument("--output-stem", type=str, default="ppg_artifact_regime")
    ap.add_argument(
        "--hybrid-method",
        type=str,
        default="filter_spiht",
        choices=sorted(HYBRID_METHOD_IDS),
    )
    ap.add_argument("--no-rvq", action="store_true")
    args = ap.parse_args()

    run_dirs = dict(DEFAULT_RVQ_RUN_DIRS)
    if args.run_dirs:
        run_dirs.update(args.run_dirs)

    out_dir = Path("results/_rvq_vs_spiht_crossover")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Loading {args.n_windows} real PPG windows ...")
    filtered, raw_native, native_snr, ref_cfg = build_real_windows(
        args.n_windows,
        reference_run=args.reference_run,
    )
    sample_rate = int(ref_cfg.data.sampling_rate)
    frame_size = int(ref_cfg.data.frame_size)

    source_files: list[str] = []
    for source in args.noise_bank_sources:
        source_files.extend(str(p) for p in sorted((args.noise_bank_root / source).glob("*.h5")))
    source_files = source_files[: args.noise_bank_files]
    print(f"[setup] Building empirical PPG noise bank from up to {len(source_files)} files ...")
    noise_bank = build_noise_bank_from_h5(
        source_files,
        target_fs=sample_rate,
        window_size=frame_size,
        max_segments=5000,
    )
    if noise_bank is None or len(noise_bank) == 0:
        raise RuntimeError("Empirical PPG noise bank is empty; widen --noise-bank-sources or --noise-bank-root.")
    print(f"[setup] Noise bank: {len(noise_bank)} residual segments.")

    columns: list[tuple[str, str, float, np.ndarray, np.ndarray, list[dict]]] = []
    for family in args.families:
        for severity_index, severity in enumerate(args.severities):
            seed = FAMILY_SEED_OFFSETS.get(family, 10000) + severity_index
            augmenter = _make_suite(
                family,
                severity,
                sample_rate=sample_rate,
                epsilon=ref_cfg.data.epsilon,
                noise_bank=noise_bank,
                seed=seed,
            )
            inp, target, metas = _apply_suite(augmenter, raw_native)
            columns.append((_artifact_label(family, severity), family, severity, inp, target, metas))

    probe_rng = np.random.default_rng(7)
    pure_probe = np.stack(
        [_normalize(_sample_noise_segment(noise_bank, frame_size, probe_rng)) for _ in range(min(args.n_windows, 128))]
    ).astype(np.float32)

    summary: dict = {
        "source": "artifact-suite",
        "modality": "ppg",
        "hybrid_method": args.hybrid_method,
        "n_windows": args.n_windows,
        "crs": args.crs,
        "columns": [label for label, _family, _severity, _inp, _target, _meta in columns],
        "families": args.families,
        "severities": args.severities,
        "native_snr_db": {
            "median": float(np.median(native_snr)),
            "mean": float(np.mean(native_snr)),
        },
        "noise_bank_size": int(len(noise_bank)),
        "by_cr": {},
        "artifact_probe": {},
    }
    json_path = out_dir / f"{args.output_stem}.json"

    try:
        for cr in args.crs:
            spiht = SpihtAcCodec(
                name=f"spiht_{cr}x",
                modality="ppg",
                sample_rate=sample_rate,
                frame_size=frame_size,
                target_cr=float(cr),
            )
            hybrid = _build_hybrid_codec(
                args.hybrid_method,
                sample_rate=sample_rate,
                frame_size=frame_size,
                target_cr=float(cr),
            )

            rvq = None
            if not args.no_rvq:
                run_dir = run_dirs.get(cr)
                if run_dir is not None and run_dir.exists():
                    rvq = RvqCodec.from_run_dir(run_dir, modality="ppg")
                else:
                    print(f"  [skip] RVQ {cr}x: no run dir ({run_dir}).")

            print(f"\n=== CR {cr}x ===")
            block: dict = {"columns": [], "spiht": [], "hybrid": [], "rvq": [], "metadata": []}
            for label, family, severity, inp, target, metas in columns:
                rs = encode_decode_batch(spiht, inp)
                rh = encode_decode_batch(hybrid, inp)
                role = str(metas[0].get("recover") and "recover" or metas[0].get("remove") and "remove" or metas[0].get("abstain") and "abstain" or "mixed")
                is_abstain = bool(metas[0].get("abstain"))
                mask_visible = _visible_mask(target) if is_abstain else None

                if is_abstain and mask_visible is not None:
                    ts = _masked_prd(target, rs, mask_visible)
                    th = _masked_prd(target, rh, mask_visible)
                else:
                    ts = float(_prd(target, rs).mean())
                    th = float(_prd(target, rh).mean())
                entry_meta = {
                    "family": family,
                    "severity": severity,
                    "role": role,
                    "remove_snr_db_mean": float(
                        np.mean([meta.get("remove_snr_db", float("inf")) for meta in metas])
                    ),
                    "recover_examples": metas[0].get("recover", []),
                    "remove_examples": metas[0].get("remove", []),
                    "abstain_examples": metas[0].get("abstain", []),
                }

                rvq_str = ""
                tr = float("nan")
                if rvq is not None:
                    rr = encode_decode_batch(rvq, inp)
                    tr = _masked_prd(target, rr, mask_visible) if is_abstain and mask_visible is not None else float(_prd(target, rr).mean())
                    rvq_str = f"  RVQ={tr:6.2f} ({tr - ts:+5.2f})" if np.isfinite(tr) and np.isfinite(ts) else f"  RVQ={tr}"

                if is_abstain and mask_visible is not None:
                    hidden_mask = ~mask_visible
                    entry_meta["masked_visible_fraction"] = float(np.mean(mask_visible))
                    entry_meta["spiht_hidden_rms"] = _masked_zero_rms(rs, hidden_mask)
                    entry_meta["hybrid_hidden_rms"] = _masked_zero_rms(rh, hidden_mask)
                    if rvq is not None:
                        entry_meta["rvq_hidden_rms"] = _masked_zero_rms(rr, hidden_mask)

                block["columns"].append(label)
                block["spiht"].append(ts)
                block["hybrid"].append(th)
                block["rvq"].append(tr)
                block["metadata"].append(entry_meta)

                extra = ""
                if is_abstain and mask_visible is not None:
                    extra = (
                        f"  hiddenRMS SPIHT={entry_meta['spiht_hidden_rms']:.3f}"
                        f"  DSP={entry_meta['hybrid_hidden_rms']:.3f}"
                    )
                    if rvq is not None and "rvq_hidden_rms" in entry_meta:
                        extra += f"  RVQ={entry_meta['rvq_hidden_rms']:.3f}"

                print(
                    f"  {label:>22s}: SPIHT={ts:6.2f}  DSP={th:6.2f} ({th - ts:+5.2f}){rvq_str}{extra}"
                )

            probe_block = {
                "input": float(_pulse_autocorr_peak(pure_probe, sample_rate).mean()),
                "spiht": float(_pulse_autocorr_peak(encode_decode_batch(spiht, pure_probe), sample_rate).mean()),
                "hybrid": float(_pulse_autocorr_peak(encode_decode_batch(hybrid, pure_probe), sample_rate).mean()),
            }
            if rvq is not None:
                probe_block["rvq"] = float(
                    _pulse_autocorr_peak(encode_decode_batch(rvq, pure_probe), sample_rate).mean()
                )
            summary["artifact_probe"][f"{cr}x"] = probe_block
            summary["by_cr"][f"{cr}x"] = block
            print(
                f"  [probe] pure-noise pulse-autocorr: input={probe_block['input']:.3f}  "
                f"SPIHT={probe_block['spiht']:.3f}  DSP={probe_block['hybrid']:.3f}"
                + (f"  RVQ={probe_block['rvq']:.3f}" if rvq is not None else "")
            )
            _flush_summary(json_path, summary, status="running")
    except KeyboardInterrupt:
        _flush_summary(json_path, summary, status="interrupted")
        print(f"\nInterrupted; partial results saved to {json_path}")
        return

    _flush_summary(json_path, summary, status="completed")
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
