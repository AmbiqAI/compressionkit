"""CLI for the compression playbook.

Wired into the top-level ``compressionkit`` multiplexer as the ``playbook``
subcommand (see :func:`compressionkit.recipes._registry.dispatch`).

Subcommands::

    compressionkit playbook list [--lane L] [--modality M] [--status S] [--with-params]
    compressionkit playbook run --method ID --data PATH [--modality M] [--cr N]
                                [--sample-rate SR] [--frame-size FS] [--plot PATH]
    compressionkit playbook compare --data PATH [--lane L | --methods a,b,c]
                                    [--modality M] [--cr N] [--plot PATH]
"""

from __future__ import annotations

import argparse

from compressionkit.playbook import methods as _methods  # noqa: F401  — registers cards
from compressionkit.playbook.catalog import (
    Faithfulness,
    Status,
    Tier,
    encoder_decoder_params,
    get_method,
    list_methods,
)
from compressionkit.playbook.lanes import LANE_INFO, Lane
from compressionkit.playbook.run import RunResult, load_signal, resolve_defaults, run_method_on_signal


def _cmd_list(args: argparse.Namespace) -> int:
    lane = Lane(args.lane) if args.lane else None
    status = Status(args.status) if args.status else None
    tier = Tier(args.tier) if args.tier else None
    cards = list_methods(lane=lane, modality=args.modality, status=status, tier=tier)
    if not cards:
        print("(no methods match the filter)")
        return 0

    # Group by lane so the baseline -> robust -> experimental story reads top-down.
    by_lane: dict[Lane, list] = {}
    for card in cards:
        by_lane.setdefault(card.lane, []).append(card)

    for lane_key, lane_cards in by_lane.items():
        info = LANE_INFO.get(lane_key)
        title = info.title if info else lane_key.value
        compute = info.compute if info else ""
        print(f"\n=== {title}  ({lane_key.value})  —  {compute} ===")
        header = f"  {'TIER':12s}  {'ID':38s}  {'FAITH':11s}  {'STATUS':12s}  {'CRs':12s}  PARAMS(enc/dec)"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for card in lane_cards:
            crs = ",".join(str(c) for c in card.target_crs) or "-"
            params = "-"
            if args.with_params and card.golden_id:
                pc = encoder_decoder_params(card.golden_id, args.results_root)
                if pc is not None:
                    params = f"{pc[0]:,}/{pc[1]:,}"
            print(
                f"  {card.tier.value:12s}  {card.id:38s}  {card.faithfulness.value:11s}  "
                f"{card.status.value:12s}  {crs:12s}  {params}"
            )
            if args.rationale and card.rationale:
                print(f"      ↳ {card.rationale}")
    return 0


def _print_run_result(res: RunResult) -> None:
    print(
        f"{res.method_id:22s}  faithful_PRD={res.faithful_prd:6.2f}%  "
        f"proxy_PRD={res.proxy_prd:6.2f}%  true_CR={res.true_cr:5.2f}x  "
        f"(frames={res.n_frames}, fs={res.sample_rate}Hz, frame={res.frame_size})"
    )


def _maybe_plot(results: list[RunResult], path: str, max_samples: int = 2000) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available; skipping plot.")
        return

    n = min(max_samples, results[0].original.size)
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 2.4 * len(results)), squeeze=False)
    for ax, res in zip(axes[:, 0], results, strict=True):
        ax.plot(res.original[:n], color="0.6", lw=0.8, label="input")
        ax.plot(res.proxy[:n], color="tab:green", lw=0.8, label="clean proxy")
        ax.plot(res.reconstruction[:n], color="tab:red", lw=0.9, label="reconstruction")
        ax.set_title(
            f"{res.method_id} — faithful {res.faithful_prd:.1f}% / proxy {res.proxy_prd:.1f}% / {res.true_cr:.1f}x",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")


def _cmd_show(args: argparse.Namespace) -> int:
    card = get_method(args.method)
    print(f"{card.display_name}  [{card.id}]")
    print(f"  lane        {card.lane.value}")
    print(f"  tier        {card.tier.value}")
    print(f"  family      {card.family}")
    print(f"  faithfulness{'':1s}{card.faithfulness.value}")
    print(f"  status      {card.status.value}")
    print(f"  modality    {', '.join(card.modality)}")
    print(f"  target CRs  {', '.join(str(c) for c in card.target_crs) or '-'}")
    print(f"  runnable    {card.runnable}")
    stages = card.stages
    if stages is not None:
        pre, tf, enc, ent = stages
        print("  pipeline    preprocess -> transform -> encoder -> entropy")
        print(f"              {pre}  ->  {tf}  ->  {enc}  ->  {ent}")
    if card.summary:
        print(f"  summary     {card.summary}")
    if card.rationale:
        print(f"  rationale   {card.rationale}")
    if card.edge_notes:
        print(f"  edge        {card.edge_notes}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    card = get_method(args.method)
    signal = load_signal(args.data)
    res = run_method_on_signal(
        card,
        signal,
        modality=args.modality,
        target_cr=args.cr,
        sample_rate=args.sample_rate,
        frame_size=args.frame_size,
    )
    _print_run_result(res)
    if args.plot:
        _maybe_plot([res], args.plot)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    if args.methods:
        ids = [m.strip() for m in args.methods.split(",") if m.strip()]
        cards = [get_method(m) for m in ids]
    else:
        lane = Lane(args.lane) if args.lane else None
        cards = [c for c in list_methods(lane=lane, modality=args.modality) if c.runnable]
    cards = [c for c in cards if c.runnable]
    if not cards:
        print("(no runnable methods selected; trained methods need a golden run)")
        return 1

    signal = load_signal(args.data)
    sr, fs = resolve_defaults(args.modality, args.sample_rate, args.frame_size)
    print(f"data: {args.data}  modality={args.modality}  fs={sr}Hz  frame={fs}  target_CR={args.cr}x\n")
    results: list[RunResult] = []
    for card in cards:
        res = run_method_on_signal(
            card,
            signal,
            modality=args.modality,
            target_cr=args.cr,
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
        )
        results.append(res)
        _print_run_result(res)
    if args.plot:
        _maybe_plot(results, args.plot)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from compressionkit.playbook.benchmark import run_benchmark, run_benchmark_grid

    if args.methods:
        method_ids = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        lane = Lane(args.lane) if args.lane else None
        method_ids = [c.id for c in list_methods(lane=lane, modality="ecg") if c.runnable]

    if args.sweep:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
        severities = [float(s) for s in args.severities.split(",") if s.strip()]
        print(
            f"synthetic-truth SWEEP: n={args.n_windows}  target_CR={args.cr}x  "
            f"families={families}  severities={severities}\n"
            "metric = truth_PRD vs true clean ECG (lower=better); SNR = median input SNR (dB)\n"
        )
        grid, snr = run_benchmark_grid(
            method_ids,
            families=families,
            severities=severities,
            n_windows=args.n_windows,
            frame_size=args.frame_size,
            sample_rate=args.sample_rate,
            target_cr=args.cr,
            seed=args.seed,
        )
        if not grid:
            print("(no runnable methods produced results)")
            return 1
        for family in families:
            cols = [(family, s) for s in severities]
            snr_row = "  ".join(f"{snr[c]:>6.1f}" for c in cols)
            print(f"[{family}]  severities {severities}")
            print(f"    {'SNR(dB)':32s}  {snr_row}")
            for method_id, conds in grid.items():
                cells = "  ".join(f"{conds[c].truth_prd:>6.1f}" if c in conds else f"{'-':>6s}" for c in cols)
                print(f"    {method_id:32s}  {cells}")
            print()
        return 0

    print(
        f"synthetic-truth battery: n={args.n_windows}  family={args.family}  severity={args.severity}  "
        f"target_CR={args.cr}x  (ECG {args.sample_rate}Hz, frame={args.frame_size})\n"
    )
    results = run_benchmark(
        method_ids,
        n_windows=args.n_windows,
        frame_size=args.frame_size,
        sample_rate=args.sample_rate,
        target_cr=args.cr,
        family=args.family,
        severity=args.severity,
        seed=args.seed,
    )
    if not results:
        print("(no runnable methods produced results)")
        return 1

    header = f"{'METHOD':40s}  {'truth_PRD':>9s}  {'faithful_PRD':>12s}  {'imprint':>7s}  {'true_CR':>7s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.method_id:40s}  {r.truth_prd:9.2f}  {r.faithful_prd:12.2f}  {r.imprint:7.3f}  {r.true_cr:6.2f}x")
    print(
        "\ntruth_PRD vs the true clean ECG (unbiased) | faithful_PRD vs noisy input | "
        "imprint = invented periodic structure on pure noise (lower=better)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compressionkit playbook", description="Browse and run compression methods.")
    sub = parser.add_subparsers(dest="action", metavar="ACTION", required=True)

    p_list = sub.add_parser("list", help="List catalog methods.")
    p_list.add_argument("--lane", choices=[lane.value for lane in Lane], default=None)
    p_list.add_argument("--modality", choices=["ppg", "ecg"], default=None)
    p_list.add_argument("--status", choices=[s.value for s in Status], default=None)
    p_list.add_argument("--tier", choices=[t.value for t in Tier], default=None)
    p_list.add_argument("--rationale", action="store_true", help="Print the one-line rationale per method.")
    p_list.add_argument("--with-params", action="store_true", help="Load enc/dec param counts for golden cards.")
    p_list.add_argument("--results-root", default="results")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Show a method's full card and pipeline stages.")
    p_show.add_argument("--method", required=True)
    p_show.set_defaults(func=_cmd_show)

    p_run = sub.add_parser("run", help="Run one method on your own signal.")
    p_run.add_argument("--method", required=True)
    p_run.add_argument("--data", required=True, help="Path to .npy/.npz/.csv signal.")
    p_run.add_argument("--modality", choices=["ppg", "ecg"], default="ecg")
    p_run.add_argument("--cr", type=float, default=8.0)
    p_run.add_argument("--sample-rate", type=int, default=None)
    p_run.add_argument("--frame-size", type=int, default=None)
    p_run.add_argument("--plot", default=None, help="Optional PNG path for a reconstruction plot.")
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser("compare", help="Compare several methods on the same signal.")
    p_cmp.add_argument("--data", required=True, help="Path to .npy/.npz/.csv signal.")
    p_cmp.add_argument("--lane", choices=[lane.value for lane in Lane], default=None)
    p_cmp.add_argument("--methods", default=None, help="Comma-separated method ids (overrides --lane).")
    p_cmp.add_argument("--modality", choices=["ppg", "ecg"], default="ecg")
    p_cmp.add_argument("--cr", type=float, default=8.0)
    p_cmp.add_argument("--sample-rate", type=int, default=None)
    p_cmp.add_argument("--frame-size", type=int, default=None)
    p_cmp.add_argument("--plot", default=None, help="Optional PNG path for a reconstruction plot.")
    p_cmp.set_defaults(func=_cmd_compare)

    p_bench = sub.add_parser("benchmark", help="Triple-metric benchmark on synthetic clean-truth ECG (unbiased).")
    p_bench.add_argument("--lane", choices=[lane.value for lane in Lane], default=None)
    p_bench.add_argument("--methods", default=None, help="Comma-separated method ids (overrides --lane).")
    p_bench.add_argument("--n-windows", type=int, default=64)
    p_bench.add_argument("--cr", type=float, default=8.0)
    p_bench.add_argument("--family", default="motion", choices=["colored", "mains", "motion", "lead_off", "weak_leak"])
    p_bench.add_argument("--severity", type=float, default=0.5)
    p_bench.add_argument(
        "--sweep", action="store_true", help="Sweep across --families x --severities and report truth_PRD vs SNR."
    )
    p_bench.add_argument(
        "--families", default="colored,mains,motion,lead_off,weak_leak", help="Comma-separated families for --sweep."
    )
    p_bench.add_argument("--severities", default="0.25,0.5,0.75,1.0", help="Comma-separated severities for --sweep.")
    p_bench.add_argument("--sample-rate", type=int, default=256)
    p_bench.add_argument("--frame-size", type=int, default=512)
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


# Suppress the unused-import lint for the registration side effect.
_ = Faithfulness


if __name__ == "__main__":
    import sys

    sys.exit(main())
