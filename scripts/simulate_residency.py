#!/usr/bin/env python
"""Phase 2 (Option B): reactive-residency simulation over Phase-0 logs.

Replays each run's logged attention through Full / Frozen (FlexiCache-style)
/ Reactive (WakeKV) residency policies, sweeping the per-head page budget,
and reports the miss-rate-vs-memory tradeoff. The headline test (C2): at
matched GPU memory, does reactive miss less than frozen in shifting-role
regimes (long CoT, multi-turn)?

  python scripts/simulate_residency.py runs/<model>/<task> \
      [--budgets 8 16 32 64] [--page-size 16] [--unstable-frac 0.25] [--refresh 16]

No GPU. Writes residency.md into the task dir.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wakekv.residency import sweep, wanted_stream_from_log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--unstable-frac", type=float, default=0.25)
    ap.add_argument("--refresh", type=int, default=16)
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    run_dirs = sorted(d for d in task_dir.iterdir() if (d / "log.npz").exists())
    if not run_dirs:
        sys.exit(f"no runs under {task_dir}")

    # Aggregate stats across runs, keyed by (policy, budget).
    agg: dict[tuple, list] = defaultdict(list)
    total_steps = 0
    for rd in run_dirs:
        data = np.load(rd / "log.npz")
        stream, n_units = wanted_stream_from_log(data["topk_idx"], args.page_size)
        total_steps += len(stream)
        for st in sweep(stream, n_units, args.budgets,
                        unstable_frac=args.unstable_frac, refresh=args.refresh):
            agg[(st.policy, st.budget)].append(st)

    def mean(key, attr):
        return float(np.mean([getattr(s, attr) for s in agg[key]]))

    lines = [f"# Residency simulation — {task_dir}", "",
             f"- runs: {len(run_dirs)}  |  page size: {args.page_size} tokens  |  "
             f"frozen: {int(args.unstable_frac*100)}% heads kept full, "
             f"stable refreshed every {args.refresh} steps", "",
             "Miss = wanted page not GPU-resident (reactive: a fetch stall; "
             "frozen: static classification mis-served). Memory = mean GPU "
             "pages resident across the run.", "",
             "| policy | budget | miss rate | mean resident pages | peak | fetches | stall steps |",
             "|---|---|---|---|---|---|---|"]

    # full first
    fk = ("full", None)
    lines.append(f"| full | - | {mean(fk,'miss_rate'):.3f} | "
                 f"{mean(fk,'mean_resident_pages'):.0f} | "
                 f"{int(mean(fk,'peak_resident_pages'))} | 0 | 0 |")
    for b in args.budgets:
        for pol in ("frozen", "reactive"):
            k = (pol, b)
            lines.append(
                f"| {pol} | {b} | {mean(k,'miss_rate'):.3f} | "
                f"{mean(k,'mean_resident_pages'):.0f} | "
                f"{int(mean(k,'peak_resident_pages'))} | "
                f"{int(mean(k,'fetches'))} | {int(mean(k,'stall_steps'))} |")

    # Verdict: at each budget, does reactive miss less than frozen at <= its memory?
    wins = 0
    for b in args.budgets:
        r, f = ("reactive", b), ("frozen", b)
        if mean(r, "miss_rate") <= mean(f, "miss_rate") and \
           mean(r, "mean_resident_pages") <= mean(f, "mean_resident_pages"):
            wins += 1
    lines += ["", "## Read",
              f"Reactive dominates frozen (<= miss AND <= memory) at "
              f"{wins}/{len(args.budgets)} budgets.",
              "A clean win at matched memory supports C2 (frozen classification "
              "leaves quality/efficiency on the table in shifting regimes). If "
              "frozen wins, the reactive story needs the untested regimes or a "
              "smarter demotion policy — informative either way.",
              "",
              "Caveats: page granularity from logged top-k (not full KV); "
              "stall COUNT is a proxy for stall TIME (Phase 2b measures real "
              "PCIe); scout-scale models. This sim ranks policies, it does not "
              "predict absolute serving throughput."]

    (task_dir / "residency.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {task_dir}/residency.md")


if __name__ == "__main__":
    main()
