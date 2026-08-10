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
        for pol in ("frozen", "reactive", "evict"):
            k = (pol, b)
            lines.append(
                f"| {pol} | {b} | {mean(k,'miss_rate'):.3f} | "
                f"{mean(k,'mean_resident_pages'):.0f} | "
                f"{int(mean(k,'peak_resident_pages'))} | "
                f"{int(mean(k,'fetches'))} | {int(mean(k,'stall_steps'))} |")

    # Verdict via the miss-vs-memory PARETO frontier, NOT same-budget rows:
    # frozen keeps its 'unstable' heads fully resident, so at the same budget
    # label the two policies sit at different memory. The honest comparison
    # interpolates reactive's miss to each frozen point's MEMORY.
    react_front = sorted(
        (mean(("reactive", b), "mean_resident_pages"),
         mean(("reactive", b), "miss_rate")) for b in args.budgets
    )

    def interp_miss(mem: float) -> float | None:
        for (m0, r0), (m1, r1) in zip(react_front, react_front[1:]):
            if m0 <= mem <= m1:
                return r0 if m1 == m0 else r0 + (mem - m0) / (m1 - m0) * (r1 - r0)
        return None

    wins = total = 0
    detail = []
    for b in args.budgets:
        mf = mean(("frozen", b), "mean_resident_pages")
        rf = mean(("frozen", b), "miss_rate")
        ri = interp_miss(mf)
        if ri is None:
            continue
        total += 1
        if ri <= rf + 1e-9:
            wins += 1
        detail.append(f"  - at ~{mf:.0f} pages: frozen {rf:.3f} vs reactive {ri:.3f}"
                      f" {'(reactive better)' if ri <= rf else '(frozen better)'}")

    lines += ["", "## Read (matched-memory Pareto)",
              f"Reactive misses <= frozen at matched memory at "
              f"**{wins}/{total}** frozen operating points.",
              *detail, "",
              "A win here supports C2 (frozen classification leaves quality on "
              "the table in shifting regimes). NOTE the miss asymmetry: a "
              "reactive miss is a paid fetch STALL (quality preserved), a frozen "
              "miss is a quality GAP (page unavailable until rerank). So reactive "
              "trades memory for stalls, not for accuracy.",
              "",
              "Caveats: reactive fetches far more than frozen (see column) — the "
              "stall COUNT here is a proxy for stall TIME, which only Phase 2b "
              "measures on real PCIe. Page granularity from logged top-k (not "
              "full KV); scout-scale models. This sim ranks policies on the "
              "memory/miss frontier; it does not predict serving throughput."]

    # C3: reactive (offload) vs evict (destroy), SAME budget. No interpolation
    # needed here, unlike the frozen comparison above — both policies share
    # the identical LRU cap, so memory is matched by construction, not by
    # interpolating onto the other's operating point.
    c3_wins = c3_total = 0
    c3_detail = []
    for b in args.budgets:
        rr = mean(("reactive", b), "miss_rate")
        er = mean(("evict", b), "miss_rate")
        rm = mean(("reactive", b), "mean_resident_pages")
        em = mean(("evict", b), "mean_resident_pages")
        c3_total += 1
        if rr <= er + 1e-9:
            c3_wins += 1
        c3_detail.append(
            f"  - budget {b} (~{rm:.0f} vs ~{em:.0f} pages): "
            f"reactive {rr:.3f} vs evict {er:.3f} "
            f"{'(reactive better)' if rr <= er else '(evict better)'}"
        )
    lines += ["", "## C3 read (reversible vs. destructive demotion, same budget)",
              f"Reactive (offload) misses <= evict (destroy) at "
              f"**{c3_wins}/{c3_total}** matched budgets.",
              *c3_detail, "",
              "Unlike frozen, reactive and evict share the exact same LRU "
              "eviction schedule — what stays resident and when something "
              "gets pushed out is identical between them, since that's "
              "driven only by demand and the budget cap. The only thing "
              "that differs is what happens to a page AFTER eviction: "
              "reactive can get it back with one paid stall; evict never "
              "gets it back at all. A win here isolates reversibility "
              "itself as the source of the advantage, holding dynamism, "
              "budget, and eviction order fixed on both sides."]

    (task_dir / "residency.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {task_dir}/residency.md")


if __name__ == "__main__":
    main()
