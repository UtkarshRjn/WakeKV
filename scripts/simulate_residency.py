#!/usr/bin/env python
"""Residency simulation over logged attention.

Replays each run through full, frozen (FlexiCache-style), reactive
(WakeKV), and evict (same LRU schedule, destructive demotion), plus
SnapKV, uniform R-KV, and ReasonAlloc. See wakekv/residency.py for what
each baseline simplifies. Sweeps the per-head page budget and reports
miss rate against memory. At matched memory, does reactive miss no more
than frozen? At the same budget, does it miss no more than destructive
eviction, and no more than the three baselines at matched memory?

  python scripts/simulate_residency.py runs/<model>/<task> \
      [--budgets 8 16 32 64] [--page-size 16] [--unstable-frac 0.25] [--refresh 16]
      [--rkv-buffer 128] [--reasonalloc-delta 128] [--reasonalloc-mu 0.25]

No GPU. Writes residency.md into the task dir. The three baseline
policies need topk_val (attention weights, not just page ids) in the log
— every attention log has it (wakekv/instrument.py always records it) — and
are skipped with a warning, per run, if it's missing.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wakekv.residency import page_score_stream_from_log, sweep, wanted_stream_from_log

BASELINE_POLICIES = ("snapkv", "rkv_uniform", "reasonalloc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--unstable-frac", type=float, default=0.25)
    ap.add_argument("--refresh", type=int, default=16)
    ap.add_argument("--rkv-buffer", type=int, default=128,
                    help="uniform R-KV: steps between recompression passes")
    ap.add_argument("--reasonalloc-delta", type=int, default=128,
                    help="ReasonAlloc: steps between per-head reallocation")
    ap.add_argument("--reasonalloc-mu", type=float, default=0.25,
                    help="ReasonAlloc: starvation-floor fraction of a layer's budget")
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    run_dirs = sorted(d for d in task_dir.iterdir() if (d / "log.npz").exists())
    if not run_dirs:
        sys.exit(f"no runs under {task_dir}")

    # Aggregate stats across runs, keyed by (policy, budget).
    agg: dict[tuple, list] = defaultdict(list)
    total_steps = 0
    baselines_skipped = 0
    for rd in run_dirs:
        data = np.load(rd / "log.npz")
        stream, n_units = wanted_stream_from_log(data["topk_idx"], args.page_size)
        total_steps += len(stream)

        scored_stream = n_heads = None
        if "topk_val" in data:
            scored_stream, _ = page_score_stream_from_log(
                data["topk_idx"], data["topk_val"], args.page_size
            )
            n_heads = int(data["topk_idx"].shape[2])  # [S, L, H, K]
        else:
            baselines_skipped += 1
            print(f"[warn] {rd.name}: no topk_val in log, skipping SnapKV/"
                  "R-KV/ReasonAlloc for this run", file=sys.stderr)

        for st in sweep(stream, n_units, args.budgets,
                        scored_stream=scored_stream, n_heads=n_heads,
                        rkv_buffer=args.rkv_buffer,
                        reasonalloc_delta=args.reasonalloc_delta,
                        reasonalloc_mu=args.reasonalloc_mu,
                        unstable_frac=args.unstable_frac, refresh=args.refresh):
            agg[(st.policy, st.budget)].append(st)

    def have(key) -> bool:
        return bool(agg.get(key))

    def mean(key, attr):
        return float(np.mean([getattr(s, attr) for s in agg[key]]))

    def pareto_front(policy: str, budgets) -> list[tuple[float, float]]:
        """`policy`'s own (mean resident pages, miss rate) points across its
        budget sweep, sorted by memory -- the frontier `interp_miss` walks."""
        return sorted(
            (mean((policy, b), "mean_resident_pages"), mean((policy, b), "miss_rate"))
            for b in budgets if have((policy, b))
        )

    def interp_miss(front: list[tuple[float, float]], mem: float) -> float | None:
        """Piecewise-linear interpolation of `front`'s miss rate at memory
        `mem`; None if `mem` falls outside `front`'s own memory range."""
        for (m0, r0), (m1, r1) in zip(front, front[1:]):
            if m0 <= mem <= m1:
                return r0 if m1 == m0 else r0 + (mem - m0) / (m1 - m0) * (r1 - r0)
        return None

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
        for pol in ("frozen", "reactive", "evict", *BASELINE_POLICIES):
            k = (pol, b)
            if not have(k):
                continue  # baseline skipped this budget (no topk_val in any run)
            lines.append(
                f"| {pol} | {b} | {mean(k,'miss_rate'):.3f} | "
                f"{mean(k,'mean_resident_pages'):.0f} | "
                f"{int(mean(k,'peak_resident_pages'))} | "
                f"{int(mean(k,'fetches'))} | {int(mean(k,'stall_steps'))} |")
    if baselines_skipped:
        lines.append(f"\n*({baselines_skipped}/{len(run_dirs)} runs had no "
                     "topk_val — snapkv/rkv_uniform/reasonalloc rows above "
                     "are averaged over the remaining runs only.)*")

    # Verdict via the miss-vs-memory PARETO frontier, NOT same-budget rows:
    # frozen keeps its 'unstable' heads fully resident, so at the same budget
    # label the two policies sit at different memory. The honest comparison
    # interpolates reactive's miss to each frozen point's MEMORY.
    react_front = pareto_front("reactive", args.budgets)

    wins = total = 0
    detail = []
    for b in args.budgets:
        mf = mean(("frozen", b), "mean_resident_pages")
        rf = mean(("frozen", b), "miss_rate")
        ri = interp_miss(react_front, mf)
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
              "Frozen classification keeps a head's role fixed, so a shifting "
              "head is mis-served until the next refresh. A reactive miss is "
              "one recoverable fetch; a frozen miss is a page that stays "
              "unavailable. Reactive trades memory for stalls, not for a "
              "lost history.",
              "",
              "Caveats, as in the paper: the stall count here is not stall "
              "time on real PCIe, and the comparison is against a "
              "FlexiCache-style policy rather than that system's exact code. "
              "Pages come from logged top-k, not a full KV. This ranks "
              "policies on the memory/miss curve; it does not predict "
              "serving throughput."]

    # Reactive (offload) vs evict (destroy), same budget. No interpolation
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
    lines += ["", "## Reversible vs. destructive demotion, same budget",
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

    # Reactive vs. SnapKV, uniform R-KV, and ReasonAlloc. Same nominal
    # budget on both sides; realized memory still differs, so the matched
    # comparison below interpolates onto memory, as the paper does.
    baseline_labels = {
        "snapkv": "SnapKV (frozen prefill selection)",
        "rkv_uniform": "uniform R-KV (importance-ranked destructive pruning)",
        "reasonalloc": "ReasonAlloc (per-head reallocation, destructive)",
    }
    any_baseline_data = False
    for pol in BASELINE_POLICIES:
        rows = [b for b in args.budgets if have(("reactive", b)) and have((pol, b))]
        if not rows:
            continue
        any_baseline_data = True
        wins_b = 0
        detail_b = []
        for b in rows:
            rr = mean(("reactive", b), "miss_rate")
            pr = mean((pol, b), "miss_rate")
            rm = mean(("reactive", b), "mean_resident_pages")
            pm = mean((pol, b), "mean_resident_pages")
            if rr <= pr + 1e-9:
                wins_b += 1
            detail_b.append(
                f"  - budget {b} (~{rm:.0f} vs ~{pm:.0f} pages): "
                f"reactive {rr:.3f} vs {pol} {pr:.3f} "
                f"{'(reactive better)' if rr <= pr else f'({pol} better)'}"
            )
        lines += ["", f"## Reactive vs. {baseline_labels[pol]} — same NOMINAL budget",
                  "NOT apples-to-apples (see matched-memory read below) -- "
                  f"reactive misses <= {pol} at **{wins_b}/{len(rows)}** same-label "
                  "budgets, but the two sides can sit at very different actual "
                  "memory (page counts above). Kept for the raw numbers; trust "
                  "the matched-memory read for the fair comparison.", *detail_b]

    # Matched-MEMORY read per baseline, same logic as the frozen comparison
    # above: interpolate reactive's own (memory, miss) Pareto front onto
    # each baseline's ACTUAL realized memory at its own budget sweep, not
    # the nominal budget label. This is the fair comparison -- the nominal-
    # budget rows above can have the two sides differing by up to ~9x in
    # realized memory (see PR discussion), which the interpolation corrects
    # the same way the frozen-vs-reactive comparison already does.
    any_matched = False
    for pol in BASELINE_POLICIES:
        pol_points = [
            (mean((pol, b), "mean_resident_pages"), mean((pol, b), "miss_rate"))
            for b in args.budgets if have((pol, b))
        ]
        if not pol_points:
            continue
        wins_m = total_m = 0
        detail_m = []
        for pm, pr in pol_points:
            ri = interp_miss(react_front, pm)
            if ri is None:
                continue
            any_matched = True
            total_m += 1
            if ri <= pr + 1e-9:
                wins_m += 1
            detail_m.append(
                f"  - at ~{pm:.0f} pages: {pol} {pr:.3f} vs reactive {ri:.3f} "
                f"{'(reactive better)' if ri <= pr else f'({pol} better)'}"
            )
        if total_m == 0:
            lines += ["", f"## Reactive vs. {baseline_labels[pol]} — matched memory",
                      f"No {pol} operating point fell inside reactive's own "
                      "memory range, so no interpolated comparison is possible "
                      "here (see the nominal-budget numbers above instead)."]
            continue
        lines += ["", f"## Reactive vs. {baseline_labels[pol]} — matched memory",
                  f"Reactive misses <= {pol} at **{wins_m}/{total_m}** matched-"
                  "memory operating points (reactive's miss rate interpolated "
                  f"onto {pol}'s own realized memory at each of its budgets).",
                  *detail_m]
    if any_matched:
        lines += ["", "This is the fair reading of the three-baseline "
                  "comparison: it holds memory constant, the way the "
                  "frozen-versus-reactive read already does, instead of comparing "
                  "at a shared nominal budget label the two sides realize "
                  "very differently."]

    if any_baseline_data:
        lines += ["", "Caveats specific to these three (see "
                  "`wakekv/residency.py`, matching the paper's appendix): "
                  "\"same budget\" is a nominal target "
                  "for rkv_uniform/reasonalloc, not continuous capping like "
                  "reactive/evict -- they only prune at periodic buffer/delta "
                  "boundaries (default every 128 steps), so realized mean "
                  "resident pages (shown above) can run well above the budget "
                  "between prunings; compare the printed page counts, not just "
                  "the budget label. SnapKV's observation window is "
                  "degraded to a single query (this log's step 0), not the "
                  "paper's multi-token window; R-KV/ReasonAlloc's importance-"
                  "vs-redundancy joint score is importance-only, since raw key "
                  "vectors (needed for redundancy) aren't logged; "
                  "ReasonAlloc's offline per-layer budget calibration is "
                  "substituted with an equal split, not the paper's "
                  "Reasoning-Wave allocation."]

    (task_dir / "residency.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {task_dir}/residency.md")


if __name__ == "__main__":
    main()
