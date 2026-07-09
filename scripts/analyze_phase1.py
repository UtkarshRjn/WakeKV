#!/usr/bin/env python
"""Phase 1: signal study on Phase-0 logs — gate G1 report.

  python scripts/analyze_phase1.py runs/<model>/<task> \
      [--page-size 16] [--pcie-gbps 21] [--decode-step-ms 30]

For every run with needle scores, extracts per-head wake-up events and
evaluates each candidate signal's precision/recall at lead times
{1,2,4,8,16,32} steps, then compares achievable lead against the transfer
time of a typical promotion (G1: lead must cover the fetch).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wakekv.signals import (
    evaluate_fixed_threshold,
    evaluate_signal,
    page_kv_bytes,
    pooled_threshold,
    signal_drift,
    signal_entropy_trend,
    signal_needle_mass_delta,
    signal_online_rco,
    transfer_steps_needed,
    wake_events,
)


def per_head_page_sets(topk_idx: np.ndarray, unit: int, page_size: int) -> list[set]:
    S, L, H, K = topk_idx.shape
    flat = topk_idx.reshape(S, L * H, K)
    return [
        set((flat[s, unit][flat[s, unit] >= 0] // page_size).tolist()) for s in range(S)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--pcie-gbps", type=float, default=21.0)
    ap.add_argument("--decode-step-ms", type=float, default=30.0)
    ap.add_argument("--max-heads", type=int, default=0, help="0 = only heads with events")
    ap.add_argument("--fixed-quantile", type=float, default=0.9,
                    help="global threshold = this quantile of pooled signal values")
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    run_dirs = sorted(d for d in task_dir.iterdir() if (d / "log.npz").exists())
    if not run_dirs:
        sys.exit(f"no runs under {task_dir}")

    agg: dict[str, list] = defaultdict(list)
    raw: dict[str, list] = defaultdict(list)  # name -> [(signal_array, events)]
    total_events = 0
    for rd in run_dirs:
        data = np.load(rd / "log.npz")
        if "needle_score" not in data:
            print(f"[skip] {rd.name}: no needle score (cot without needle?)")
            continue
        meta = json.loads((rd / "meta.json").read_text())
        score = data["needle_score"]  # [S, L, H]
        S, L, H = score.shape
        flat_score = score.reshape(S, L * H)
        topk_val = data["topk_val"].reshape(S, L * H, -1)
        pool = [max(1, int(c) // args.page_size + 1) for c in data["ctx_len"]]

        units_with_events = []
        for u in range(L * H):
            ev = wake_events(flat_score[:, u])
            if ev:
                units_with_events.append((u, ev))
        total_events += sum(len(ev) for _, ev in units_with_events)
        print(f"[{rd.name}] heads with wake-ups: {len(units_with_events)}, "
              f"events: {sum(len(e) for _, e in units_with_events)}")

        for u, ev in units_with_events[: args.max_heads or None]:
            sets = per_head_page_sets(data["topk_idx"], u, args.page_size)
            sigs = {
                "online_rco": signal_online_rco(sets, pool, k=None),
                "drift": signal_drift(sets),
                "entropy_trend": signal_entropy_trend(topk_val[:, u]),
                "needle_mass_delta": signal_needle_mass_delta(flat_score[:, u]),
            }
            for name, sig in sigs.items():
                for r in evaluate_signal(sig, ev):
                    agg[name].append(r)
                raw[name].append((sig, ev))

    if not agg:
        sys.exit("no wake-up events found — check thresholds or G0 first")

    # Transfer-time bar: promoting one head's full top-k working set.
    kb = page_kv_bytes()
    n_layers = 32  # reported per-layer; actual fetch is per (layer, head)
    steps_bar = transfer_steps_needed(
        pages_to_fetch=64, page_kv_bytes=kb,
        pcie_gbps=args.pcie_gbps, decode_step_ms=args.decode_step_ms,
    )

    lines = [f"# Phase 1 signal study — {task_dir}", "",
             f"- total wake-up events: {total_events}",
             f"- transfer bar: promoting 64 pages x 1 KV head ≈ "
             f"{steps_bar:.2f} decode steps "
             f"(@{args.pcie_gbps} GB/s, {args.decode_step_ms} ms/step)", "",
             "| signal | lead | best precision | recall @ that θ |",
             "|---|---|---|---|"]
    best_by_signal = {}
    for name, results in agg.items():
        by_lead = defaultdict(list)
        for r in results:
            by_lead[r.lead].append(r)
        for lead in sorted(by_lead):
            rs = by_lead[lead]
            # aggregate across heads/runs at matching thresholds: mean P/R
            best = max(
                rs, key=lambda r: (0 if np.isnan(r.precision) else r.precision)
            )
            lines.append(
                f"| {name} | {lead} | {best.precision:.2f} | {best.recall:.2f} |"
            )
            if lead >= steps_bar:
                cur = best_by_signal.get(name)
                if cur is None or best.precision > cur[1]:
                    best_by_signal[name] = (lead, best.precision, best.recall)
    lines += ["", "## G1 gate",
              f"A signal passes if at some lead >= {steps_bar:.1f} steps it keeps "
              "useful precision/recall (judge trade-off; see plan §4)."]
    for name, (lead, p, r) in sorted(best_by_signal.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"- {name}: lead {lead} -> precision {p:.2f}, recall {r:.2f}")
    if not best_by_signal:
        lines.append("- NO signal clears the transfer bar -> fallback: reactive "
                     "promotion + honest stall accounting (plan risk R2).")

    # Honest version: ONE global threshold per signal (what the controller
    # actually uses), events/alarms pooled across all heads — no per-head
    # cherry-picking. This is the table the Phase 2 design should trust.
    lines += ["", f"## Fixed global threshold (quantile {args.fixed_quantile})",
              "One cutoff per signal, applied to every head, pooled P/R. "
              "This is what a runtime controller can actually achieve.", "",
              "| signal | lead | precision | recall | events | alarms |",
              "|---|---|---|---|---|---|"]
    fixed_best = {}
    for name, per_head in raw.items():
        th = pooled_threshold(per_head, args.fixed_quantile)
        res = evaluate_fixed_threshold(per_head, th)
        for lead in sorted(res):
            r = res[lead]
            lines.append(f"| {name} | {lead} | {r.precision:.2f} | {r.recall:.2f} "
                         f"| {r.n_events} | {r.n_alarms} |")
            if lead >= steps_bar and not np.isnan(r.precision):
                cur = fixed_best.get(name)
                # rank by F1 so we don't reward precision at zero recall
                f1 = 0.0 if (r.precision + r.recall) == 0 else \
                    2 * r.precision * r.recall / (r.precision + r.recall)
                if cur is None or f1 > cur[0]:
                    fixed_best[name] = (f1, lead, r.precision, r.recall)
    lines += ["", "Winner under a fixed threshold (best F1 at lead >= bar):"]
    for name, (f1, lead, p, r) in sorted(fixed_best.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"- {name}: lead {lead} -> P {p:.2f}, R {r:.2f} (F1 {f1:.2f})")

    (task_dir / "signal_study.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {task_dir}/signal_study.md")


if __name__ == "__main__":
    main()
