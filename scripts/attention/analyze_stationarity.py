#!/usr/bin/env python
"""Drift null, prefix-fit frozen split, and fetch accounting. No GPU.

  python scripts/attention/analyze_stationarity.py runs/<model>/<task> [...]

Writes stationarity.md next to each task and a combined runs/stationarity.md.

Three measurements:
- ever-drifted fraction against a per-head shuffle of page-set order
- frozen miss rate when the unstable split is fit on a prefix of the trace
- reactive fetches and stall steps, and evict's repeat demands
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wakekv.metrics import drift_events
from wakekv.residency import (
    classify_unstable,
    simulate_frozen,
    simulate_reactive,
    wanted_stream_from_log,
)


def page_sets(topk_idx: np.ndarray, page_size: int) -> list[list[set]]:
    S, L, H, K = topk_idx.shape
    flat = topk_idx.reshape(S, L * H, K)
    out = []
    for u in range(L * H):
        sets = []
        for s in range(S):
            idx = flat[s, u]
            sets.append(set((idx[idx >= 0] // page_size).tolist()))
        out.append(sets)
    return out


def drift_summary(per_unit: list[list[set]]) -> dict:
    counts = [len(drift_events(sets[1:], sets[0])) for sets in per_unit if len(sets) > 1]
    if not counts:
        return {"heads": 0, "ever_drifted": float("nan"), "mean_events": float("nan")}
    arr = np.array(counts, dtype=float)
    return {
        "heads": int(arr.size),
        "ever_drifted": float((arr > 0).mean()),
        "mean_events": float(arr.mean()),
    }


def drift_null(per_unit: list[list[set]], trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    fracs = []
    means = []
    usable = [sets for sets in per_unit if len(sets) > 2]
    for _ in range(trials):
        counts = []
        for sets in usable:
            decode = sets[1:]
            order = rng.permutation(len(decode))
            shuffled = [decode[i] for i in order]
            counts.append(len(drift_events(shuffled, sets[0])))
        arr = np.array(counts, dtype=float)
        fracs.append(float((arr > 0).mean()) if arr.size else float("nan"))
        means.append(float(arr.mean()) if arr.size else float("nan"))
    f = np.array(fracs, dtype=float)
    m = np.array(means, dtype=float)
    return {
        "trials": trials,
        "null_ever_drifted_mean": float(np.nanmean(f)),
        "null_ever_drifted_std": float(np.nanstd(f)),
        "null_mean_events": float(np.nanmean(m)),
    }


def evict_repeat(stream, n_units: int, budget: int) -> dict:
    resident = [dict() for _ in range(n_units)]
    destroyed = [set() for _ in range(n_units)]
    hits = repeat = first = 0
    for t, row in enumerate(stream):
        for u, wanted in enumerate(row):
            res = resident[u]
            dead = destroyed[u]
            for p in wanted:
                if p in res:
                    res[p] = t
                    hits += 1
                elif p in dead:
                    repeat += 1
                else:
                    res[p] = t
                    first += 1
            if len(res) > budget:
                excess = len(res) - budget
                oldest = sorted(res, key=res.get)[:excess]
                for p in oldest:
                    del res[p]
                    dead.add(p)
    total = hits + repeat + first
    return {
        "accesses": total,
        "repeat_misses": repeat,
        "repeat_fraction": (repeat / total) if total else float("nan"),
    }


def one_run(path: Path, page_size: int, budgets: list[int], prefix: int, trials: int, seed: int) -> dict:
    data = np.load(path / "log.npz")
    per_unit = page_sets(data["topk_idx"], page_size)
    observed = drift_summary(per_unit)
    null = drift_null(per_unit, trials, seed)
    stream, n_units = wanted_stream_from_log(data["topk_idx"], page_size)
    prefix_n = min(prefix, len(stream))
    rows = []
    for b in budgets:
        reactive = simulate_reactive(stream, n_units, b)
        frozen = simulate_frozen(stream, n_units, b)
        causal_heads = classify_unstable(stream[:prefix_n], n_units, 0.25)
        causal = simulate_frozen(stream, n_units, b, unstable=causal_heads)
        ev = evict_repeat(stream, n_units, b)
        rows.append({
            "budget": b,
            "reactive_miss": reactive.miss_rate,
            "reactive_fetches": reactive.fetches,
            "reactive_stall_steps": reactive.stall_steps,
            "reactive_steps": reactive.steps,
            "frozen_full_miss": frozen.miss_rate,
            "frozen_prefix_miss": causal.miss_rate,
            "evict_repeat_fraction": ev["repeat_fraction"],
            "evict_repeat_misses": ev["repeat_misses"],
        })
    return {"run": path.name, "steps": int(data["topk_idx"].shape[0]), "drift": observed, "null": null, "rows": rows}


def mean_key(reports: list[dict], key: str) -> float:
    return float(np.mean([r["drift"][key] if key in r["drift"] else r["null"][key] for r in reports]))


def write_task(task_dir: Path, reports: list[dict], budgets: list[int]) -> None:
    lines = [
        f"# Stationarity — {task_dir}",
        "",
        f"- runs: {len(reports)}",
        f"- ever drifted: {mean_key(reports, 'ever_drifted'):.3f}",
        f"- mean events per head: {mean_key(reports, 'mean_events'):.2f}",
        f"- shuffle null ever drifted: {mean_key(reports, 'null_ever_drifted_mean'):.3f}"
        f" ± {mean_key(reports, 'null_ever_drifted_std'):.3f}",
        f"- shuffle null mean events: {mean_key(reports, 'null_mean_events'):.2f}",
        "",
        "| budget | reactive miss | fetches | stall steps | frozen (full trace) | frozen (prefix) | evict repeat fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b in budgets:
        subset = [row for r in reports for row in r["rows"] if row["budget"] == b]
        def m(k):
            return float(np.mean([row[k] for row in subset]))
        lines.append(
            f"| {b} | {m('reactive_miss'):.3f} | {m('reactive_fetches'):.0f} | "
            f"{m('reactive_stall_steps'):.0f} | {m('frozen_full_miss'):.3f} | "
            f"{m('frozen_prefix_miss'):.3f} | {m('evict_repeat_fraction'):.3f} |"
        )
    text = "\n".join(lines) + "\n"
    (task_dir / "stationarity.md").write_text(text)
    (task_dir / "stationarity.json").write_text(json.dumps(reports, indent=2))
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dirs", nargs="+")
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--prefix", type=int, default=32)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    combined = ["# Stationarity", ""]
    for raw in args.task_dirs:
        task_dir = Path(raw)
        run_dirs = sorted(p.parent for p in task_dir.rglob("log.npz"))
        if not run_dirs:
            print(f"no log.npz under {task_dir}", file=sys.stderr)
            continue
        reports = [
            one_run(rd, args.page_size, args.budgets, args.prefix, args.trials, args.seed)
            for rd in run_dirs
        ]
        write_task(task_dir, reports, args.budgets)
        combined.append((task_dir / "stationarity.md").read_text())
    if len(combined) > 1:
        out = Path("runs") / "stationarity.md"
        if Path("runs").is_dir():
            out.write_text("\n".join(combined))
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
