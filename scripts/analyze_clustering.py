#!/usr/bin/env python
"""Phase 1 follow-up: are wake-ups temporally CLUSTERED or UNIFORM?

Decides the fork after G1. If per-head wake-ups bunch at shared moments
(e.g. reasoning-phase boundaries where many heads wake together), a single
coarse trigger can prefetch the about-to-be-needed heads with high recall
at low alarm count -- the proactive design the per-head z-score couldn't
support. If wake-ups are spread uniformly in time, reactive fetch-on-demand
is the honest choice.

Method: pool per-head wake events into a per-step histogram (how many heads
wake at each decode step). Measure how concentrated events are in the
busiest 10% of steps, and compare against a NULL where each head's events
are scattered uniformly at random over its run (destroys any real time
structure but keeps each head's event count). Concentration well above the
null = genuine temporal clustering. Fano factor (var/mean of per-step
counts) and max simultaneous co-wake are reported as supporting evidence.

Pure numpy on existing Phase-0 logs. No GPU.

  python scripts/analyze_clustering.py runs/<model>/<task>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wakekv.signals import wake_events


def per_step_counts(events_by_head: list[list[int]], n_steps: int) -> np.ndarray:
    """Histogram: number of heads waking at each decode step."""
    counts = np.zeros(n_steps, dtype=int)
    for ev in events_by_head:
        for t in ev:
            if 0 <= t < n_steps:
                counts[t] += 1
    return counts


def concentration(counts: np.ndarray, top_frac: float = 0.1) -> float:
    """Fraction of all events falling in the busiest ``top_frac`` of steps.

    Uniform-in-time events give ~top_frac; clustering pushes it toward 1.0.
    """
    total = counts.sum()
    if total == 0:
        return float("nan")
    k = max(1, int(round(len(counts) * top_frac)))
    return float(np.sort(counts)[::-1][:k].sum() / total)


def boundary_recall(counts: np.ndarray, top_frac: float, half_window: int) -> float:
    """If we fired a trigger at the busiest ``top_frac`` steps (+- window),
    what fraction of all wake-up events would fall inside a trigger window?

    This is the coarse-trigger recall a boundary-prefetch design would get.
    """
    total = counts.sum()
    if total == 0:
        return float("nan")
    n_steps = len(counts)
    k = max(1, int(round(n_steps * top_frac)))
    trigger_steps = np.argsort(counts)[::-1][:k]
    covered = np.zeros(n_steps, dtype=bool)
    for ts in trigger_steps:
        lo = max(0, ts - half_window)
        hi = min(n_steps, ts + half_window + 1)
        covered[lo:hi] = True
    return float(counts[covered].sum() / total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--top-frac", type=float, default=0.1,
                    help="fraction of steps treated as the busiest 'boundary' steps")
    ap.add_argument("--half-window", type=int, default=4,
                    help="+-steps around a trigger step counted as covered")
    ap.add_argument("--null-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    run_dirs = sorted(d for d in task_dir.iterdir() if (d / "log.npz").exists())
    if not run_dirs:
        sys.exit(f"no runs under {task_dir}")

    rng = np.random.default_rng(args.seed)
    rows, zscores = [], []
    obs_all, null_all, brecall_all, fano_all = [], [], [], []
    total_events = 0

    for rd in run_dirs:
        data = np.load(rd / "log.npz")
        if "needle_score" not in data:
            print(f"[skip] {rd.name}: no needle score")
            continue
        score = data["needle_score"]  # [S, L, H]
        S, L, H = score.shape
        flat = score.reshape(S, L * H)
        events_by_head = [ev for u in range(L * H) if (ev := wake_events(flat[:, u]))]
        counts = per_step_counts(events_by_head, S)
        n_ev = int(counts.sum())
        if n_ev == 0:
            continue
        total_events += n_ev

        obs = concentration(counts, args.top_frac)
        fano = float(counts.var() / counts.mean()) if counts.mean() > 0 else float("nan")
        brecall = boundary_recall(counts, args.top_frac, args.half_window)

        # Null: scatter each head's events uniformly over [0, S).
        null = np.empty(args.null_trials)
        for i in range(args.null_trials):
            nc = np.zeros(S, dtype=int)
            for ev in events_by_head:
                for t in rng.integers(0, S, size=len(ev)):
                    nc[t] += 1
            null[i] = concentration(nc, args.top_frac)
        nmu, nsd = float(null.mean()), float(null.std())
        z = (obs - nmu) / nsd if nsd > 1e-9 else float("nan")

        obs_all.append(obs); null_all.append(nmu)
        brecall_all.append(brecall); fano_all.append(fano); zscores.append(z)
        rows.append(f"| {rd.name} | {S} | {n_ev} | {obs:.2f} | {nmu:.2f} | "
                    f"{z:+.1f} | {fano:.1f} | {int(counts.max())} | {brecall:.2f} |")

    if not rows:
        sys.exit("no wake-up events found in any run")

    mean_obs = float(np.nanmean(obs_all))
    mean_null = float(np.nanmean(null_all))
    mean_z = float(np.nanmean(zscores))
    mean_brecall = float(np.nanmean(brecall_all))
    ratio = mean_obs / mean_null if mean_null else float("nan")

    # Verdict: clustered if observed concentration clears the null decisively
    # AND a coarse trigger would actually catch most events.
    clustered = (mean_z >= 3.0) and (ratio >= 1.5) and (mean_brecall >= 0.6)

    lines = [f"# Wake-up clustering — {task_dir}", "",
             f"- total wake-up events: {total_events}",
             f"- top-frac (boundary steps): {args.top_frac}  |  "
             f"trigger half-window: +-{args.half_window}", "",
             "| run | steps | events | concentration | null | z | Fano | max co-wake | boundary-recall |",
             "|---|---|---|---|---|---|---|---|---|",
             *rows, "",
             f"- mean concentration {mean_obs:.2f} vs null {mean_null:.2f} "
             f"(ratio {ratio:.2f}, z {mean_z:+.1f})",
             f"- mean boundary-trigger recall: {mean_brecall:.2f}", "",
             "## Verdict",
             f"{'CLUSTERED' if clustered else 'NOT clearly clustered'} — "
             + ("wake-ups bunch in time well beyond chance and a coarse "
                "boundary trigger would catch most of them: the proactive "
                "boundary-prefetch design is worth pursuing."
                if clustered else
                "wake-ups are close to uniform-in-time (or a coarse trigger "
                "misses too many): accept the reactive fetch-on-demand design "
                "(plan risk R2). The G1 negative result stands."),
             "",
             "Reading: concentration = share of events in the busiest 10% of "
             "steps (uniform ~= 0.10). z = std devs above the shuffled null. "
             "Fano > 1 = bursty. boundary-recall = events caught if we fired "
             "at those busiest steps +- window."]

    (task_dir / "clustering.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {task_dir}/clustering.md")


if __name__ == "__main__":
    main()
