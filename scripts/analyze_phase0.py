#!/usr/bin/env python
"""Phase 0 analysis: churn statistics + G0 gate report from logged runs.

  python scripts/analyze_phase0.py runs/<model>/<task> [--page-size 16] \
      [--score-thresholds 0.1 0.2 0.3]

Produces per-run and aggregate statistics:
- Table-1-style churn stats (binary copy-paste activation): active
  heads/step, unique heads, adjacent-step Jaccard, entropy (2602.11162)
- the same stats under continuous-score thresholds (robustness check for
  the binary-artifact caveat)
- per-head temporal stability (FlexiCache RCO, W=16) on page-level top-k
  sets, plus the unstable-head fraction
- HeteroCache-style drift events vs. the prefill baseline
Writes summary.md and plots into the run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wakekv.metrics import (
    adjacent_jaccard_series,
    drift_events,
    jaccard,
    summarize_niah_run,
    temporal_stability,
)


def page_sets(topk_idx: np.ndarray, page_size: int) -> list[list[set]]:
    """[S, L, H, K] index array -> per-(layer*head) list of per-step page sets."""
    S, L, H, K = topk_idx.shape
    flat = topk_idx.reshape(S, L * H, K)
    return [
        [set((flat[s, u][flat[s, u] >= 0] // page_size).tolist()) for s in range(S)]
        for u in range(L * H)
    ]


def continuous_stats(scores: np.ndarray, thr: float) -> dict:
    active = scores > thr
    return summarize_niah_run(active)


def analyze_run(run_dir: Path, page_size: int, thresholds: list[float]) -> dict:
    data = np.load(run_dir / "log.npz")
    meta = json.loads((run_dir / "meta.json").read_text())
    S, L, H, K = data["topk_idx"].shape
    report: dict = {"run": run_dir.name, "steps": int(S), "meta_task": meta.get("task")}

    if "copy_paste" in data:
        report["binary"] = summarize_niah_run(data["copy_paste"])
        report["continuous"] = {
            str(t): continuous_stats(data["needle_score"], t) for t in thresholds
        }

    # FlexiCache-style temporal stability on page-level top-k sets.
    n_pages_per_step = [max(1, int(c) // page_size + 1) for c in data["ctx_len"]]
    per_unit = page_sets(data["topk_idx"], page_size)
    k_pages = max(1, K // page_size)
    ts_means = []
    window = min(16, S)
    for unit_sets in per_unit:
        ts = temporal_stability(unit_sets, n_pages_per_step, k_pages, window=window)
        if len(ts):
            ts_means.append(float(ts.mean()))
    ts_means = np.array(ts_means)
    report["temporal_stability"] = {
        "mean": float(ts_means.mean()),
        "p10": float(np.percentile(ts_means, 10)),
        "p90": float(np.percentile(ts_means, 90)),
        "unstable_fraction_ts<0.5": float((ts_means < 0.5).mean()),
    }

    # Drift events vs prefill baseline (step 0 is the last prompt token).
    events_per_unit = [
        len(drift_events(unit_sets[1:], unit_sets[0])) for unit_sets in per_unit
    ]
    report["drift"] = {
        "mean_events_per_head": float(np.mean(events_per_unit)),
        "heads_with_any_event_fraction": float(np.mean([e > 0 for e in events_per_unit])),
    }
    return report


def plot_run(run_dir: Path, report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(run_dir / "log.npz")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    if "copy_paste" in data:
        cp = data["copy_paste"]
        S = cp.shape[0]
        flat = cp.reshape(S, -1)
        variated = np.argsort(flat.sum(0))[::-1][:12]
        axes[0].imshow(flat[:, variated].T, aspect="auto", interpolation="nearest")
        axes[0].set_title("Most-active heads: activation over decode steps")
        axes[0].set_xlabel("decode step")
        axes[0].set_ylabel("head (top 12)")
        active_sets = [set(np.flatnonzero(flat[t])) for t in range(S)]
        adj = adjacent_jaccard_series(active_sets)
        axes[1].plot(adj)
        axes[1].set_title("Adjacent-step Jaccard of active head set")
        axes[1].set_xlabel("decode step")
        axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(run_dir / "churn.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", help="runs/<model>/<task>")
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--score-thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    run_dirs = sorted(d for d in task_dir.iterdir() if (d / "log.npz").exists())
    if not run_dirs:
        sys.exit(f"no runs found under {task_dir}")

    reports = []
    for rd in run_dirs:
        rep = analyze_run(rd, args.page_size, args.score_thresholds)
        plot_run(rd, rep)
        reports.append(rep)
        print(json.dumps(rep, indent=2))

    lines = [f"# Phase 0 summary — {task_dir}", ""]
    binary_reports = [r for r in reports if "binary" in r]
    if binary_reports:
        adj = [r["binary"]["adjacent_jaccard_mean"] for r in binary_reports]
        uniq = [r["binary"]["unique_heads"] for r in binary_reports]
        lines += [
            f"- runs: {len(reports)}",
            f"- adjacent-step Jaccard (binary): mean {np.mean(adj):.3f} "
            f"(2602.11162 reported 0.28-0.51 across models)",
            f"- unique active heads: mean {np.mean(uniq):.0f}",
        ]
    ts = [r["temporal_stability"]["unstable_fraction_ts<0.5"] for r in reports]
    dr = [r["drift"]["heads_with_any_event_fraction"] for r in reports]
    lines += [
        f"- unstable-head fraction (TS<0.5, FlexiCache metric): mean {np.mean(ts):.3f}",
        f"- heads with >=1 drift event (HeteroCache metric): mean {np.mean(dr):.3f}",
        "",
        "## G0 gate",
        "PASS requires: adjacent-step Jaccard well below 1.0 AND a non-trivial",
        "unstable/drifting head fraction in the target regimes (see",
        "RESEARCH_PLAN.md section 4). Judge against the numbers above.",
    ]
    (task_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (task_dir / "summary.json").write_text(json.dumps(reports, indent=2))
    print(f"\nwrote {task_dir}/summary.md")


if __name__ == "__main__":
    main()
