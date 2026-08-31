#!/usr/bin/env python
"""Generate the two figures wakekv_mlforsys.tex is missing (both \\NEEDFIG
placeholders): the per-head activity heatmap and the memory-vs-miss-rate
Pareto curves. No GPU needed -- both read only the committed Phase-0 logs
(log.npz) under runs/, the same inputs analyze_phase0.py and
simulate_residency.py already use.

  # Figure 1: per-step per-head activity heatmap (paper's Fig. 1)
  python scripts/make_figures.py heatmap runs/<model>/<cot_task> \\
      --out paper/figures/churn_heatmap.png

  # Figure 2: memory-vs-miss-rate Pareto curves (paper's Fig. 2)
  # one subplot per task dir passed; the paper overlays one NIAH + one CoT combo
  python scripts/make_figures.py pareto runs/<model>/<niah_task> runs/<model>/<cot_task> \\
      --out paper/figures/pareto_curves.png

Run this wherever runs/ actually lives (this repo checkout may not have
it locally -- see paper/README.md); copy the resulting PNGs into
paper/figures/ and swap each \\NEEDFIG{...} block in wakekv_mlforsys.tex
for \\includegraphics.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wakekv.residency import page_score_stream_from_log, sweep, wanted_stream_from_log


def _run_dirs(task_dir: Path) -> list[Path]:
    run_dirs = sorted(d for d in task_dir.iterdir() if (d / "log.npz").exists())
    if not run_dirs:
        sys.exit(f"no runs found under {task_dir}")
    return run_dirs


def cmd_heatmap(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task_dir = Path(args.task_dir)
    run_dirs = _run_dirs(task_dir)
    run_dir = next((d for d in run_dirs if d.name == args.run), run_dirs[0]) \
        if args.run else run_dirs[0]

    data = np.load(run_dir / "log.npz")
    if "copy_paste" not in data:
        sys.exit(f"{run_dir}: log.npz has no copy_paste array (binary retrieval "
                  "score) -- heatmap needs a run instrumented with it")
    cp = data["copy_paste"]
    S = cp.shape[0]
    flat = cp.reshape(S, -1)
    top = np.argsort(flat.sum(0))[::-1][: args.top_n]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.imshow(flat[:, top].T, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("decode step")
    ax.set_ylabel(f"head (top {args.top_n})")
    ax.set_title(f"{task_dir.name} / {run_dir.name}")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out} (run: {run_dir}, {S} steps, top {args.top_n} of {flat.shape[1]} heads)")


def _pareto_points(
    task_dir: Path, budgets: list[int], page_size: int
) -> dict[str, list[tuple[float, float]]]:
    """policy -> sorted [(mean resident pages, miss rate), ...] across `budgets`,
    aggregated (mean) across every run under `task_dir`. Same computation
    simulate_residency.py does internally, factored out here for plotting."""
    run_dirs = _run_dirs(task_dir)
    agg: dict[tuple, list] = defaultdict(list)
    for rd in run_dirs:
        data = np.load(rd / "log.npz")
        stream, n_units = wanted_stream_from_log(data["topk_idx"], page_size)
        scored_stream = n_heads = None
        if "topk_val" in data:
            scored_stream, _ = page_score_stream_from_log(
                data["topk_idx"], data["topk_val"], page_size
            )
            n_heads = int(data["topk_idx"].shape[2])
        for st in sweep(stream, n_units, budgets, scored_stream=scored_stream, n_heads=n_heads):
            agg[(st.policy, st.budget)].append(st)

    def mean(key: tuple, attr: str) -> float | None:
        vals = agg.get(key)
        return float(np.mean([getattr(s, attr) for s in vals])) if vals else None

    points: dict[str, list[tuple[float, float]]] = {}
    for policy in ("frozen", "reactive", "evict"):
        pts = []
        for b in budgets:
            mem = mean((policy, b), "mean_resident_pages")
            miss = mean((policy, b), "miss_rate")
            if mem is not None and miss is not None:
                pts.append((mem, miss))
        points[policy] = sorted(pts)
    return points


def cmd_pareto(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task_dirs = [Path(t) for t in args.task_dirs]
    fig, axes = plt.subplots(1, len(task_dirs), figsize=(4.2 * len(task_dirs), 3.4), squeeze=False)
    axes = axes[0]

    style = {"frozen": ("--", "o"), "reactive": ("-", "s"), "evict": (":", "^")}
    for ax, task_dir in zip(axes, task_dirs):
        points = _pareto_points(task_dir, args.budgets, args.page_size)
        for policy, (ls, marker) in style.items():
            pts = points[policy]
            if not pts:
                continue
            mem, miss = zip(*pts)
            ax.plot(mem, miss, ls, marker=marker, label=policy, linewidth=2.2, markersize=7)
        ax.set_xlabel("mean resident pages", fontsize=13)
        ax.set_ylabel("miss rate", fontsize=13)
        ax.set_title(task_dir.name, fontsize=14)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(fontsize=11)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out} ({len(task_dirs)} panel(s), budgets={args.budgets})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    hp = sub.add_parser("heatmap", help="Figure 1: per-step per-head activity heatmap")
    hp.add_argument("task_dir", help="runs/<model>/<task>")
    hp.add_argument("--run", help="specific run subdir name (default: first found)")
    hp.add_argument("--top-n", type=int, default=12)
    hp.add_argument("--out", default="paper/figures/churn_heatmap.png")
    hp.set_defaults(func=cmd_heatmap)

    pp = sub.add_parser("pareto", help="Figure 2: memory-vs-miss-rate Pareto curves")
    pp.add_argument("task_dirs", nargs="+", help="one or more runs/<model>/<task> dirs, one subplot each")
    pp.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64])
    pp.add_argument("--page-size", type=int, default=16)
    pp.add_argument("--out", default="paper/figures/pareto_curves.png")
    pp.set_defaults(func=cmd_pareto)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
