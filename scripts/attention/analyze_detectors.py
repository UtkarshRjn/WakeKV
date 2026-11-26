#!/usr/bin/env python
"""Offline bakeoff of demand detectors on synthetic or logged page stats.

When a log provides page_max and topk_idx only, temporal detectors and
oracle are scored. Quest / centroid need query and key extrema arrays
(``query.npy``, ``kmin.npz``, ``kmax.npz``) next to the run.

  python scripts/attention/analyze_detectors.py runs/<model>/<task> --budget 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wakekv.demand_detector import (
    detect_centroid,
    detect_oracle,
    detect_previous_demand,
    detect_previous_page_max,
    detect_quest,
    detect_random,
    score_selection,
)


def _wanted_pages(idx_row: np.ndarray, page_size: int) -> set[int]:
    valid = idx_row[idx_row >= 0]
    return set((valid // page_size).tolist()) if valid.size else set()


def evaluate_run(run_dir: Path, budget: int, page_size: int, seed: int = 0) -> dict:
    data = np.load(run_dir / "log.npz")
    topk_idx = data["topk_idx"]  # [S,L,H,K]
    page_max = data["page_max"] if "page_max" in data.files else None
    steps, layers, heads, _ = topk_idx.shape
    rng = np.random.default_rng(seed)
    totals = {
        name: {"page_recall": [], "page_precision": [], "latency_s": [], "summary_bytes": []}
        for name in (
            "random", "previous_demand", "previous_page_max",
            "key_centroid", "quest", "oracle_top64",
        )
    }
    has_quest = (run_dir / "kmin.npz").exists() and (run_dir / "query.npy").exists()
    query = np.load(run_dir / "query.npy") if has_quest else None  # [S,L,H,D]
    kmin_all = np.load(run_dir / "kmin.npz") if has_quest else None
    kmax_all = np.load(run_dir / "kmax.npz") if (run_dir / "kmax.npz").exists() else None

    prev_wanted = [[set() for _ in range(heads)] for _ in range(layers)]
    for t in range(steps):
        for layer in range(layers):
            for head in range(heads):
                wanted = _wanted_pages(topk_idx[t, layer, head], page_size)
                # treat all pages seen so far as CPU-eligible except a fake GPU set
                all_pages = set(range(int(np.max(topk_idx[t, layer, head] // page_size)) + 1))
                gpu = set(list(wanted)[: max(1, len(wanted) // 2)]) if wanted else set()
                cpu = all_pages - gpu
                if not cpu:
                    continue
                results = [
                    detect_random(cpu, budget, rng),
                    detect_previous_demand(prev_wanted[layer][head], cpu, budget),
                    detect_oracle(wanted, cpu, budget),
                ]
                if page_max is not None and t > 0:
                    results.append(
                        detect_previous_page_max(page_max[t - 1, layer, head], cpu, budget)
                    )
                if has_quest and query is not None and t < query.shape[0]:
                    q = query[t, layer, head]
                    key = f"l{layer}_h{head}"
                    if kmin_all is not None and key in kmin_all.files:
                        # kmin arrays stored as [P, D] dense
                        kmin = {p: kmin_all[key][p] for p in cpu if p < kmin_all[key].shape[0]}
                        kmax = {p: kmax_all[key][p] for p in cpu if p < kmax_all[key].shape[0]}
                        cents = {p: 0.5 * (kmin[p] + kmax[p]) for p in kmin if p in kmax}
                        results.append(detect_quest(q, kmin, kmax, cpu, budget))
                        results.append(detect_centroid(q, cents, cpu, budget))
                for det in results:
                    metrics = score_selection(det.selected, wanted & cpu)
                    bucket = totals[det.name]
                    bucket["page_recall"].append(metrics["page_recall"])
                    bucket["page_precision"].append(metrics["page_precision"])
                    bucket["latency_s"].append(det.latency_s)
                    bucket["summary_bytes"].append(det.summary_bytes)
                prev_wanted[layer][head] = wanted

    summary = {}
    for name, bucket in totals.items():
        if not bucket["page_recall"]:
            continue
        summary[name] = {
            "page_recall": float(np.mean(bucket["page_recall"])),
            "page_precision": float(np.mean(bucket["page_precision"])),
            "mean_latency_s": float(np.mean(bucket["latency_s"])),
            "mean_summary_bytes": float(np.mean(bucket["summary_bytes"])),
            "n": len(bucket["page_recall"]),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--page-size", type=int, default=16)
    args = ap.parse_args()
    task_dir = Path(args.task_dir)
    run_dirs = sorted(d for d in task_dir.iterdir() if (d / "log.npz").exists())
    if not run_dirs:
        sys.exit(f"no runs under {task_dir}")
    all_sum = []
    for rd in run_dirs:
        all_sum.append(evaluate_run(rd, args.budget, args.page_size))
    # average across runs
    keys = sorted({k for s in all_sum for k in s})
    lines = [f"# Detector bakeoff (C={args.budget})", "",
             "| detector | page recall | precision | latency s | summary bytes |",
             "|---|---:|---:|---:|---:|"]
    merged = {}
    for k in keys:
        vals = [s[k] for s in all_sum if k in s]
        merged[k] = {
            "page_recall": float(np.mean([v["page_recall"] for v in vals])),
            "page_precision": float(np.mean([v["page_precision"] for v in vals])),
            "mean_latency_s": float(np.mean([v["mean_latency_s"] for v in vals])),
            "mean_summary_bytes": float(np.mean([v["mean_summary_bytes"] for v in vals])),
        }
        lines.append(
            f"| {k} | {merged[k]['page_recall']:.3f} | {merged[k]['page_precision']:.3f} | "
            f"{merged[k]['mean_latency_s']:.2e} | {merged[k]['mean_summary_bytes']:.0f} |"
        )
    out_md = task_dir / "detectors.md"
    out_md.write_text("\n".join(lines) + "\n")
    out_json = task_dir / "detectors.json"
    out_json.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
