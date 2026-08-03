#!/usr/bin/env python
"""Tabulate the M2b-2 WakeKV rerank-interval sweep.

Reads the throughput JSONs written by wakekv_rerank_sweep.sh
(``wakekv-<label>.json``) and prints a markdown table of generation
throughput vs rerank interval, plus each reactive point's speedup relative
to the stock-FlexiCache baseline.

Optionally reads a companion quality directory populated by
wakekv_longbench_check.sh (``wakekv-longbench-<label>.json``) and adds
a per-task F1/Rouge column so quality preservation is visible next to
throughput.

Usage:
    python scripts/wakekv_sweep_table.py <throughput_dir> [--quality-dir DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


def _label(path: str) -> str:
    return re.sub(r"^wakekv-longbench-|^wakekv-|\.json$", "", os.path.basename(path))


def _load_json_dir(directory: str, pattern: str) -> dict[str, dict]:
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    out: dict[str, dict] = {}
    for f in files:
        try:
            out[_label(f)] = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _quality_summary(data: dict) -> tuple[float | None, dict[str, float]]:
    """LongBench JSONs come in a few shapes across FlexiCache versions.
    Try common ones. Returns (mean_score, {task: score})."""
    per_task: dict[str, float] = {}
    if "per_task" in data and isinstance(data["per_task"], dict):
        for task, v in data["per_task"].items():
            score = v.get("f1") if isinstance(v, dict) else v
            if isinstance(score, (int, float)):
                per_task[task] = float(score)
    else:
        for key, v in data.items():
            if key in ("elapsed_time", "output_tokens_per_second", "config", "model"):
                continue
            if isinstance(v, (int, float)):
                per_task[key] = float(v)
            elif isinstance(v, dict) and "f1" in v:
                per_task[key] = float(v["f1"])
    mean = sum(per_task.values()) / len(per_task) if per_task else None
    return mean, per_task


def _interval_of(label: str) -> int:
    m = re.search(r"reactive-r(\d+)", label)
    return int(m.group(1)) if m else -1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("throughput_dir")
    ap.add_argument("--quality-dir", default=None,
                    help="directory with wakekv-longbench-*.json to add a "
                         "quality column")
    args = ap.parse_args(argv[1:])

    tp = _load_json_dir(args.throughput_dir, "wakekv-*.json")
    if not tp:
        print(f"no wakekv-*.json under {args.throughput_dir}", file=sys.stderr)
        return 1

    quality = {}
    if args.quality_dir:
        raw = _load_json_dir(args.quality_dir, "wakekv-longbench-*.json")
        for k, v in raw.items():
            quality[k] = _quality_summary(v)

    base = tp.get("stock")
    base_tps = base.get("output_tokens_per_second") if base else None
    # fullkv (no compression, dense attention) sorts first as the ceiling
    # reference; stock (FlexiCache baseline) next; reactive-r* by interval.
    order = sorted(tp, key=lambda l: (l != "fullkv", l != "stock", _interval_of(l)))

    print(f"\n# M2b-2 — WakeKV reactive sweep ({args.throughput_dir})\n")
    header = "| config | rerank interval | gen tok/s | elapsed (s) | vs stock |"
    sep = "|---|---|---:|---:|---:|"
    if args.quality_dir:
        header += " LongBench mean |"
        sep += "---:|"
    print(header)
    print(sep)
    for label in order:
        d = tp[label]
        tps = d.get("output_tokens_per_second")
        elapsed = d.get("elapsed_time")
        if label == "fullkv":
            interval = "n/a (dense)"
            vs = f"{tps / base_tps:.2f}×" if (base_tps and tps) else "—"
        elif label == "stock":
            interval, vs = "16 (native)", "1.00× (baseline)"
        else:
            interval = str(_interval_of(label))
            vs = f"{tps / base_tps:.2f}×" if (base_tps and tps) else "—"
        tps_s = f"{tps:.1f}" if tps is not None else "—"
        el_s = f"{elapsed:.1f}" if elapsed is not None else "—"
        row = f"| {label} | {interval} | {tps_s} | {el_s} | {vs} |"
        if args.quality_dir:
            q = quality.get(label)
            row += f" {q[0]:.2f} |" if (q and q[0] is not None) else " — |"
        print(row)

    if args.quality_dir and quality:
        print("\n## Per-task LongBench\n")
        # union of all tasks across configs, preserving insertion order
        tasks: list[str] = []
        for _, per in quality.values():
            for t in per:
                if t not in tasks:
                    tasks.append(t)
        print("| config | " + " | ".join(tasks) + " |")
        print("|---" * (len(tasks) + 1) + "|")
        for label in order:
            q = quality.get(label)
            per = q[1] if q else {}
            cells = [f"{per[t]:.2f}" if t in per else "—" for t in tasks]
            print(f"| {label} | " + " | ".join(cells) + " |")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
