#!/usr/bin/env python
"""Tabulate the M2b-2 WakeKV rerank-interval sweep.

Reads the throughput JSONs written by wakekv_rerank_sweep.sh
(``wakekv-<label>.json``) and prints a markdown table of generation
throughput vs rerank interval, plus each reactive point's speedup relative
to the stock-FlexiCache baseline.

Usage: python scripts/wakekv_sweep_table.py <results_dir>
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys


def _label(path: str) -> str:
    return re.sub(r"^wakekv-|\.json$", "", os.path.basename(path))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: wakekv_sweep_table.py <results_dir>", file=sys.stderr)
        return 2
    results_dir = argv[1]
    files = sorted(glob.glob(os.path.join(results_dir, "wakekv-*.json")))
    if not files:
        print(f"no wakekv-*.json under {results_dir}", file=sys.stderr)
        return 1

    rows = {}
    for f in files:
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        rows[_label(f)] = d

    baseline = rows.get("stock")
    base_tps = baseline.get("output_tokens_per_second") if baseline else None

    def interval_of(label: str) -> float:
        m = re.search(r"reactive-r(\d+)", label)
        return int(m.group(1)) if m else -1

    order = sorted(rows, key=lambda l: (l != "stock", interval_of(l)))

    print(f"\n# M2b-2 — WakeKV reactive rerank-interval sweep ({results_dir})\n")
    print("| config | rerank interval | gen tok/s | elapsed (s) | vs stock |")
    print("|---|---|---:|---:|---:|")
    for label in order:
        d = rows[label]
        tps = d.get("output_tokens_per_second")
        elapsed = d.get("elapsed_time")
        if label == "stock":
            interval, vs = "16 (native)", "1.00× (baseline)"
        else:
            interval = str(interval_of(label))
            vs = f"{tps / base_tps:.2f}×" if (base_tps and tps) else "—"
        tps_s = f"{tps:.1f}" if tps is not None else "—"
        el_s = f"{elapsed:.1f}" if elapsed is not None else "—"
        print(f"| {label} | {interval} | {tps_s} | {el_s} | {vs} |")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
