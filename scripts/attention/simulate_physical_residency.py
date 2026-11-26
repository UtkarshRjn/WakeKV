#!/usr/bin/env python
"""GQA-aware physical-byte residency sweep over logged attention.

  python scripts/attention/simulate_physical_residency.py runs/<model>/<task> \\
      [--budgets 8 16 32 64] [--page-size 16] [--head-dim 128]

Reads n_kv_heads from meta.json when present; otherwise assumes MHA
(n_kv_heads == n_q_heads). Writes physical_residency.md into the task dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wakekv.physical_residency import matched_memory_wins, simulate_physical
from wakekv.residency import wanted_stream_from_log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--unstable-frac", type=float, default=0.25)
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    run_dirs = sorted(d for d in task_dir.iterdir() if (d / "log.npz").exists())
    if not run_dirs:
        sys.exit(f"no runs under {task_dir}")

    agg: dict[tuple, list] = defaultdict(list)
    for rd in run_dirs:
        data = np.load(rd / "log.npz")
        stream, n_units = wanted_stream_from_log(data["topk_idx"], args.page_size)
        n_layers = int(data["topk_idx"].shape[1])
        n_q = int(data["topk_idx"].shape[2])
        meta_path = rd / "meta.json"
        n_kv = n_q
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            n_kv = int(meta.get("n_kv_heads", n_q))
        assert n_units == n_layers * n_q
        for policy in ("full", "frozen", "reactive", "evict"):
            budgets = [None] if policy == "full" else args.budgets
            for b in budgets:
                st = simulate_physical(
                    stream,
                    n_layers=n_layers,
                    n_q_heads=n_q,
                    n_kv_heads=n_kv,
                    policy=policy,
                    budget=b,
                    page_tokens=args.page_size,
                    head_dim=args.head_dim,
                    dtype_bytes=args.dtype_bytes,
                    unstable_frac=args.unstable_frac,
                )
                agg[(st.policy, st.budget)].append(st)

    lines = [
        "# Physical (GQA) residency",
        "",
        f"page_tokens={args.page_size} head_dim={args.head_dim} dtype_bytes={args.dtype_bytes}",
        "",
        "| policy | B | miss | mean pages | peak pages | peak KV bytes | compulsory | reloads |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    flat = []
    for (pol, b), sts in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1] or -1)):
        miss = float(np.mean([s.logical.miss_rate for s in sts]))
        mean_p = float(np.mean([s.mean_resident_pages for s in sts]))
        peak_p = float(np.mean([s.peak_resident_pages for s in sts]))
        peak_b = float(np.mean([s.peak_kv_bytes() for s in sts]))
        comp = float(np.mean([s.compulsory_admissions for s in sts]))
        rel = float(np.mean([s.reloads for s in sts]))
        lines.append(
            f"| {pol} | {b if b is not None else '-'} | {miss:.4f} | {mean_p:.1f} | "
            f"{peak_p:.1f} | {peak_b:.0f} | {comp:.0f} | {rel:.0f} |"
        )
        flat.extend(sts)

    wins = matched_memory_wins(flat, "frozen", "reactive")
    lines += [
        "",
        "## Matched peak-byte reactive vs frozen",
        f"wins={wins['wins']} ties={wins['ties']} losses={wins['losses']}",
        "",
    ]
    out = task_dir / "physical_residency.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
