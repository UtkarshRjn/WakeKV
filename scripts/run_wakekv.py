#!/usr/bin/env python
"""Run FlexiCache-in-vLLM with WakeKV reactive residency installed.

Wraps FlexiCache's benchmark harness: installs the WakeKV monkey-patch,
then hands off to whatever vLLM command line you would otherwise use.
The shim MUST be installed before vLLM's engine initialization, so this
script installs first and then imports vLLM.

Examples
--------
Reactive, rerank every step, one benchmark run::

    python scripts/run_wakekv.py --mode reactive --rerank-interval 1 -- \\
        python -m vllm.entrypoints.openai.api_server \\
        --model mistralai/Mistral-7B-Instruct-v0.2 \\
        --enable-flexicache --num-unstable-heads 64 --topK-budget 64 \\
        --unstable-heads-profile-task gov_report

Identity mode (correctness sanity check --- should match stock FlexiCache)::

    python scripts/run_wakekv.py --mode identity -- \\
        <same vLLM command as above>

The ``--rerank-interval`` argument overrides FlexiCache's rerank frequency
regardless of what you pass to vLLM's ``--rerank-frequency`` --- the whole
point of the shim is to force this value.

Anything after ``--`` is executed as a subprocess with the shim already
active in its Python environment.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make wakekv importable regardless of where this script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--mode",
        choices=("reactive", "identity", "off"),
        default="reactive",
        help="Shim mode. reactive = WakeKV; identity = pass-through (correctness "
        "check); off = don't patch anything.",
    )
    ap.add_argument(
        "--rerank-interval",
        type=int,
        default=1,
        help="Value forced into FlexiCacheConfig.rerank_frequency in reactive "
        "mode. Sweep {1, 2, 4, 8, 16} for the M2b-2 Pareto.",
    )
    ap.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="Command to run after installing the shim. Use `--` to separate.",
    )
    args = ap.parse_args()

    # Install the monkey-patch BEFORE importing anything vllm-adjacent.
    from wakekv import flexicache_shim

    flexicache_shim.install(mode=args.mode, rerank_interval=args.rerank_interval)
    print(
        f"[run_wakekv] shim installed: mode={args.mode}, "
        f"rerank_interval={args.rerank_interval}",
        file=sys.stderr,
    )

    argv = list(args.argv)
    while argv and argv[0] == "--":
        argv.pop(0)
    if not argv:
        print(
            "[run_wakekv] no command supplied after --; nothing to execute.",
            file=sys.stderr,
        )
        return 0

    # os.execvp shares our process (and thus the patched module globals).
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    sys.exit(main())
