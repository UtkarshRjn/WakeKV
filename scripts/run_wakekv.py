#!/usr/bin/env python
"""Run FlexiCache-in-vLLM with WakeKV reactive residency installed.

Wraps FlexiCache's benchmark harness (or ``vllm serve``): it forces the
WakeKV monkey-patch to be active in the process that initializes vLLM, then
runs whatever command you give after ``--``.

Why not just install-then-exec?
-------------------------------
The shim is an in-process monkey-patch of ``FlexiCacheConfig``. An earlier
version installed it here and then ``os.execvp``'d the target — but
``execvp`` replaces the process image, throwing away the patch, so the target
silently ran **stock** FlexiCache. Instead we exec the target *through*
``python -m wakekv._shimmed_main``, a tiny bootstrap that installs the shim
inside the target's own interpreter (before vLLM initializes) and then hands
off to the real target. The mode/interval travel via environment variables.

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
        python benchmarks/benchmark_throughput.py <args>

The target command must start with a Python interpreter (``python`` /
``python3``) so the shim can be installed in-process. Wrap console scripts as
``python -m <module> ...``.

The ``--rerank-interval`` argument overrides FlexiCache's rerank frequency
regardless of what you pass to vLLM's ``--rerank-frequency`` --- forcing that
value is the whole point of the shim.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _strip_separators(argv: list[str]) -> list[str]:
    """Drop leading ``--`` tokens argparse's REMAINDER leaves in place."""
    argv = list(argv)
    while argv and argv[0] == "--":
        argv.pop(0)
    return argv


def build_child_command(target_argv: list[str]) -> list[str]:
    """Rewrite the user's target into a shim-bootstrapped invocation.

    ``[python, script.py, ...]``  -> ``[python, -m, wakekv._shimmed_main, script.py, ...]``
    ``[python, -m, mod, ...]``    -> ``[python, -m, wakekv._shimmed_main, -m, mod, ...]``

    The bootstrap (``wakekv._shimmed_main``) installs the shim inside the
    target interpreter, then ``runpy``'s the real target, so the patch is live
    before vLLM initializes.

    Returns ``[]`` if no target was supplied. Raises ``ValueError`` if the
    target does not start with a Python interpreter (the shim can only be
    installed in-process, so the target must be a Python invocation).
    """
    target_argv = _strip_separators(target_argv)
    if not target_argv:
        return []

    exe = os.path.basename(target_argv[0])
    if not exe.startswith("python"):
        raise ValueError(
            f"run_wakekv needs the target to start with a Python interpreter "
            f"so the shim can be installed in-process; got {target_argv[0]!r}. "
            f"Wrap console scripts as: python -m <module> ..."
        )

    return [target_argv[0], "-m", "wakekv._shimmed_main", *target_argv[1:]]


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
        "mode. Sweep {1, 2, 4, 8, 16} for the paper's throughput-quality curve.",
    )
    ap.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="Command to run after installing the shim. Use `--` to separate.",
    )
    args = ap.parse_args()

    if args.rerank_interval < 1:
        print(
            f"[run_wakekv] --rerank-interval must be >= 1, got {args.rerank_interval}",
            file=sys.stderr,
        )
        return 2

    try:
        child = build_child_command(args.argv)
    except ValueError as e:
        print(f"[run_wakekv] {e}", file=sys.stderr)
        return 2

    if not child:
        print(
            "[run_wakekv] no command supplied after --; nothing to execute.",
            file=sys.stderr,
        )
        return 0

    # Pass mode/interval to the in-child bootstrap, and make `wakekv` importable
    # in the child regardless of where it's launched from.
    env = os.environ.copy()
    env["WAKEKV_MODE"] = args.mode
    env["WAKEKV_RERANK_INTERVAL"] = str(args.rerank_interval)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + existing_pp if existing_pp else "")

    print(
        f"[run_wakekv] launching target through in-process shim: "
        f"mode={args.mode}, rerank_interval={args.rerank_interval}",
        file=sys.stderr,
    )

    # execvpe (not execvp): the shim now installs INSIDE the child via the
    # bootstrap, so replacing this launcher process is fine and intended.
    os.execvpe(child[0], child, env)


if __name__ == "__main__":
    sys.exit(main())
