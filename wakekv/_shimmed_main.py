"""In-process bootstrap: install the WakeKV shim, then run the real target.

Invoked by ``scripts/system/run_wakekv.py`` as::

    python -m wakekv._shimmed_main <script.py | -m module> [args...]

Why this exists
---------------
The WakeKV shim is an *in-process* monkey-patch of ``FlexiCacheConfig``. It
must be active in the same Python interpreter that initializes vLLM's engine.
The old ``run_wakekv.py`` installed the patch and then ``os.execvp``'d the
target — but ``execvp`` replaces the process image, discarding every patched
module in memory, so the target ran **stock** FlexiCache, unpatched.

This bootstrap runs *inside the target's interpreter* (``run_wakekv.py``
execs ``python -m wakekv._shimmed_main <target>``). It installs the shim here,
before the target's own code runs, and then hands off to the real target via
``runpy`` — so the patch is live when the target initializes vLLM. Mode and
rerank interval are read from environment variables set by ``run_wakekv.py``.
"""

from __future__ import annotations

import os
import runpy
import sys


def main() -> int:
    mode = os.environ.get("WAKEKV_MODE", "off")
    interval_raw = os.environ.get("WAKEKV_RERANK_INTERVAL", "1")
    try:
        interval = int(interval_raw)
    except ValueError:
        print(
            f"[wakekv._shimmed_main] invalid WAKEKV_RERANK_INTERVAL={interval_raw!r}",
            file=sys.stderr,
        )
        return 2

    if mode not in ("off", "uninstall"):
        # Import lazily: the target env has FlexiCache; a bare test env may not.
        from wakekv import flexicache_shim

        flexicache_shim.install(mode=mode, rerank_interval=interval)
        print(
            f"[wakekv._shimmed_main] shim installed in-process: "
            f"mode={mode}, rerank_interval={interval}",
            file=sys.stderr,
        )

    # sys.argv here is [<_shimmed_main path>, <target...>]; drop argv[0].
    argv = sys.argv[1:]
    if not argv:
        print("[wakekv._shimmed_main] no target command to run", file=sys.stderr)
        return 0

    if argv[0] == "-m":
        if len(argv) < 2:
            print("[wakekv._shimmed_main] '-m' requires a module name", file=sys.stderr)
            return 2
        module = argv[1]
        # Mimic `python -m module ...`: argv[0] becomes the module, rest are args.
        sys.argv = [module, *argv[2:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        script = argv[0]
        # Mimic `python script.py ...`: argv, and the script's own directory on
        # sys.path[0] (runpy.run_path does NOT add it, unlike the interpreter),
        # so the target's sibling imports resolve.
        sys.argv = list(argv)
        sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
        runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
