"""FlexiCache → WakeKV monkey-patch shim (Phase 2b / M2b-1b).

WakeKV reactive residency turns out to be **FlexiCache with two knobs
changed**. FlexiCache already provides everything we need:
- sparse decode over top-B pages per head (their Triton kernel),
- MinMax-scored promote via ``GPUModelRunner.reload_kv_cache_h2d`` and
  demote via ``KVCacheManager.free_pages_decode_phase``,
- a step gate on ``num_decode_step % rerank_frequency == 0``.

WakeKV reactive is: ``unstable_heads = []`` (every head goes through the
top-B + reservoir path, none is "keep-everything-resident") and
``rerank_frequency = 1`` (rerank every step). That's it. No new kernels,
no new transfer paths. This shim installs those two overrides at
FlexiCache config-init time and leaves the rest of their code untouched
so upstream bug fixes pull cleanly.

Usage
-----
Call *before* the vLLM engine starts, i.e. before FlexiCacheConfig gets
instantiated (which happens at ``vllm/v1/engine/core.py:67``).

::

    import wakekv.flexicache_shim as shim
    shim.install(mode="reactive", rerank_interval=1)

    # ...now start vLLM normally
    from vllm import LLM
    llm = LLM(model=..., enable_flexicache=True, ...)

Modes
-----
- ``"reactive"``: WakeKV. Force ``unstable_heads = []`` and override
  ``rerank_frequency`` with ``rerank_interval``. Sweep the interval to
  trace the paper's Pareto.
- ``"identity"``: correctness check. Patch installs but preserves
  whatever FlexiCache's own config would have chosen — running with this
  mode should produce identical output to stock FlexiCache, proving the
  patching mechanism itself doesn't distort anything.
- ``"off"``: no-op alias for ``uninstall``.

Design notes
------------
- We hook ``FlexiCacheConfig._build`` (not ``initialize``) because that's
  the one place their loaded ``self.unstable_heads`` list is converted
  into the runtime data structures (``layer_to_{stable,unstable}_heads``
  sets + CUDA bitmask tensors) that the hot path actually consults. By
  mutating ``self.unstable_heads`` and ``self.rerank_frequency`` right
  before ``_build`` runs, we get the correct downstream globals for free.
- We do NOT touch ``_populate_globals``, ``reload_kv_cache_h2d``, or
  ``free_pages_decode_phase``. Those are FlexiCache's mechanism, and we
  keep them verbatim so their code owns performance and correctness.
- The shim is idempotent: calling ``install`` twice re-applies with the
  new mode. Calling ``uninstall`` restores the original ``_build``.
- Standalone tests use ``_apply_config_override`` on a mock config
  object; the actual monkey-patch runs only when FlexiCache is importable.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger("wakekv.shim")

# ------------------------------------------------------------------- state
_original_build: Callable[[Any], Any] | None = None
_current_mode: str = "off"
_current_rerank_interval: int = 16
_flexicache_module_name: str = "vllm.v1.flexicache.config"


@dataclass
class ShimReport:
    """Summary of what a single ``_build`` call ended up with. Useful
    to log or assert against in a smoke test after vLLM has initialized."""

    mode: str
    unstable_heads_count_before: int
    unstable_heads_count_after: int
    rerank_frequency_before: int
    rerank_frequency_after: int

    def __str__(self) -> str:
        return (
            f"[wakekv shim] mode={self.mode}  "
            f"unstable heads: {self.unstable_heads_count_before}"
            f" → {self.unstable_heads_count_after}  "
            f"rerank frequency: {self.rerank_frequency_before}"
            f" → {self.rerank_frequency_after}"
        )


_last_report: ShimReport | None = None


# --------------------------------------------------------- public interface
def install(mode: str = "reactive", rerank_interval: int = 1) -> None:
    """Install the FlexiCache monkey-patch.

    Parameters
    ----------
    mode
        - ``"reactive"``: WakeKV. ``unstable_heads`` is cleared, so no head
          is kept fully resident; every head takes the sparse top-B path.
          ``rerank_frequency`` is set to ``rerank_interval``.
        - ``"identity"``: pass-through — keeps whatever FlexiCache loaded.
          Used for the M2b-1b correctness check.
        - ``"off"`` / ``"uninstall"``: remove the patch.
    rerank_interval
        Value forced into ``FlexiCacheConfig.rerank_frequency`` in
        ``reactive`` mode. Sweep {1, 2, 4, 8, 16} for the paper's Pareto.
    """
    if mode in ("off", "uninstall"):
        uninstall()
        return

    if mode not in ("reactive", "identity"):
        raise ValueError(
            f"mode must be 'reactive', 'identity', or 'off'; got {mode!r}"
        )
    if rerank_interval < 1:
        raise ValueError(f"rerank_interval must be >= 1, got {rerank_interval}")

    global _current_mode, _current_rerank_interval
    _current_mode = mode
    _current_rerank_interval = rerank_interval

    _apply_patch()
    log.info(
        "wakekv shim installed: mode=%s, rerank_interval=%d",
        mode,
        rerank_interval,
    )


def uninstall() -> None:
    """Restore FlexiCache's original ``_build`` method."""
    global _original_build, _current_mode
    if _original_build is None:
        return
    cfg_cls = _import_config_class()
    cfg_cls._build = _original_build
    _original_build = None
    _current_mode = "off"
    log.info("wakekv shim uninstalled")


def last_report() -> ShimReport | None:
    """Report from the most recent ``_build`` call. None if the shim's
    patched ``_build`` hasn't fired yet (i.e., vLLM hasn't init'd)."""
    return _last_report


# ------------------------------------------------------- core patch logic
def _apply_config_override(config: Any) -> ShimReport:
    """Mutate a FlexiCacheConfig instance's fields in place per current
    mode. Extracted so a unit test can hit it against a mock object.

    Called *before* ``_build`` runs, so downstream tensors/globals are
    regenerated from the overridden values.
    """
    unstable_before = list(getattr(config, "unstable_heads", []))
    freq_before = int(getattr(config, "rerank_frequency", -1))

    if _current_mode == "reactive":
        config.unstable_heads = []
        # Keep the scalar consistent with the (now empty) list, else FlexiCache's
        # block-count assertion takes the num_unstable_heads>0 branch and expects
        # the un-rounded total (off-by-one vs the 0-unstable-head allocation).
        config.num_unstable_heads = 0
        config.unstable_heads_portion = 0.0
        config.rerank_frequency = _current_rerank_interval
    elif _current_mode == "identity":
        pass  # keep whatever the config loaded from model_data
    else:
        raise RuntimeError(f"unexpected shim mode {_current_mode!r}")

    unstable_after = list(getattr(config, "unstable_heads", []))
    freq_after = int(getattr(config, "rerank_frequency", -1))

    report = ShimReport(
        mode=_current_mode,
        unstable_heads_count_before=len(unstable_before),
        unstable_heads_count_after=len(unstable_after),
        rerank_frequency_before=freq_before,
        rerank_frequency_after=freq_after,
    )
    global _last_report
    _last_report = report
    return report


def _apply_patch() -> None:
    """Wrap ``FlexiCacheConfig._build`` so our override runs first."""
    global _original_build
    cfg_cls = _import_config_class()

    # Idempotency: if we've already patched, restore the pristine _build
    # before re-wrapping so we don't stack wrappers.
    if _original_build is not None:
        cfg_cls._build = _original_build

    original_build = cfg_cls._build
    _original_build = original_build

    def _patched_build(self, *args, **kwargs):
        report = _apply_config_override(self)
        log.info(str(report))
        return original_build(self, *args, **kwargs)

    _patched_build.__wrapped__ = original_build  # for introspection
    _patched_build._wakekv_shim = True             # sentinel for tests
    cfg_cls._build = _patched_build


def _import_config_class():
    """Locate ``FlexiCacheConfig`` at monkey-patch time.

    Not at module import time — the shim is imported by test code that
    doesn't have FlexiCache installed. We raise a clear error if the
    caller invokes ``install`` in an environment without FlexiCache.
    """
    try:
        module = importlib.import_module(_flexicache_module_name)
    except ImportError as e:
        raise RuntimeError(
            f"Cannot import {_flexicache_module_name!r}. "
            "The wakekv shim needs FlexiCache installed and importable. "
            "Are you running this from a vLLM+FlexiCache environment?"
        ) from e

    try:
        return module.FlexiCacheConfig
    except AttributeError as e:
        raise RuntimeError(
            f"Module {_flexicache_module_name!r} loaded but has no "
            "FlexiCacheConfig class. Has the upstream layout changed? "
            "Update _flexicache_module_name in wakekv/flexicache_shim.py."
        ) from e
