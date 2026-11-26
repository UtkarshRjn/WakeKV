"""Tests for KV pool, transfers, runtime adapter, adaptive budgets."""

from __future__ import annotations

import numpy as np
import pytest

from wakekv.adaptive_budget import AdaptiveAllocator, AdaptiveReactiveController
from wakekv.kv_pool import KvPool
from wakekv.profiling import Profiler
from wakekv.runtime_adapter import RuntimeConfig, WakeKVRuntime
from wakekv.transfer_manager import TransferManager


def test_kv_pool_capacity_and_protection():
    pool = KvPool(2)
    assert pool.admit(0, 1) is not None
    assert pool.admit(0, 2) is not None
    assert pool.admit(0, 3) is None
    pool.protect(0, 1)
    with pytest.raises(RuntimeError):
        pool.release(0, 1)
    pool.unprotect(0, 1)
    pool.release(0, 1)
    assert pool.admit(0, 3) is not None


def test_transfer_manager_fetch_and_demote():
    pool = KvPool(2)
    tm = TransferManager(pool, page_bytes=100)
    moved = tm.fetch_batch([(0, 1), (0, 2), (0, 3)])
    assert len(moved) == 2
    assert tm.stats.h2d_pages == 2
    tm.end_attention(moved)
    demoted = tm.demote_batch(moved)
    assert len(demoted) == 2
    assert pool.occupancy == 0


def test_runtime_step_with_oracle_wanted():
    cfg = RuntimeConfig(
        n_units=2, budget=2, pool_capacity=4, page_bytes=64, fetch_candidates=2
    )
    rt = WakeKVRuntime(cfg)
    # seed reservoir so detector/oracle has something to fetch
    rt.controller._reservoir[0].add(5)
    rt.controller._reservoir[1].add(7)
    out = rt.step(0, queries={}, oracle_wanted={0: {5}, 1: {7}})
    assert out["fetched"] >= 1
    assert "miss_rate" in out


def test_adaptive_allocator_sums_to_total():
    alloc = AdaptiveAllocator(4, total_budget=20, min_per_head=2)
    for u in range(4):
        alloc.observe(u, reuse=0.1 * u, drift=0.2 * u, reloads=float(u), attention_mass=0.5)
    budgets = alloc.reallocate()
    assert sum(budgets) == 20
    assert all(b >= 2 for b in budgets)


def test_adaptive_controller_runs():
    ctrl = AdaptiveReactiveController(2, total_budget=6)
    for t in range(16):
        ctrl.begin_step(t)
        ctrl.on_unit(0, {t % 5, (t + 1) % 5})
        ctrl.on_unit(1, {t % 3})
        ctrl.end_step()
    assert ctrl.ctrl.stats.steps == 16
    assert sum(ctrl.allocator.budgets) == 6


def test_profiler_percentiles():
    p = Profiler()
    for i in range(10):
        p.begin(f"r{i}")
        p.end(output_tokens=10, peak_gpu_bytes=100 * i)
    s = p.summary()
    assert s["n"] == 10
    assert s["tpot_p50_ms"] > 0
    assert s["peak_gpu_bytes"] == 900
