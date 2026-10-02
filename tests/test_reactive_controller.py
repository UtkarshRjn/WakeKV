"""Unit tests for the standalone ReactiveController."""

from __future__ import annotations

import numpy as np
import pytest

from wakekv.reactive_controller import ReactiveController, StepEvent


# ------------------------------------------------------------- construction
def test_bad_args():
    with pytest.raises(ValueError):
        ReactiveController(n_units=1, budget=-1)
    with pytest.raises(ValueError):
        ReactiveController(n_units=1, budget=4, rerank_interval=0)
    with pytest.raises(ValueError):
        ReactiveController(n_units=-1, budget=4)
    with pytest.raises(ValueError):
        ReactiveController(n_units=1, budget=4, demotion="destroy")  # typo


def test_step_boundary_discipline():
    ctrl = ReactiveController(n_units=1, budget=4)
    with pytest.raises(RuntimeError):
        ctrl.on_unit_lru(0, {1, 2})  # must call begin_step first
    ctrl.begin_step(0)
    with pytest.raises(RuntimeError):
        ctrl.begin_step(1)  # already in a step
    ctrl.end_step()
    with pytest.raises(RuntimeError):
        ctrl.end_step()  # end without begin


# ------------------------------------------------------------------ LRU mode
def test_lru_empty_state():
    ctrl = ReactiveController(n_units=1, budget=4)
    ctrl.begin_step(0)
    ev = ctrl.on_unit_lru(0, set())
    ctrl.end_step()
    assert ev == StepEvent()  # no fetches, no evictions, no misses
    d = ctrl.stats.as_dict()
    assert d["steps"] == 1 and d["total_wanted"] == 0
    assert d["peak_resident_pages"] == 0


def test_lru_first_touch_within_budget():
    ctrl = ReactiveController(n_units=1, budget=4)
    ctrl.begin_step(0)
    ev = ctrl.on_unit_lru(0, {1, 2, 3})
    ctrl.end_step()
    assert ev.fetches == set() and ev.evictions == set() and ev.n_misses == 0
    assert ctrl.resident_pages(0) == {1, 2, 3}
    assert ctrl.reservoir_pages(0) == set()


def test_lru_evicts_least_recently_wanted_at_capacity():
    """Fill to budget over several steps, then push in more — oldest goes."""
    ctrl = ReactiveController(n_units=1, budget=3)
    for step, want in enumerate([{1}, {2}, {3}, {4}]):
        ctrl.begin_step(step)
        ctrl.on_unit_lru(0, want)
        ctrl.end_step()
    # Wanted in order 1, 2, 3, 4 — with budget 3, page 1 (oldest) is evicted.
    assert ctrl.resident_pages(0) == {2, 3, 4}
    assert ctrl.reservoir_pages(0) == {1}


def test_lru_touching_a_page_refreshes_it():
    """Access an old page before the sweep — it should NOT be evicted."""
    ctrl = ReactiveController(n_units=1, budget=3)
    for step, want in enumerate([{1}, {2}, {3}]):
        ctrl.begin_step(step)
        ctrl.on_unit_lru(0, want)
        ctrl.end_step()
    # Now touch page 1 (oldest) and add page 4. Page 2 should get evicted, not 1.
    ctrl.begin_step(3)
    ctrl.on_unit_lru(0, {1, 4})
    ctrl.end_step()
    assert ctrl.resident_pages(0) == {1, 3, 4}
    assert ctrl.reservoir_pages(0) == {2}


def test_lru_fetch_on_demand_is_a_miss():
    """Page evicted, then wanted again → miss + fetch."""
    ctrl = ReactiveController(n_units=1, budget=2)
    for step, want in enumerate([{1}, {2}, {3}]):
        ctrl.begin_step(step)
        ctrl.on_unit_lru(0, want)
        ctrl.end_step()
    # Page 1 is now in the reservoir. Wanting it again → miss.
    ctrl.begin_step(3)
    ev = ctrl.on_unit_lru(0, {1})
    ctrl.end_step()
    assert ev.fetches == {1}
    assert ev.n_misses == 1
    assert 1 in ctrl.resident_pages(0)
    d = ctrl.stats.as_dict()
    assert d["total_fetches"] == 1
    assert d["total_misses"] == 1
    assert d["stall_steps"] == 1


def test_lru_rerank_interval_defers_eviction():
    """With rerank_interval=4, resident set may exceed budget between reranks."""
    ctrl = ReactiveController(n_units=1, budget=2, rerank_interval=4)
    for step in range(3):  # steps 0, 1, 2 — no rerank yet (step % 4 != 0 for 1, 2)
        ctrl.begin_step(step)
        ctrl.on_unit_lru(0, {step + 1})
        ctrl.end_step()
    # Step 0 does rerank (0 % 4 == 0) but resident had 1 page ≤ budget, so no evict.
    # Steps 1, 2 grow resident to 3 pages > budget, but no eviction.
    assert ctrl.resident_pages(0) == {1, 2, 3}

    # Step 4 (4 % 4 == 0) triggers eviction; want a fresh page to keep the LRU
    # order deterministic (touching an existing page would refresh it).
    ctrl.begin_step(4)
    ctrl.on_unit_lru(0, {99})
    ctrl.end_step()
    # Resident had {1,2,3,99}; budget 2 → evict oldest {1, 2}, keep {3, 99}.
    assert ctrl.resident_pages(0) == {3, 99}


def test_lru_matches_original_simulator_on_synthetic_shift():
    """Regression check: on the same synthetic shifting-role stream used to
    validate the Phase 2a simulator, the controller must produce the SAME
    miss rate and mean-resident-pages as ``simulate_reactive``."""
    from wakekv.residency import simulate_reactive

    # Build the exact synthetic stream from PR #3's sanity check.
    S, H, B = 300, 4, 12
    stream = []
    for t in range(S):
        row = []
        for h in range(H):
            center = (t * 2 + h * 7) % 120
            row.append(frozenset({(center + d) for d in range(6)}))
        stream.append(row)

    # Drive the controller.
    ctrl = ReactiveController(n_units=H, budget=B, rerank_interval=1)
    for t, row in enumerate(stream):
        ctrl.begin_step(t)
        for u, wanted in enumerate(row):
            ctrl.on_unit_lru(u, set(wanted))
        ctrl.end_step()
    got = ctrl.stats.as_dict()

    # Reference: the pre-refactor simulator.
    ref = simulate_reactive(stream, H, B)

    assert got["miss_rate"] == pytest.approx(ref.miss_rate)
    assert got["mean_resident_pages"] == pytest.approx(ref.mean_resident_pages)
    assert got["peak_resident_pages"] == ref.peak_resident_pages
    assert got["total_fetches"] == ref.fetches
    assert got["stall_steps"] == ref.stall_steps


# --------------------------------------------------------------- evict mode
def test_evict_never_recovers_a_destroyed_page():
    """Phase 3 / E3c (ReasonAlloc-style): unlike offload, a demoted page
    never comes back, no matter how many times it's re-wanted."""
    ctrl = ReactiveController(n_units=1, budget=2, demotion="evict")
    for step, want in enumerate([{1}, {2}, {3}]):
        ctrl.begin_step(step)
        ctrl.on_unit_lru(0, want)
        ctrl.end_step()
    # Page 1 is now destroyed (evicted at budget 2, page 3 pushed it out).
    assert ctrl.destroyed_pages(0) == {1}
    assert ctrl.reservoir_pages(0) == set()  # nothing ever goes to reservoir

    # Want it back, twice in a row -- both are permanent misses, no fetch.
    for step in (3, 4):
        ctrl.begin_step(step)
        ev = ctrl.on_unit_lru(0, {1})
        ctrl.end_step()
        assert ev.fetches == set()
        assert ev.n_misses == 1
        assert 1 not in ctrl.resident_pages(0)
    d = ctrl.stats.as_dict()
    assert d["total_misses"] == 2
    assert d["total_fetches"] == 0  # never actually fetched, just missed


def test_evict_first_touch_is_not_a_miss():
    """A page that's never been resident before is ordinary cache growth,
    not a miss -- evict mode only penalizes RE-wanting a destroyed page."""
    ctrl = ReactiveController(n_units=1, budget=4, demotion="evict")
    ctrl.begin_step(0)
    ev = ctrl.on_unit_lru(0, {1, 2, 3})
    ctrl.end_step()
    assert ev.n_misses == 0
    assert ctrl.resident_pages(0) == {1, 2, 3}


def test_evict_matches_original_simulator_on_synthetic_shift():
    """Regression check: on the same synthetic shifting-role stream used
    for the offload regression test, evict mode must produce the SAME
    numbers as wakekv.residency.simulate_evict (already validated on real
    Phase 0/1 logs in PR #19)."""
    from wakekv.residency import simulate_evict

    S, H, B = 300, 4, 12
    stream = []
    for t in range(S):
        row = []
        for h in range(H):
            center = (t * 2 + h * 7) % 120
            row.append(frozenset({(center + d) for d in range(6)}))
        stream.append(row)

    ctrl = ReactiveController(n_units=H, budget=B, rerank_interval=1, demotion="evict")
    for t, row in enumerate(stream):
        ctrl.begin_step(t)
        for u, wanted in enumerate(row):
            ctrl.on_unit_lru(u, set(wanted))
        ctrl.end_step()
    got = ctrl.stats.as_dict()

    ref = simulate_evict(stream, H, B)

    assert got["miss_rate"] == pytest.approx(ref.miss_rate)
    assert got["mean_resident_pages"] == pytest.approx(ref.mean_resident_pages)
    assert got["peak_resident_pages"] == ref.peak_resident_pages
    assert got["total_misses"] == ref.misses


def test_evict_mode_does_not_affect_explicit_entry_point():
    """on_unit_explicit doesn't implement evict semantics (see class
    docstring) -- it should behave exactly as in offload mode regardless
    of the demotion setting, not silently do something undefined."""
    ctrl_evict = ReactiveController(n_units=1, budget=2, demotion="evict")
    ctrl_offload = ReactiveController(n_units=1, budget=2, demotion="offload")
    for ctrl in (ctrl_evict, ctrl_offload):
        ctrl.begin_step(0)
        ctrl.on_unit_explicit(0, {1, 2})
        ctrl.end_step()
        ctrl.begin_step(1)
        ctrl.on_unit_explicit(0, {3, 4})
        ctrl.end_step()
    assert ctrl_evict.reservoir_pages(0) == ctrl_offload.reservoir_pages(0) == {1, 2}
    assert ctrl_evict.destroyed_pages(0) == set()


# ------------------------------------------------------------- explicit mode
def test_explicit_diff_computes_fetches_and_evictions():
    ctrl = ReactiveController(n_units=1, budget=4)
    ctrl.begin_step(0)
    ev = ctrl.on_unit_explicit(0, {1, 2, 3})
    ctrl.end_step()
    assert ev.fetches == {1, 2, 3} and ev.evictions == set()
    assert ctrl.resident_pages(0) == {1, 2, 3}

    ctrl.begin_step(1)
    ev = ctrl.on_unit_explicit(0, {2, 3, 4})  # drop 1, add 4
    ctrl.end_step()
    assert ev.fetches == {4}
    assert ev.evictions == {1}
    assert ctrl.resident_pages(0) == {2, 3, 4}
    assert ctrl.reservoir_pages(0) == {1}


def test_explicit_rejects_oversized_desired():
    ctrl = ReactiveController(n_units=1, budget=2)
    ctrl.begin_step(0)
    with pytest.raises(ValueError, match="budget"):
        ctrl.on_unit_explicit(0, {1, 2, 3})
    ctrl.end_step()


def test_explicit_off_cadence_is_noop():
    """rerank_interval=16: only steps 0, 16, 32... apply changes."""
    ctrl = ReactiveController(n_units=1, budget=4, rerank_interval=16)

    ctrl.begin_step(0)
    ctrl.on_unit_explicit(0, {1, 2, 3, 4})
    ctrl.end_step()
    assert ctrl.resident_pages(0) == {1, 2, 3, 4}

    # Step 5 — off cadence, request different set. Should be ignored.
    ctrl.begin_step(5)
    ev = ctrl.on_unit_explicit(0, {5, 6, 7, 8})
    ctrl.end_step()
    assert ev == StepEvent()
    assert ctrl.resident_pages(0) == {1, 2, 3, 4}

    # Step 16 — on cadence again.
    ctrl.begin_step(16)
    ev = ctrl.on_unit_explicit(0, {5, 6, 7, 8})
    ctrl.end_step()
    assert ev.fetches == {5, 6, 7, 8} and ev.evictions == {1, 2, 3, 4}
    assert ctrl.resident_pages(0) == {5, 6, 7, 8}


def test_explicit_pulls_from_reservoir_not_double_fetch():
    """A page previously evicted should be pulled from reservoir on re-desire,
    but still counts as a fetch (KV bytes must move CPU → GPU)."""
    ctrl = ReactiveController(n_units=1, budget=2)
    ctrl.begin_step(0)
    ctrl.on_unit_explicit(0, {1, 2})
    ctrl.end_step()
    ctrl.begin_step(1)
    ctrl.on_unit_explicit(0, {3, 4})  # 1, 2 evicted to reservoir
    ctrl.end_step()
    assert ctrl.reservoir_pages(0) == {1, 2}

    ctrl.begin_step(2)
    ev = ctrl.on_unit_explicit(0, {1, 2})  # want them back
    ctrl.end_step()
    assert ev.fetches == {1, 2}
    assert ctrl.reservoir_pages(0) == {3, 4}


# ---------------------------------------------------------- multi-unit stats
def test_stats_aggregate_across_units():
    ctrl = ReactiveController(n_units=3, budget=2)
    ctrl.begin_step(0)
    ctrl.on_unit_lru(0, {1, 2, 3})  # 1 evict
    ctrl.on_unit_lru(1, {10, 20})   # 0 evicts
    ctrl.on_unit_lru(2, {100})      # 0 evicts
    ctrl.end_step()
    d = ctrl.stats.as_dict()
    assert d["total_wanted"] == 6
    assert d["peak_resident_pages"] == 5  # 2 + 2 + 1
    assert d["total_evictions"] == 1
