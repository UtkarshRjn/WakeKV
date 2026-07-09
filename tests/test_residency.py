import numpy as np
import pytest

from wakekv.residency import (
    classify_unstable,
    simulate_frozen,
    simulate_full,
    simulate_reactive,
    wanted_stream_from_log,
)


def _stream(rows):
    """rows: list over steps of list over heads of iterable[page] -> Stream."""
    return [[frozenset(h) for h in step] for step in rows]


def test_full_never_misses():
    stream = _stream([[[0, 1]], [[2, 3]], [[0, 4]]])
    st = simulate_full(stream, 1)
    assert st.misses == 0
    assert st.total_accesses == 6
    # every distinct page stays resident: 0,1,2,3,4 -> peak 5
    assert st.peak_resident_pages == 5


def test_reactive_evicts_and_refetches():
    # budget 2; a head cycles through pages so old ones get demoted then wanted
    stream = _stream([[[0]], [[1]], [[2]], [[0]]])  # page 0 evicted by step 2, wanted at 3
    st = simulate_reactive(stream, 1, budget=2)
    # step 3 wants page 0, which was pushed to reservoir when 2 arrived -> 1 miss/fetch
    assert st.misses == 1
    assert st.fetches == 1
    assert st.stall_steps == 1
    assert st.peak_resident_pages == 2  # capped at budget


def test_reactive_no_miss_when_budget_covers_working_set():
    stream = _stream([[[0, 1]], [[0, 1]], [[0, 1]]])
    st = simulate_reactive(stream, 1, budget=4)
    assert st.misses == 0  # everything fits, nothing evicted


def test_reactive_memory_bounded_by_budget():
    rng = np.random.default_rng(0)
    stream = _stream([[[int(x) for x in rng.integers(0, 50, 3)]] for _ in range(40)])
    st = simulate_reactive(stream, 1, budget=8)
    assert st.peak_resident_pages <= 8


def test_classify_unstable_picks_the_churning_head():
    # head 0 stable (same pages), head 1 churns every step
    stream = _stream([
        [[0, 1], [10, 11]],
        [[0, 1], [20, 21]],
        [[0, 1], [30, 31]],
        [[0, 1], [40, 41]],
    ])
    unstable = classify_unstable(stream, 2, unstable_frac=0.5)
    assert unstable == {1}


def test_frozen_unstable_head_keeps_everything():
    # single churning head classified unstable -> never misses, memory grows
    stream = _stream([[[0]], [[1]], [[2]], [[3]]])
    st = simulate_frozen(stream, 1, budget=1, unstable_frac=1.0, refresh=2)
    assert st.misses == 0
    assert st.peak_resident_pages == 4


def test_frozen_stable_head_misses_on_shift():
    # one stable-classified head whose demand shifts beyond its fixed budget
    stream = _stream([[[0]], [[1]], [[2]], [[3]], [[4]], [[5]]])
    st = simulate_frozen(stream, 1, budget=1, unstable_frac=0.0, refresh=2)
    # fixed tiny budget + moving demand -> misses accrue
    assert st.misses > 0


def test_wanted_stream_from_log_pages():
    # [S=1, L=1, H=1, K=4] with page size 16
    topk = np.array([[[[0, 15, 16, 32]]]], dtype=np.int32)  # pages 0,0,1,2
    stream, n = wanted_stream_from_log(topk, page_size=16)
    assert n == 1
    assert stream[0][0] == frozenset({0, 1, 2})


def test_wanted_stream_drops_padding():
    topk = np.array([[[[5, -1, -1, -1]]]], dtype=np.int32)
    stream, _ = wanted_stream_from_log(topk, page_size=16)
    assert stream[0][0] == frozenset({0})  # only page 0 (token 5), pads dropped
