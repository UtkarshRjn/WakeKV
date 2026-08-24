import numpy as np
import pytest

from wakekv.residency import (
    classify_unstable,
    page_score_stream_from_log,
    simulate_evict,
    simulate_frozen,
    simulate_full,
    simulate_reactive,
    simulate_reasonalloc,
    simulate_rkv_uniform,
    simulate_snapkv,
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


def test_evict_never_recovers_a_destroyed_page():
    # budget 2: 0,1 fill it; wanting 2 evicts 0 (destroyed); every later
    # want of 0 misses again -- no single fetch fixes it, unlike reactive.
    stream = _stream([[[0]], [[1]], [[2]], [[0]], [[0]], [[0]]])
    st = simulate_evict(stream, 1, budget=2)
    assert st.misses == 3          # steps 3, 4, 5 all miss on page 0
    assert st.fetches == 0         # nothing is ever recovered, so nothing is "fetched"


def test_evict_misses_at_least_as_much_as_reactive_same_budget():
    # Identical stream and budget fed to both policies: reactive recovers
    # page 0 after one fetch and stops missing on it; evict never does.
    stream = _stream([[[0]], [[1]], [[2]], [[0]], [[0]], [[0]]])
    react = simulate_reactive(stream, 1, budget=2)
    evict = simulate_evict(stream, 1, budget=2)
    assert react.misses == 1       # one recoverable stall
    assert evict.misses == 3       # same page, same budget, never recovers
    assert evict.misses >= react.misses
    # Memory is matched: same LRU cap drives both, so no interpolation
    # is needed to compare them fairly (unlike reactive vs. frozen).
    assert evict.peak_resident_pages == react.peak_resident_pages


def test_evict_no_miss_when_budget_covers_working_set():
    stream = _stream([[[0, 1]], [[0, 1]], [[0, 1]]])
    st = simulate_evict(stream, 1, budget=4)
    assert st.misses == 0  # nothing ever exceeds budget, nothing destroyed


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


def _scored_stream(rows):
    """rows: list over steps of list over heads of dict[page, weight] -> ScoredStream."""
    return [[dict(h) for h in step] for step in rows]


# --------------------------------------------------------------- score log


def test_page_score_stream_sums_weights_within_a_page():
    # tokens 0,15 -> page 0; token 16 -> page 1; token 32 -> page 2
    topk_idx = np.array([[[[0, 15, 16, 32]]]], dtype=np.int32)
    topk_val = np.array([[[[0.4, 0.3, 0.2, 0.1]]]], dtype=np.float16)
    stream, n = page_score_stream_from_log(topk_idx, topk_val, page_size=16)
    assert n == 1
    scores = stream[0][0]
    assert scores[0] == pytest.approx(0.7, abs=1e-3)  # 0.4 + 0.3
    assert scores[1] == pytest.approx(0.2, abs=1e-3)
    assert scores[2] == pytest.approx(0.1, abs=1e-3)


def test_page_score_stream_drops_padding():
    topk_idx = np.array([[[[5, -1, -1, -1]]]], dtype=np.int32)
    topk_val = np.array([[[[0.9, 0.0, 0.0, 0.0]]]], dtype=np.float16)
    stream, _ = page_score_stream_from_log(topk_idx, topk_val, page_size=16)
    assert stream[0][0] == {0: pytest.approx(0.9, abs=1e-3)}


# ------------------------------------------------------------------ SnapKV


def test_snapkv_selects_by_attention_score_not_presence():
    # budget 2: pages 2 (score 9) and 0 (score 5) win; 1 and 3 lose despite
    # being wanted just as often -- selection is score-ranked, not a count.
    prefill_scores = [{0: 5.0, 1: 3.0, 2: 9.0, 3: 1.0}]
    stream = _stream([[[2]], [[1]], [[0]]])
    st = simulate_snapkv(stream, 1, budget=2, prefill_scores=prefill_scores)
    assert st.misses == 1        # only step 1's want of excluded page 1
    assert st.total_accesses == 3


def test_snapkv_excluded_page_never_recovers():
    # unlike reactive/frozen-refresh, SnapKV never reranks after prefill:
    # a page excluded once misses on every subsequent want, forever.
    prefill_scores = [{0: 5.0, 1: 3.0, 2: 9.0, 3: 1.0}]  # budget 2 excludes 1, 3
    stream = _stream([[[1]], [[1]], [[1]]])
    st = simulate_snapkv(stream, 1, budget=2, prefill_scores=prefill_scores)
    assert st.misses == 3
    assert st.fetches == 0  # nothing is ever recovered


def test_snapkv_always_keeps_generated_tokens():
    # prompt pages {0, 1}; budget 1 keeps only page 1 (higher score), so
    # page 0 (prompt) always misses -- but page 5 (id >= boundary, i.e. a
    # post-prompt/generated page) is always retained once first wanted.
    prefill_scores = [{0: 1.0, 1: 2.0}]  # boundary = max(keys) + 1 = 2
    stream = _stream([[[1]], [[0]], [[5]], [[5]]])
    st = simulate_snapkv(stream, 1, budget=1, prefill_scores=prefill_scores)
    assert st.misses == 1          # only step 1's want of excluded prompt page 0
    assert st.total_accesses == 4  # step2's and step3's wants of page 5 both hit


# --------------------------------------------------------------- R-KV uniform


def test_rkv_uniform_evicts_by_importance_not_recency():
    # page 1 is touched once, early, but with high weight; page 0 is
    # touched later (more recent) but with tiny weight. A recency-based
    # rule (LRU, like simulate_evict) would evict page 1 (older). R-KV's
    # importance-ranked rule instead evicts page 0 (lower accumulated
    # score) despite it being the most recently wanted page.
    scored_stream = _scored_stream([
        [{1: 5.0}],
        [{}],
        [{}],
        [{0: 0.01}],          # buffer=4 triggers recompression after this step
        [{0: 1.0, 1: 1.0}],   # re-want both: page 0 should miss, page 1 should hit
    ])
    st = simulate_rkv_uniform(scored_stream, 1, budget=1, buffer=4)
    assert st.misses == 1
    assert st.total_accesses == 4
    assert st.peak_resident_pages <= 1


def test_rkv_uniform_no_pruning_under_budget():
    scored_stream = _scored_stream([[{0: 1.0}], [{0: 1.0}], [{0: 1.0}], [{0: 1.0}]])
    st = simulate_rkv_uniform(scored_stream, 1, budget=4, buffer=4)
    assert st.misses == 0


# ------------------------------------------------------------- ReasonAlloc


def test_reasonalloc_head_budgets_floor_prevents_starvation():
    from wakekv.residency import _reasonalloc_head_budgets

    # two heads with zero qualifying demand, one with plenty; the floor
    # (mu * b_layer / H) still guarantees each starved head >= 1 page.
    budgets = _reasonalloc_head_budgets([0.0, 0.0, 100.0], b_layer=10, mu=0.25)
    assert sum(budgets) == 10
    assert budgets[0] >= 1
    assert budgets[1] >= 1


def test_reasonalloc_rejects_n_units_not_multiple_of_n_heads():
    scored_stream = _scored_stream([[{0: 1.0}]])
    with pytest.raises(ValueError):
        simulate_reasonalloc(scored_stream, n_units=3, n_heads=2, budget=1)


def test_reasonalloc_reallocates_budget_from_low_to_high_importance_head():
    # one layer, two heads. head 0 gets six low, distinct-scored pages;
    # head 1 gets three much higher-scored pages. Pooling + thresholding
    # at b_layer=4 (budget=2 * n_heads=2) gives head 1 a bigger share of
    # the shared layer budget than head 0 -- something a FIXED per-unit
    # budget (simulate_evict) can never do, since it never reallocates
    # across heads.
    scored_stream = _scored_stream([
        [
            {10: 0.06, 11: 0.05, 12: 0.04, 13: 0.03, 14: 0.02, 15: 0.01},
            {20: 9.0, 21: 8.0, 22: 7.0},
        ],
        [{}, {}],  # step 1 -> (t+1)=2 triggers reallocation (delta=2)
        [{11: 0.5}, {22: 0.1}],  # re-want a head-0 page (evicted) and a
                                  # head-1 page (kept): probe the outcome
    ])
    st = simulate_reasonalloc(
        scored_stream, n_units=2, n_heads=2, budget=2, delta=2, mu=0.25
    )
    # head 0 shrank to budget 1 (only its top-scored page 10 survives) so
    # its lower-scored page 11 was evicted and misses on re-want; head 1
    # grew to budget 3 (fits all three of its pages) so page 22 still hits.
    assert st.misses == 1
    assert st.total_accesses == 6 + 3 + 1 + 1
