import numpy as np
import pytest

from wakekv.metrics import (
    activation_entropy,
    adjacent_jaccard_series,
    drift_events,
    jaccard,
    overlap_coefficient,
    rco,
    stability_score_vs_prefill,
    summarize_niah_run,
    temporal_stability,
)


def test_jaccard_basic():
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1}, set()) == 0.0


def test_overlap_coefficient():
    assert overlap_coefficient({1, 2, 3}, {2, 3}) == 1.0
    assert overlap_coefficient({1, 2}, {3, 4}) == 0.0
    assert overlap_coefficient(set(), {1}) == 0.0


def test_rco_bounds():
    # identical sets -> 1; disjoint sets -> clamped to 0
    assert rco({1, 2, 3, 4}, {1, 2, 3, 4}, k=4, n=100) == pytest.approx(1.0)
    assert rco({1, 2}, {3, 4}, k=2, n=100) == 0.0
    # half overlap, k=2, n=4: raw=0.5, chance=0.5 -> corrected 0
    assert rco({1, 2}, {2, 3}, k=2, n=4) == pytest.approx(0.0)
    # degenerate pool: n <= k means overlap is forced
    assert rco({1}, {1}, k=1, n=1) == 1.0


def test_adjacent_jaccard_series():
    sets = [{1, 2}, {2, 3}, {2, 3}]
    out = adjacent_jaccard_series(sets)
    assert out.tolist() == pytest.approx([1 / 3, 1.0])


def test_activation_entropy_uniform():
    counts = np.zeros(100)
    counts[:20] = 5  # uniform over 20 heads
    assert activation_entropy(counts) == pytest.approx(np.log(20))
    assert activation_entropy(np.zeros(10)) == 0.0


def test_temporal_stability_static_vs_churning():
    static = [{0, 1, 2, 3}] * 20
    pools = [100] * 20
    ts_static = temporal_stability(static, pools, k=4, window=8)
    assert ts_static.min() == pytest.approx(1.0)
    churn = [{i, i + 1, i + 2, i + 3} for i in range(0, 200, 10)]
    ts_churn = temporal_stability(churn, [1000] * len(churn), k=4, window=8)
    assert ts_churn.max() == 0.0


def test_stability_vs_prefill():
    prefill = {1, 2, 3, 4}
    steady = [prefill] * 9 + [{9, 10, 11, 12}]
    assert stability_score_vs_prefill(steady, prefill) == 1.0


def test_drift_events_fire_and_reset():
    prefill = {1, 2, 3, 4}
    # 8 steady steps, then a sustained shift -> exactly one event, then
    # the baseline resets and the new regime is steady again.
    sets = [prefill] * 8 + [{20, 21, 22, 23}] * 12
    ev = drift_events(sets, prefill, tau=0.5, window=8)
    assert len(ev) == 1
    assert 8 <= ev[0] < 20


def test_summarize_niah_run_shapes():
    rng = np.random.default_rng(0)
    cp = rng.random((50, 4, 8)) > 0.9
    rep = summarize_niah_run(cp)
    assert rep["total_heads"] == 32
    assert 0 <= rep["adjacent_jaccard_mean"] <= 1
    assert rep["unique_heads"] <= 32
    assert rep["entropy_baseline_ln20"] == pytest.approx(np.log(20))
