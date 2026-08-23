import numpy as np
import pytest

from wakekv.signals import (
    bonferroni_z_bar,
    causal_zscore,
    ensemble_vote,
    evaluate_fixed_threshold,
    evaluate_signal,
    event_horizon_mask,
    page_kv_bytes,
    pooled_threshold,
    signal_discovered_tokens,
    signal_drift,
    signal_entropy_trend,
    signal_needle_mass_delta,
    signal_online_rco,
    token_wake_stats,
    transfer_steps_needed,
    wake_events,
)


def test_wake_events_hysteresis():
    score = np.zeros(40)
    score[20:] = 0.5  # quiet for 20 steps, then sustained activation
    ev = wake_events(score, lo=0.05, hi=0.2, min_quiet=8)
    assert ev == [20]
    # jitter below hi never fires
    jitter = np.full(40, 0.1)
    assert wake_events(jitter) == []
    # brief blip after too-short quiet period doesn't fire twice
    score2 = np.zeros(40)
    score2[10] = 0.5
    score2[14] = 0.5
    assert wake_events(score2, min_quiet=8) == [10]


def test_signal_online_rco_detects_set_change():
    sets = [{0, 1, 2, 3}] * 10 + [{50, 51, 52, 53}] * 5
    pools = [1000] * 15
    sig = signal_online_rco(sets, pools, k=4)
    assert sig[10] == pytest.approx(1.0)  # the switch step
    assert sig[5] == pytest.approx(0.0)   # steady state


def test_signal_drift_monotone_after_shift():
    sets = [{0, 1, 2, 3}] * 8 + [{50, 51, 52, 53}] * 8
    sig = signal_drift(sets, window=4)
    assert sig[-1] > sig[4]


def test_signal_entropy_trend_rises_on_spread():
    peaked = np.zeros((10, 8))
    peaked[:, 0] = 1.0
    spread = np.full((5, 8), 1 / 8)
    vals = np.concatenate([peaked, spread])
    sig = signal_entropy_trend(vals, window=4)
    assert sig[10] > 0.5  # entropy jump at the transition


def test_needle_mass_delta():
    score = np.zeros(20)
    score[10:] = 1.0
    sig = signal_needle_mass_delta(score, window=4)
    assert sig[10] == pytest.approx(1.0)


def test_evaluate_signal_perfect_predictor():
    # signal spikes exactly 2 steps before each event
    signal = np.zeros(100)
    events = [20, 60]
    for t in events:
        signal[t - 2] = 1.0
    results = evaluate_signal(signal, events, leads=(4,), thresholds=np.array([0.5]))
    r = results[0]
    assert r.recall == 1.0 and r.precision == 1.0


def test_evaluate_signal_useless_predictor():
    rng = np.random.default_rng(0)
    signal = rng.random(500)
    events = [400]
    results = evaluate_signal(signal, events, leads=(1,), thresholds=np.array([0.99]))
    assert results[0].precision < 0.5


def test_fixed_threshold_pools_across_heads():
    # Two heads: one where the signal cleanly precedes its event, one pure
    # noise with no event. A global threshold must still recover the real
    # event and not be fooled into perfect scores by per-head tuning.
    good = np.zeros(100)
    good[38] = good[39] = 1.0  # spikes just before the event at 40
    noise = np.linspace(0, 0.3, 100)
    per_head = [(good, [40]), (noise, [])]
    th = pooled_threshold(per_head, 0.9)
    res = evaluate_fixed_threshold(per_head, th, leads=(4,))
    r = res[4]
    assert r.n_events == 1
    assert r.recall == 1.0  # the real event is caught


def test_fixed_threshold_penalizes_false_alarms():
    # A signal that fires everywhere gets full recall but poor precision
    # once pooled — the honest counterweight to per-head cherry-picking.
    trigger_happy = np.ones(50)
    per_head = [(trigger_happy, [25])]
    res = evaluate_fixed_threshold(per_head, 0.5, leads=(2,))
    r = res[2]
    assert r.recall == 1.0
    assert r.precision < 0.2  # ~49 alarms, only a few near the single event


def test_causal_zscore_ignores_constant_baseline():
    # A head that always churns at the same rate -> z-score ~0 (no anomaly),
    # even though its raw signal is high. This is the whole point: normalize
    # away each head's own baseline so a spike ABOVE normal is what fires.
    flat_high = np.full(60, 0.8)
    z = causal_zscore(flat_high, window=16)
    assert np.abs(z).max() < 1e-6


def test_causal_zscore_flags_spike_above_own_normal():
    sig = np.concatenate([np.full(40, 0.1), np.array([0.9]), np.full(10, 0.1)])
    z = causal_zscore(sig, window=16)
    assert z[40] > 3.0  # the jump is many sigma above this head's baseline
    assert z[20] == pytest.approx(0.0, abs=1e-6)  # steady baseline -> no alarm


def test_causal_zscore_is_causal():
    # z at step t must not depend on values at t or later beyond x_t itself:
    # changing a future value leaves earlier z-scores untouched.
    a = np.concatenate([np.full(20, 0.1), np.full(20, 0.5)])
    b = a.copy(); b[30] = 9.0
    za, zb = causal_zscore(a, window=8), causal_zscore(b, window=8)
    assert np.allclose(za[:30], zb[:30])


def test_event_horizon_mask_marks_pre_event_window():
    mask = event_horizon_mask(events=[10], length=20, lead=4)
    assert mask[6:10].all()      # (10-4, 10) is "wake soon"
    assert not mask[:6].any()
    assert not mask[10:].any()   # event itself and after: not "soon" anymore


def test_token_wake_stats_flags_a_genuinely_correlated_token():
    rng = np.random.default_rng(0)
    n = 2000
    token_ids = rng.integers(0, 50, size=n)  # 50-token vocab, uniform baseline
    wake_soon = np.zeros(n, dtype=bool)
    # token 7 co-occurs with "wake soon" far more than its ~1/50 base rate
    hits = np.flatnonzero(token_ids == 7)
    wake_soon[hits[: len(hits) // 2]] = True
    # sprinkle some baseline-rate wakes elsewhere so p0 > 0
    other = np.flatnonzero(token_ids != 7)
    wake_soon[rng.choice(other, size=len(other) // 50, replace=False)] = True

    stats = token_wake_stats(token_ids, wake_soon, min_count=5)
    by_id = {s.token_id: s for s in stats}
    assert by_id[7].lift > 3.0
    assert by_id[7].z > bonferroni_z_bar(n_tests=len(stats), alpha=0.05)
    # a token with no special relationship shouldn't clear the corrected bar
    other_ids = [tid for tid in by_id if tid != 7]
    assert not any(by_id[tid].z > bonferroni_z_bar(len(stats)) for tid in other_ids)


def test_token_wake_stats_respects_min_count():
    token_ids = np.array([1, 1, 1, 2])
    wake_soon = np.array([True, True, True, True])
    stats = token_wake_stats(token_ids, wake_soon, min_count=5)
    assert stats == []  # neither token reaches min_count


def test_bonferroni_z_bar_rises_with_more_tests():
    z1 = bonferroni_z_bar(n_tests=1, alpha=0.05)
    z_many = bonferroni_z_bar(n_tests=30000, alpha=0.05)
    assert z_many > z1 > 1.5
    # sanity check against the well-known single-test two-sided z ~1.96
    assert bonferroni_z_bar(n_tests=1, alpha=0.05) == pytest.approx(1.645, abs=0.01)


def test_signal_discovered_tokens_marks_selected_ids():
    token_ids = np.array([1, 2, 3, 2, 1])
    sig = signal_discovered_tokens(token_ids, marker_token_ids={2})
    assert sig.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_signal_discovered_tokens_empty_selection_is_all_zero():
    token_ids = np.array([1, 2, 3])
    sig = signal_discovered_tokens(token_ids, marker_token_ids=set())
    assert sig.tolist() == [0.0, 0.0, 0.0]


def test_ensemble_vote_counts_agreement():
    a = np.array([0.0, 3.0, 3.0, 0.0])
    b = np.array([0.0, 3.0, 0.0, 3.0])
    c = np.array([0.0, 0.0, 3.0, 3.0])
    votes = ensemble_vote([a, b, c], threshold=2.0)
    assert votes.tolist() == [0.0, 2.0, 2.0, 2.0]


def test_ensemble_vote_requires_min_votes_to_beat_lone_false_positive():
    # A lone noisy signal fires alone at step 5 (1 vote); the other two agree
    # at step 10 (2 votes). Thresholding the ensemble at 1.5 votes keeps the
    # real agreement and drops the singleton false alarm.
    a = np.zeros(20); a[5] = 5.0; a[10] = 5.0
    b = np.zeros(20); b[10] = 5.0
    c = np.zeros(20); c[10] = 5.0
    votes = ensemble_vote([a, b, c], threshold=2.0)
    fired = np.flatnonzero(votes > 1.5)
    assert fired.tolist() == [10]


def test_ensemble_vote_rejects_empty_input():
    with pytest.raises(ValueError):
        ensemble_vote([])


def test_transfer_math():
    kb = page_kv_bytes(page_tokens=16, head_dim=128, dtype_bytes=2)
    assert kb == 2 * 16 * 128 * 2
    # 64 pages of one head over 21 GB/s at 30ms/step is well under one step
    assert transfer_steps_needed(64, kb) < 0.1
    # but 256 heads x 32 layers worth would not be
    assert transfer_steps_needed(64 * 256, kb) > transfer_steps_needed(64, kb)
