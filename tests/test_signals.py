import numpy as np
import pytest

from wakekv.signals import (
    causal_zscore,
    evaluate_fixed_threshold,
    evaluate_signal,
    page_kv_bytes,
    pooled_threshold,
    signal_drift,
    signal_entropy_trend,
    signal_needle_mass_delta,
    signal_online_rco,
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


def test_transfer_math():
    kb = page_kv_bytes(page_tokens=16, head_dim=128, dtype_bytes=2)
    assert kb == 2 * 16 * 128 * 2
    # 64 pages of one head over 21 GB/s at 30ms/step is well under one step
    assert transfer_steps_needed(64, kb) < 0.1
    # but 256 heads x 32 layers worth would not be
    assert transfer_steps_needed(64 * 256, kb) > transfer_steps_needed(64, kb)
