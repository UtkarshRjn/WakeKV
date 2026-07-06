import numpy as np
import pytest

from wakekv.signals import (
    evaluate_signal,
    page_kv_bytes,
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


def test_transfer_math():
    kb = page_kv_bytes(page_tokens=16, head_dim=128, dtype_bytes=2)
    assert kb == 2 * 16 * 128 * 2
    # 64 pages of one head over 21 GB/s at 30ms/step is well under one step
    assert transfer_steps_needed(64, kb) < 0.1
    # but 256 heads x 32 layers worth would not be
    assert transfer_steps_needed(64 * 256, kb) > transfer_steps_needed(64, kb)
