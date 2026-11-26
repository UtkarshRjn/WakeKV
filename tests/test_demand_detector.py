"""Tests for Quest-style and baseline demand detectors."""

from __future__ import annotations

import numpy as np

from wakekv.demand_detector import (
    detect_oracle,
    detect_previous_demand,
    detect_quest,
    detect_random,
    quest_score,
    score_selection,
    update_key_extrema,
)


def test_quest_score_upper_bounds_dot():
    q = np.array([1.0, -1.0])
    kmin = np.array([-2.0, -3.0])
    kmax = np.array([4.0, 5.0])
    # max of q*kmin and q*kmax per dim: max(1*-2, 1*4)=4; max(-1*-3, -1*5)=3
    assert quest_score(q, kmin, kmax) == 7.0


def test_detect_quest_ranks_true_page_high():
    d = 8
    q = np.ones(d)
    kmin, kmax = {}, {}
    # page 0 aligned with q; page 1 anti-aligned
    update_key_extrema(kmin, kmax, 0, np.ones((4, d)))
    update_key_extrema(kmin, kmax, 1, -np.ones((4, d)))
    det = detect_quest(q, kmin, kmax, {0, 1}, budget=1)
    assert det.selected == {0}


def test_score_selection_and_oracle():
    wanted = {1, 2, 3}
    selected = {2, 4}
    m = score_selection(selected, wanted)
    assert m["n_hit"] == 1
    assert abs(m["page_recall"] - 1 / 3) < 1e-9
    ora = detect_oracle(wanted, {1, 2, 3, 9}, budget=2)
    assert len(ora.selected) == 2
    assert ora.selected <= {1, 2, 3, 9}


def test_previous_demand_prefers_overlap():
    det = detect_previous_demand({5, 6}, {5, 6, 7, 8}, budget=2)
    assert 5 in det.selected and 6 in det.selected


def test_random_respects_budget():
    rng = np.random.default_rng(0)
    det = detect_random(set(range(20)), budget=3, rng=rng)
    assert len(det.selected) == 3
