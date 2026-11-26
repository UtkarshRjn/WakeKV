"""Practical demand detectors for CPU-resident KV pages.

Quest-style upper bound on unscaled attention for a page:

    U_p(q) = sum_j max(q_j * kmin_{p,j}, q_j * kmax_{p,j})

Detectors rank CPU-resident pages and return a candidate set of size C.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorResult:
    name: str
    selected: set[int]
    scores: dict[int, float]
    latency_s: float
    summary_bytes: int


def quest_score(q: np.ndarray, kmin: np.ndarray, kmax: np.ndarray) -> float:
    """Upper bound on q·k for k in [kmin, kmax] componentwise."""
    return float(np.maximum(q * kmin, q * kmax).sum())


def centroid_score(q: np.ndarray, centroid: np.ndarray) -> float:
    return float(np.dot(q, centroid))


def select_top(scores: dict[int, float], budget: int) -> set[int]:
    if budget <= 0 or not scores:
        return set()
    ranked = sorted(scores, key=scores.get, reverse=True)
    return set(ranked[:budget])


def detect_random(cpu_pages: set[int], budget: int, rng: np.random.Generator) -> DetectorResult:
    t0 = time.perf_counter()
    pages = list(cpu_pages)
    if len(pages) <= budget:
        chosen = set(pages)
    else:
        chosen = set(rng.choice(pages, size=budget, replace=False).tolist())
    return DetectorResult("random", chosen, {p: 0.0 for p in chosen}, time.perf_counter() - t0, 0)


def detect_previous_demand(prev_wanted: set[int], cpu_pages: set[int], budget: int) -> DetectorResult:
    t0 = time.perf_counter()
    cand = prev_wanted & cpu_pages
    # fill remainder arbitrarily for a fair budget comparison
    rest = list(cpu_pages - cand)
    scores = {p: 1.0 for p in cand}
    for p in rest:
        scores[p] = 0.0
    selected = select_top(scores, budget) if scores else set()
    # prefer prev demanded
    ordered = list(cand) + rest
    selected = set(ordered[:budget])
    return DetectorResult(
        "previous_demand", selected, scores, time.perf_counter() - t0, 0
    )


def detect_previous_page_max(
    page_max_prev: np.ndarray, cpu_pages: set[int], budget: int
) -> DetectorResult:
    t0 = time.perf_counter()
    scores = {p: float(page_max_prev[p]) for p in cpu_pages if p < len(page_max_prev)}
    selected = select_top(scores, budget)
    return DetectorResult(
        "previous_page_max", selected, scores, time.perf_counter() - t0,
        int(page_max_prev.nbytes),
    )


def detect_centroid(
    q: np.ndarray, centroids: dict[int, np.ndarray], cpu_pages: set[int], budget: int
) -> DetectorResult:
    t0 = time.perf_counter()
    scores = {
        p: centroid_score(q, centroids[p]) for p in cpu_pages if p in centroids
    }
    selected = select_top(scores, budget)
    summary = sum(v.nbytes for v in centroids.values())
    return DetectorResult(
        "key_centroid", selected, scores, time.perf_counter() - t0, summary
    )


def detect_quest(
    q: np.ndarray,
    kmin: dict[int, np.ndarray],
    kmax: dict[int, np.ndarray],
    cpu_pages: set[int],
    budget: int,
) -> DetectorResult:
    t0 = time.perf_counter()
    scores = {
        p: quest_score(q, kmin[p], kmax[p])
        for p in cpu_pages
        if p in kmin and p in kmax
    }
    selected = select_top(scores, budget)
    summary = sum(v.nbytes for v in kmin.values()) + sum(v.nbytes for v in kmax.values())
    return DetectorResult("quest", selected, scores, time.perf_counter() - t0, summary)


def detect_oracle(wanted: set[int], cpu_pages: set[int], budget: int) -> DetectorResult:
    t0 = time.perf_counter()
    hit = wanted & cpu_pages
    rest = list(cpu_pages - hit)
    ordered = list(hit) + rest
    selected = set(ordered[:budget])
    scores = {p: 1.0 if p in hit else 0.0 for p in selected}
    return DetectorResult("oracle_top64", selected, scores, time.perf_counter() - t0, 0)


def score_selection(
    selected: set[int],
    wanted: set[int],
    attention_mass: dict[int, float] | None = None,
) -> dict:
    """Page recall / precision and optional attention-mass recall."""
    if not wanted:
        return {
            "page_recall": 1.0,
            "page_precision": 1.0 if not selected else 0.0,
            "attention_mass_recall": 1.0,
            "n_wanted": 0,
            "n_selected": len(selected),
            "n_hit": 0,
        }
    hit = selected & wanted
    recall = len(hit) / len(wanted)
    precision = len(hit) / len(selected) if selected else 0.0
    mass_rec = None
    if attention_mass is not None:
        tot = sum(attention_mass.get(p, 0.0) for p in wanted) or 1.0
        kept = sum(attention_mass.get(p, 0.0) for p in hit)
        mass_rec = kept / tot
    return {
        "page_recall": recall,
        "page_precision": precision,
        "attention_mass_recall": mass_rec,
        "n_wanted": len(wanted),
        "n_selected": len(selected),
        "n_hit": len(hit),
    }


def update_key_extrema(
    kmin: dict[int, np.ndarray],
    kmax: dict[int, np.ndarray],
    page: int,
    keys: np.ndarray,
) -> None:
    """``keys`` is [tokens_in_page, d]."""
    lo = keys.min(axis=0)
    hi = keys.max(axis=0)
    if page not in kmin:
        kmin[page] = lo.astype(np.float32, copy=True)
        kmax[page] = hi.astype(np.float32, copy=True)
    else:
        np.minimum(kmin[page], lo, out=kmin[page])
        np.maximum(kmax[page], hi, out=kmax[page])
