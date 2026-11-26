"""Score page-max prefetch against the pages touched by logged top-k.

The predictor at step t may use the page-max vector from step t-1.
It may not use the current top-k. A same-step score is reported only as
an upper bound: that one sees the attention row it is trying to predict.
"""

from __future__ import annotations

import numpy as np


def _wanted(idx_row: np.ndarray, page_size: int) -> set[int]:
    valid = idx_row[idx_row >= 0]
    if valid.size == 0:
        return set()
    return set((valid // page_size).tolist())


def _top_pages(page_row: np.ndarray, budget: int) -> set[int]:
    n = min(budget, page_row.shape[0])
    if n <= 0:
        return set()
    pick = np.argpartition(page_row, -n)[-n:]
    return set(int(p) for p in pick if page_row[p] > 0)


def prefetch_recall(
    page_max: np.ndarray,
    topk_idx: np.ndarray,
    page_size: int,
    budget: int,
) -> dict:
    """``page_max`` is [S, L, H, P], ``topk_idx`` is [S, L, H, K]."""
    steps, layers, heads, _ = page_max.shape
    causal_hit = causal_tot = 0
    same_hit = same_tot = 0
    persist_hit = persist_tot = 0
    prev_wanted: list[list[set[int]]] = [
        [set() for _ in range(heads)] for _ in range(layers)
    ]
    for t in range(steps):
        for layer in range(layers):
            for head in range(heads):
                wanted = _wanted(topk_idx[t, layer, head], page_size)
                if not wanted:
                    continue
                same = _top_pages(page_max[t, layer, head], budget)
                same_hit += len(wanted & same)
                same_tot += len(wanted)
                if t > 0:
                    causal = _top_pages(page_max[t - 1, layer, head], budget)
                    causal_hit += len(wanted & causal)
                    causal_tot += len(wanted)
                    persist_hit += len(wanted & prev_wanted[layer][head])
                    persist_tot += len(wanted)
                prev_wanted[layer][head] = wanted

    def rate(hit: int, tot: int) -> float | None:
        return (hit / tot) if tot else None

    return {
        "budget_pages": budget,
        "page_size": page_size,
        "causal_page_max_recall": rate(causal_hit, causal_tot),
        "previous_demand_recall": rate(persist_hit, persist_tot),
        "same_step_page_max_recall": rate(same_hit, same_tot),
        "causal_demands": causal_tot,
    }
