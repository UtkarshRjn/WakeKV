"""Pure-numpy metrics for head-churn analysis (Phase 0).

Definitions follow the source papers exactly:
- adjacent-step Jaccard, activation entropy: "Retrieval Heads are Dynamic"
  (arXiv:2602.11162, Table 1)
- Random-Corrected Overlap (RCO), temporal stability: FlexiCache
  (arXiv:2511.00868, Eqs. for RCO/TS)
- overlap coefficient, drift trigger: HeteroCache (arXiv:2601.13684,
  Eqs. 1, 2, 9, 10)
"""

from __future__ import annotations

import numpy as np


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def overlap_coefficient(a: set, b: set) -> float:
    """|A ∩ B| / min(|A|, |B|) (Szymkiewicz; HeteroCache Eq. 1)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def adjacent_jaccard_series(active_sets: list[set]) -> np.ndarray:
    """Jaccard between consecutive steps' active-head sets.

    Steps where both sets are empty contribute 1.0 by convention; pairs
    with exactly one empty set contribute 0.0. Report alongside the
    fraction of empty steps so this convention can't hide degenerate runs.
    """
    return np.array(
        [jaccard(active_sets[t], active_sets[t + 1]) for t in range(len(active_sets) - 1)]
    )


def activation_entropy(activation_counts: np.ndarray) -> float:
    """Natural-log entropy of the per-head activation distribution.

    Baseline for comparison: uniform over top-20 static heads = ln(20) ≈ 3.0.
    """
    total = activation_counts.sum()
    if total == 0:
        return 0.0
    p = activation_counts[activation_counts > 0] / total
    return float(-(p * np.log(p)).sum())


def rco(a: set, b: set, k: int, n: int) -> float:
    """Random-Corrected Overlap between two top-k index sets (FlexiCache).

    RCO = max(0, (|A∩B|/K − K/N) / (1 − K/N)); 0 = chance-level, 1 = identical.
    ``n`` is the candidate-pool size (e.g. number of pages at that step).
    """
    if n <= k or k == 0:
        return 1.0
    raw = len(a & b) / k
    chance = k / n
    return max(0.0, (raw - chance) / (1.0 - chance))


def temporal_stability(
    topk_sets: list[set], pool_sizes: list[int], k: int, window: int = 16
) -> np.ndarray:
    """Per-step temporal stability TS(s) = mean_{Δ=1..W-1} RCO(s, s+Δ).

    Returns an array of length len(topk_sets) - window + 1.
    """
    out = []
    for s in range(len(topk_sets) - window + 1):
        vals = [
            rco(topk_sets[s], topk_sets[s + d], k, pool_sizes[s + d])
            for d in range(1, window)
        ]
        out.append(float(np.mean(vals)))
    return np.array(out)


def stability_score_vs_prefill(topk_sets: list[set], prefill_set: set) -> float:
    """HeteroCache Eq. 2: median over decode steps of overlap vs prefill top-k."""
    if not topk_sets:
        return 0.0
    return float(np.median([overlap_coefficient(s, prefill_set) for s in topk_sets]))


def drift_events(
    topk_sets: list[set],
    prefill_set: set,
    tau: float = 0.5,
    window: int = 8,
) -> list[int]:
    """HeteroCache Eqs. 9-10: windowed-median overlap vs a resettable baseline.

    Returns the decode-step indices where a drift event fires. After each
    event the baseline resets to that step's top-k set.
    """
    base = set(prefill_set)
    events: list[int] = []
    history: list[float] = []
    for t, s in enumerate(topk_sets):
        history.append(overlap_coefficient(s, base))
        if len(history) >= window and float(np.median(history[-window:])) < tau:
            events.append(t)
            base = set(s)
            history = []
    return events


def summarize_niah_run(copy_paste: np.ndarray) -> dict:
    """Table-1-style statistics from a binary activation tensor.

    ``copy_paste``: bool array [steps, layers, heads].
    """
    steps, layers, heads = copy_paste.shape
    flat = copy_paste.reshape(steps, layers * heads)
    active_sets = [set(np.flatnonzero(flat[t])) for t in range(steps)]
    per_step_counts = flat.sum(axis=1)
    counts_per_head = flat.sum(axis=0)
    adj = adjacent_jaccard_series(active_sets)
    static_top20 = set(np.argsort(counts_per_head)[::-1][:20])
    jac_static = [jaccard(a, static_top20) for a in active_sets if a]
    return {
        "steps": int(steps),
        "total_heads": int(layers * heads),
        "active_per_step_mean": float(per_step_counts.mean()),
        "active_per_step_std": float(per_step_counts.std()),
        "unique_heads": int((counts_per_head > 0).sum()),
        "adjacent_jaccard_mean": float(adj.mean()) if len(adj) else float("nan"),
        "empty_step_fraction": float((per_step_counts == 0).mean()),
        "jaccard_vs_static_top20": float(np.mean(jac_static)) if jac_static else float("nan"),
        "activation_entropy": activation_entropy(counts_per_head),
        "entropy_baseline_ln20": float(np.log(20)),
    }
