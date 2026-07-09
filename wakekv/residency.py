"""Reactive-residency simulator (Phase 2, Option B).

Replays a Phase-0 log as a stream of per-step, per-head "wanted page" sets
(the pages a head attended that step, from the logged top-k) and simulates
KV-cache residency under three policies, at page granularity per head:

- **full**     — every seen page stays GPU-resident. Quality ceiling, memory
                 ceiling, zero misses. The reference point.
- **frozen**   — FlexiCache-style. Heads are classified stable/unstable ONCE,
                 offline, from their own churn. Unstable heads keep their full
                 KV resident; stable heads keep a fixed top-B set, refreshed
                 every R steps. The classification never changes during decode.
- **reactive** — WakeKV. Every head is capped at B resident pages chosen by
                 recent demand (LRU). A wanted page sitting on CPU is fetched
                 on demand (a decode stall) and promoted; the page it evicts
                 goes to the CPU reservoir and is never lost.

A "miss" = a wanted page that was not GPU-resident when the head needed it.
For reactive that means a fetch-on-demand stall; for frozen it means the
fixed classification mis-served the head. Sweeping the per-head budget B
traces a miss-rate-vs-memory Pareto curve; the C2 claim is that reactive
dominates frozen in shifting-role regimes (long CoT, multi-turn).

Faithful to the logged attention (not a model re-run). Pure Python + numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wakekv.metrics import jaccard

# A stream is: list over decode steps of (list over heads of frozenset[page]).
Stream = list[list[frozenset]]


@dataclass
class SimStats:
    policy: str
    budget: int | None
    steps: int
    total_accesses: int          # (head, step, wanted-page) tuples
    misses: int                  # wanted page not resident at need time
    fetches: int                 # CPU->GPU page loads (reactive: == misses)
    stall_steps: int             # decode steps with >=1 fetch (reactive)
    resident_page_steps: int     # sum over steps of total resident pages
    peak_resident_pages: int     # max total resident pages at any step

    @property
    def miss_rate(self) -> float:
        return self.misses / self.total_accesses if self.total_accesses else float("nan")

    @property
    def mean_resident_pages(self) -> float:
        return self.resident_page_steps / self.steps if self.steps else 0.0

    def as_row(self) -> dict:
        return {
            "policy": self.policy,
            "budget": self.budget,
            "miss_rate": round(self.miss_rate, 4),
            "mean_resident_pages": round(self.mean_resident_pages, 1),
            "peak_resident_pages": self.peak_resident_pages,
            "fetches": self.fetches,
            "stall_steps": self.stall_steps,
        }


def wanted_stream_from_log(topk_idx: np.ndarray, page_size: int = 16) -> tuple[Stream, int]:
    """[S, L, H, K] top-k token indices -> (stream, n_units).

    Pads are -1. A page id is ``token_index // page_size``. Each (step, head)
    becomes the frozenset of distinct pages that head attended that step.
    """
    S, L, H, K = topk_idx.shape
    flat = topk_idx.reshape(S, L * H, K)
    n_units = L * H
    stream: Stream = []
    for s in range(S):
        row = []
        for u in range(n_units):
            idx = flat[s, u]
            pages = (idx[idx >= 0] // page_size)
            row.append(frozenset(pages.tolist()))
        stream.append(row)
    return stream, n_units


def classify_unstable(stream: Stream, n_units: int, unstable_frac: float = 0.25) -> set[int]:
    """FlexiCache-style offline split: the least-stable ``unstable_frac`` of
    heads (lowest mean consecutive-step Jaccard of their wanted pages)."""
    jac = np.ones(n_units)
    for u in range(n_units):
        sets = [stream[t][u] for t in range(len(stream))]
        vals = [
            jaccard(sets[t], sets[t + 1])
            for t in range(len(sets) - 1)
            if sets[t] or sets[t + 1]
        ]
        jac[u] = float(np.mean(vals)) if vals else 1.0
    k = int(round(n_units * unstable_frac))
    return set(np.argsort(jac)[:k].tolist()) if k else set()


def _tally_memory(resident: list[dict], stats: SimStats) -> None:
    total = sum(len(r) for r in resident)
    stats.resident_page_steps += total
    stats.peak_resident_pages = max(stats.peak_resident_pages, total)


def simulate_full(stream: Stream, n_units: int) -> SimStats:
    """Keep every seen page resident forever. Zero misses; memory ceiling."""
    resident = [dict() for _ in range(n_units)]
    st = SimStats("full", None, len(stream), 0, 0, 0, 0, 0, 0)
    for t, row in enumerate(stream):
        for u, wanted in enumerate(row):
            for p in wanted:
                st.total_accesses += 1
                resident[u][p] = t
        _tally_memory(resident, st)
    return st


def simulate_reactive(stream: Stream, n_units: int, budget: int) -> SimStats:
    """Per-head LRU cap of ``budget`` pages; fetch-on-demand from CPU."""
    resident = [dict() for _ in range(n_units)]   # page -> last-used step (on GPU)
    reservoir = [set() for _ in range(n_units)]    # pages on CPU
    st = SimStats("reactive", budget, len(stream), 0, 0, 0, 0, 0, 0)
    for t, row in enumerate(stream):
        fetched_this_step = False
        for u, wanted in enumerate(row):
            for p in wanted:
                st.total_accesses += 1
                if p in resident[u]:
                    resident[u][p] = t
                elif p in reservoir[u]:
                    st.misses += 1
                    st.fetches += 1
                    fetched_this_step = True
                    reservoir[u].discard(p)
                    resident[u][p] = t
                else:
                    resident[u][p] = t  # brand-new page: normal cache growth
            while len(resident[u]) > budget:
                lru = min(resident[u], key=resident[u].get)
                del resident[u][lru]
                reservoir[u].add(lru)
        if fetched_this_step:
            st.stall_steps += 1
        _tally_memory(resident, st)
    return st


def simulate_frozen(
    stream: Stream,
    n_units: int,
    budget: int,
    unstable_frac: float = 0.25,
    refresh: int = 16,
) -> SimStats:
    """FlexiCache-style: unstable heads keep full KV resident; stable heads
    keep a fixed top-``budget`` set refreshed every ``refresh`` steps."""
    unstable = classify_unstable(stream, n_units, unstable_frac)
    resident = [dict() for _ in range(n_units)]       # page -> last-used step
    window_demand = [dict() for _ in range(n_units)]   # page -> count since refresh
    st = SimStats("frozen", budget, len(stream), 0, 0, 0, 0, 0, 0)
    for t, row in enumerate(stream):
        for u, wanted in enumerate(row):
            if u in unstable:
                for p in wanted:
                    st.total_accesses += 1
                    resident[u][p] = t          # keep everything resident
                continue
            # stable head: fixed resident set, misses when demand strays
            for p in wanted:
                st.total_accesses += 1
                if p in resident[u]:
                    resident[u][p] = t
                else:
                    st.misses += 1
                window_demand[u][p] = window_demand[u].get(p, 0) + 1
        # periodic rerank of stable heads to the window's top-``budget`` pages
        if (t + 1) % refresh == 0:
            for u in range(n_units):
                if u in unstable or not window_demand[u]:
                    continue
                top = sorted(window_demand[u], key=window_demand[u].get, reverse=True)[:budget]
                new_res = {p: t for p in top}
                st.fetches += sum(1 for p in new_res if p not in resident[u])
                resident[u] = new_res
                window_demand[u] = {}
        _tally_memory(resident, st)
    return st


def sweep(stream: Stream, n_units: int, budgets: list[int], **frozen_kw) -> list[SimStats]:
    """Full once, plus reactive and frozen at each budget."""
    out = [simulate_full(stream, n_units)]
    for b in budgets:
        out.append(simulate_reactive(stream, n_units, b))
        out.append(simulate_frozen(stream, n_units, b, **frozen_kw))
    return out
