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
- **evict**    — LRU-cap ablation. Identical LRU cap and eviction rule to
                 reactive (same B, same order), but demotion is destructive:
                 an evicted page has nowhere to go and is simply gone. A
                 later want of it is not a one-time recoverable stall — it
                 misses every time, forever, since nothing brings it back.
                 A controlled foil to isolate "reversibility" as a
                 variable — NOT a reimplementation of any specific paper
                 (see `simulate_reasonalloc` for that).

Plus three faithful Phase-3-baseline reimplementations, each reusing this
module's page-granular Stream where the method's own algorithm allows it,
and a companion `page_score_stream_from_log` for the two that need
per-page attention SCORES (not just wanted/not-wanted):

- **snapkv**      — Li et al., NeurIPS 2024 (arXiv:2404.14469). Frozen,
                    ONE-SHOT prefill selection by observation-window
                    attention vote; never reranks.
- **rkv_uniform** — Cai et al., arXiv:2505.24133 ("uniform" = R-KV's own
                    behavior: one global budget shared by every head/layer,
                    no per-head adaptivity). Destructive, periodic,
                    importance-ranked pruning.
- **reasonalloc** — Liu et al., arXiv:2606.11164v1 (no official code).
                    Destructive, but the budget itself is reallocated
                    PER HEAD every Δ steps by an importance-threshold
                    rule — the actual decode-time-dynamic-but-evict-only
                    contrast case for WakeKV's claim (`evict` above is a
                    plain-LRU ablation foil, not this).

See each function's docstring for the paper's real algorithm, exactly what
was simplified to fit this page-granular, already-logged-attention
simulator, and why.

A "miss" = a wanted page that was not GPU-resident when the head needed it.
For reactive that means a fetch-on-demand stall; for frozen it means the
fixed classification mis-served the head; for evict it means a permanent
quality gap with no recovery path at all. Sweeping the per-head budget B
traces a miss-rate-vs-memory Pareto curve; the C2 claim is that reactive
dominates frozen in shifting-role regimes (long CoT, multi-turn); the C3
claim (reactive vs. evict, same B — no memory-matching interpolation
needed, since both share the identical LRU cap) is that reversibility
itself is what wins, not just dynamism.

Faithful to the logged attention (not a model re-run). Pure Python + numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wakekv.metrics import jaccard

# A stream is: list over decode steps of (list over heads of frozenset[page]).
Stream = list[list[frozenset]]

# A scored stream carries the same page-presence information PLUS per-page
# attention weight: list over decode steps of (list over heads of
# {page: summed attention weight}). Needed by policies that select or
# threshold on attention magnitude, not just wanted/not-wanted
# (SnapKV, R-KV, ReasonAlloc) -- see `page_score_stream_from_log`.
ScoredStream = list[list[dict[int, float]]]


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


def page_score_stream_from_log(
    topk_idx: np.ndarray, topk_val: np.ndarray, page_size: int = 16
) -> tuple[ScoredStream, int]:
    """[S, L, H, K] top-k token indices + attention weights -> (scored
    stream, n_units). Page-granular analogue of ``wanted_stream_from_log``
    for policies that need attention MAGNITUDE, not just page presence
    (SnapKV's observation-window vote; R-KV/ReasonAlloc's importance
    term). A page's score = sum of the logged top-k attention weights of
    the tokens within it that made the logged top-k (a page/token outside
    the logged top-k contributes 0 -- same truncation caveat as
    ``wanted_stream_from_log``; this also stands in for the papers'
    explicit pooling/clustering step, since a page IS the clustering unit
    at this granularity).
    """
    S, L, H, K = topk_idx.shape
    idx_flat = topk_idx.reshape(S, L * H, K)
    val_flat = topk_val.reshape(S, L * H, K)
    n_units = L * H
    stream: ScoredStream = []
    for s in range(S):
        row: list[dict[int, float]] = []
        for u in range(n_units):
            idx = idx_flat[s, u]
            val = val_flat[s, u]
            mask = idx >= 0
            pages = (idx[mask] // page_size).tolist()
            weights = val[mask].tolist()
            scores: dict[int, float] = {}
            for p, w in zip(pages, weights):
                scores[p] = scores.get(p, 0.0) + float(w)
            row.append(scores)
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
    """Per-head LRU cap of ``budget`` pages; fetch-on-demand from CPU.

    Thin driver around ``ReactiveController`` — same state machine backs
    both the simulator here and the FlexiCache monkey-patch shim in PR #8,
    so simulator numbers and real-system numbers use identical bookkeeping.
    """
    from wakekv.reactive_controller import ReactiveController

    ctrl = ReactiveController(n_units, budget, rerank_interval=1)
    for t, row in enumerate(stream):
        ctrl.begin_step(t)
        for u, wanted in enumerate(row):
            ctrl.on_unit_lru(u, set(wanted))
        ctrl.end_step()

    s = ctrl.stats
    return SimStats(
        policy="reactive",
        budget=budget,
        steps=s.steps,
        total_accesses=s.total_wanted,
        misses=s.total_misses,
        fetches=s.total_fetches,
        stall_steps=s.stall_steps,
        resident_page_steps=s.resident_page_steps,
        peak_resident_pages=s.peak_resident_pages,
    )


def simulate_evict(stream: Stream, n_units: int, budget: int) -> SimStats:
    """Per-head LRU cap of ``budget`` pages, DESTRUCTIVE demotion (C3).

    Same eviction rule as ``simulate_reactive`` — LRU, same budget — so the
    two share an eviction *schedule*: what stays resident and when
    something gets pushed out is identical between them, since that's
    driven purely by the wanted-page stream and the cap, not by what
    happens to a page afterward. The only difference is what "afterward"
    means: reactive offloads to a reservoir (recoverable, one stall);
    evict just drops it (unrecoverable, ReasonAlloc-style — "demotion
    permanently destroys KV").

    Consequently a page that comes back after being evicted is not a
    one-time cost here — it misses on *every* subsequent want, forever,
    because nothing ever re-admits it. This is deliberate: it's what makes
    the miss RATE actually diverge from reactive's (a naive
    reservoir-vs-nothing swap with instant "recompute" on re-want would
    produce an identical miss rate to reactive, since residency-set
    evolution only depends on the LRU rule, not on eviction's
    consequence — that would silently hide the exact effect C3 exists to
    measure).
    """
    resident: list[dict[int, int]] = [dict() for _ in range(n_units)]
    destroyed: list[set[int]] = [set() for _ in range(n_units)]
    st = SimStats("evict", budget, len(stream), 0, 0, 0, 0, 0, 0)
    for t, row in enumerate(stream):
        for u, wanted in enumerate(row):
            res = resident[u]
            dead = destroyed[u]
            for p in wanted:
                st.total_accesses += 1
                if p in res:
                    res[p] = t
                elif p in dead:
                    st.misses += 1  # gone for good; not re-admitted
                else:
                    res[p] = t  # true first-touch: ordinary cache growth
            if len(res) > budget:
                excess = len(res) - budget
                oldest = sorted(res.keys(), key=res.__getitem__)[:excess]
                for p in oldest:
                    del res[p]
                    dead.add(p)  # destructive: no reservoir, gone forever
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


def simulate_snapkv(
    stream: Stream,
    n_units: int,
    budget: int,
    prefill_scores: list[dict[int, float]],
) -> SimStats:
    """SnapKV (Li et al., "SnapKV: LLM Knows What You Are Looking for
    Before Generation," NeurIPS 2024, arXiv:2404.14469; official code
    github.com/FasterDecoding/SnapKV, Apache 2.0). Verified against the
    paper's Section 3, not paraphrase: attention from queries in an
    "observation window" (the prompt's last W tokens) is summed per
    prefix position, a 1D pooling layer ("Apply 1D pooling for
    clustering," kernel_size, stride 1) is applied to the vote before
    top-k, and the resulting top-k=floor(p*L_prefix) prefix positions per
    head are kept. That selection is FROZEN through decoding; the
    observation window and every subsequently generated token are always
    kept too (appended every step, uncompressed) -- selection never
    reranks.

    Simplifications from what wakekv/instrument.py actually logs:
    - The log's step 0 is a SINGLE query (the last prompt token attending
      over the whole prompt), not SnapKV's multi-token observation window
      -- there is no other pre-generation query in the log. This is a
      W=1 degenerate case of the paper's pooling, not the true window;
      flagged, not silently substituted.
    - The paper's explicit 1D max-pool "clustering" kernel over raw token
      positions is replaced by this simulator's existing page-granular
      aggregation (`page_score_stream_from_log` sums attention weight
      within a page) -- a page already IS the clustering unit here, so a
      separate pooling pass would double up, not add fidelity.
    - `prefill_scores[u]` is unit u's page->weight dict from step 0 of a
      `page_score_stream_from_log` stream (the observation-window vote).
      The prompt/generation boundary isn't threaded through the Stream
      abstraction, so it's inferred per unit as
      ``max(prefill_scores[u]) + 1`` (a page id at or above that is
      treated as post-prompt and always retained) -- a documented
      heuristic, reasonable because decode queries have strong recency
      bias so the true last-prompt page is very likely in step 0's top-k.

    A miss here is PERMANENT like `simulate_evict`, not a recoverable
    stall: SnapKV never reranks after prefill, so a prompt page excluded
    from the initial top-`budget` selection misses on every later want.
    """
    resident: list[dict[int, int]] = []
    boundary: list[int] = []
    for u in range(n_units):
        scores = prefill_scores[u] if u < len(prefill_scores) else {}
        top = sorted(scores, key=scores.get, reverse=True)[:budget]
        resident.append({p: -1 for p in top})
        boundary.append((max(scores) + 1) if scores else 0)

    st = SimStats("snapkv", budget, len(stream), 0, 0, 0, 0, 0, 0)
    for t, row in enumerate(stream):
        for u, wanted in enumerate(row):
            res = resident[u]
            b = boundary[u]
            for p in wanted:
                st.total_accesses += 1
                if p in res:
                    res[p] = t
                elif p >= b:
                    res[p] = t  # post-prompt page: SnapKV always retains generated tokens
                else:
                    st.misses += 1  # frozen prefill selection excluded it; never recovers
        _tally_memory(resident, st)
    return st


def simulate_rkv_uniform(
    scored_stream: ScoredStream,
    n_units: int,
    budget: int,
    buffer: int = 128,
) -> SimStats:
    """Uniform R-KV (Cai et al., "R-KV: Redundancy-aware KV Cache
    Compression for Reasoning Models," arXiv:2505.24133, NeurIPS 2025;
    official code github.com/Zefan-Cai/R-KV). Verified against the
    paper's Method section, not paraphrase: importance I is the mean
    attention weight from a trailing observation window (Eq. 4);
    redundancy R is a softmax-normalized mean cosine similarity between a
    candidate token's key vector and every other candidate's key vector
    (Eq. 5); the joint score is Z = lambda*I - (1-lambda)*R, lambda=0.1
    default (Eq. 7); recompression is buffer-triggered, every
    B_buffer=128 newly generated tokens, dropping to a fixed B_budget by
    keeping the highest-Z tokens (destructive, no reservoir -- the paper
    never mentions offload/recovery).

    AMBIGUITY FLAG ("uniform R-KV" is RESEARCH_PLAN.md's own naming, not
    the paper's): resolved by reading BOTH R-KV and ReasonAlloc
    (arXiv:2606.11164v1) together -- R-KV itself never varies its budget
    per head/layer (one global B_budget, applied identically everywhere);
    ReasonAlloc's own Section 6 explicitly frames its contribution as
    reusing R-KV's Eq. 7 score under ITS OWN adaptive per-head
    reallocation (see `simulate_reasonalloc`). So "uniform R-KV" = R-KV
    exactly as published: same score, ONE fixed budget shared by every
    head, no adaptivity. This reading is well-supported, not a guess.

    SIMPLIFICATION (a real data gap, not a judgment call): the redundancy
    term needs raw per-token KEY VECTORS to compute cosine similarity.
    wakekv/instrument.py deliberately never logs those -- only top-k
    attention weights/indices are kept, to bound memory (see its module
    docstring: "we never accumulate full rows"). No Phase-0 log this repo
    has can reconstruct redundancy. This implements IMPORTANCE ONLY
    (Eq. 4; equivalent to lambda=1 in Eq. 7) -- an honest partial
    reimplementation of R-KV's joint score, not the full method. Not
    silently patched with a fabricated redundancy proxy; a faithful
    redundancy term needs a fresh instrumentation pass that also logs key
    vectors (out of scope here -- flagged for Phase 3 planning).

    Eviction is destructive, like `simulate_evict`, but the RULE differs:
    this drops the lowest-accumulated-importance pages within a periodic
    buffer window, not the least-recently-wanted ones.
    """
    resident: list[dict[int, float]] = [dict() for _ in range(n_units)]
    destroyed: list[set[int]] = [set() for _ in range(n_units)]
    st = SimStats("rkv_uniform", budget, len(scored_stream), 0, 0, 0, 0, 0, 0)
    for t, row in enumerate(scored_stream):
        for u, scores in enumerate(row):
            res = resident[u]
            dead = destroyed[u]
            for p, w in scores.items():
                st.total_accesses += 1
                if p in res:
                    res[p] += w
                elif p in dead:
                    st.misses += 1  # compressed away earlier; R-KV has no reservoir
                else:
                    res[p] = w  # first-touch: freshly computed KV, ordinary growth
        if (t + 1) % buffer == 0:
            for u in range(n_units):
                res = resident[u]
                if len(res) > budget:
                    excess = len(res) - budget
                    weakest = sorted(res.keys(), key=res.__getitem__)[:excess]
                    for p in weakest:
                        del res[p]
                        destroyed[u].add(p)
                for p in res:
                    res[p] = 0.0  # reset the trailing-importance window
        _tally_memory(resident, st)
    return st


def _reasonalloc_head_budgets(raw_demand: list[float], b_layer: int, mu: float) -> list[int]:
    """Per-head budgets within one layer's fixed total `b_layer`, floored
    at `mu * b_layer / H` and rescaled to sum exactly to `b_layer`.

    Simplified stand-in for ReasonAlloc's robustification pipeline
    (power-smoothing exponent 0.5, clip to [0.25, 2] x mean, rescale +
    round -- Eqs. 1-3, reused for both the offline layer split and this
    online head split per the paper's Section 5.3). We keep the two
    properties the paper's own ablation motivates as essential (a
    starvation-proof floor; reallocation proportional to demand) and drop
    the exact smoothing curve, which the paper itself justifies only
    qualitatively ("stabilizes... prevents oscillation") -- a secondary
    numerical detail, not the mechanism under test.
    """
    h = len(raw_demand)
    if h == 0 or b_layer <= 0:
        return [0] * h
    b_min = mu * b_layer / h
    if b_min * h >= b_layer:  # floor alone already exhausts the layer budget
        return _round_preserving_sum([b_layer / h] * h, b_layer)
    remaining = b_layer - b_min * h
    excess = [max(d - b_min, 0.0) for d in raw_demand]
    total_excess = sum(excess)
    if total_excess > 0:
        alloc = [b_min + remaining * e / total_excess for e in excess]
    else:
        alloc = [b_min + remaining / h] * h
    return _round_preserving_sum(alloc, b_layer)


def _round_preserving_sum(alloc: list[float], total: int) -> list[int]:
    floored = [int(a) for a in alloc]
    remainder = total - sum(floored)
    order = sorted(range(len(alloc)), key=lambda i: alloc[i] - floored[i], reverse=True)
    for i in order[:remainder]:
        floored[i] += 1
    return floored


def simulate_reasonalloc(
    scored_stream: ScoredStream,
    n_units: int,
    n_heads: int,
    budget: int,
    delta: int = 128,
    mu: float = 0.25,
) -> SimStats:
    """ReasonAlloc (Liu et al., "ReasonAlloc: Hierarchical Decoding-Time
    KV Cache Budget Allocation for Reasoning Models," arXiv:2606.11164v1,
    Jun 2026 preprint). NO OFFICIAL CODE -- confirmed by RESEARCH_PLAN.md
    and a fresh arXiv/GitHub search as of this reimplementation (Aug
    2026); this is reconstructed directly from the paper text (Sections
    5.2/5.3, Algorithm 1), not from memory of the name.

    Real algorithm, two levels:
    (a) OFFLINE, once per architecture: a layer-wise "Reasoning Wave"
        budget from <=8 held-out probe prompts (attention-mass-
        concentration threshold rho=0.93), power-smoothed (exponent
        gamma=0.5), clipped to [0.25, 2] x mean, rescaled to a global
        budget B -- static per-layer budgets B^(l).
    (b) ONLINE, every Delta=128 decode steps (Algorithm 1, line 11): pool
        the utility score across every head in a layer -- "for all
        experiments we instantiate the generic utility score using the
        state-of-the-art redundancy-aware formulation from R-KV:
        S = alpha*I + (1-alpha)*R, alpha=0.1" (Section 6) -- set tau to
        the B^(l)-th largest pooled score, count each head's raw demand
        r_i = #{tokens with score >= tau}, then robustify with a
        head-level floor B_min = mu*B^(l)/H, mu=0.25 (Section 5.3, to
        prevent the starvation death-spiral the paper describes: a
        starved head loses the history needed to ever score well again).
        Eviction inside each head's new budget is delegated to the
        underlying policy -- in practice R-KV's top-k retention by the
        same score (destructive; the paper never mentions offload,
        reversibility, or CPU memory anywhere -- confirmed by direct
        reading of the arXiv HTML).

    This -- not `simulate_evict` -- is the paper-grounded contrast case
    for WakeKV's "prior decode-time reallocation is evict-only" claim:
    `simulate_evict` is a plain-LRU ablation foil (recency-triggered,
    fixed per-unit budget), explicitly not meant to stand in for any
    specific paper. ReasonAlloc's actual mechanism differs on both axes
    that matter for the claim: its reallocation TRIGGER is an
    importance-threshold recount (not LRU), and its per-head BUDGET
    itself moves over time (not fixed) -- while demotion, like
    `simulate_evict`, still permanently destroys the KV it drops.

    SIMPLIFICATIONS (declared, not silent):
    - (a), the offline Reasoning-Wave layer split, needs a separate
      probe-prompt calibration pass this simulator has no access to (it
      only replays one run's own already-logged attention). We substitute
      an EQUAL per-layer budget (B^(l) = budget * n_heads for every
      layer) and do NOT claim to reproduce the Reasoning-Wave shape. Only
      (b) -- the online head-wise reallocation, which is also the part
      that is actually decode-time-dynamic and the direct point of
      comparison against WakeKV -- is faithfully reimplemented.
    - the robustification pipeline is simplified; see
      `_reasonalloc_head_budgets`.
    - like `simulate_rkv_uniform`, the R-KV utility score's redundancy
      term needs raw key vectors this repo's logs never captured (see
      wakekv/instrument.py) -- this uses the importance term only, same
      flagged gap, propagated from R-KV since ReasonAlloc reuses it
      verbatim.

    Confidence: MEDIUM-HIGH on the online head-wise mechanism (read
    directly from Section 5.3/Algorithm 1); LOW/not-attempted on the
    offline layer split (substituted with a flat default rather than
    forced) -- flagged per RESEARCH_PLAN.md's "never fabricate, flag what
    is unverified" convention.
    """
    if n_heads <= 0 or n_units % n_heads != 0:
        raise ValueError(f"n_units ({n_units}) must be a multiple of n_heads ({n_heads})")
    n_layers = n_units // n_heads
    b_layer = budget * n_heads  # simplification: equal per layer, see docstring

    resident: list[dict[int, float]] = [dict() for _ in range(n_units)]
    destroyed: list[set[int]] = [set() for _ in range(n_units)]

    st = SimStats("reasonalloc", budget, len(scored_stream), 0, 0, 0, 0, 0, 0)
    for t, row in enumerate(scored_stream):
        for u, scores in enumerate(row):
            res = resident[u]
            dead = destroyed[u]
            for p, w in scores.items():
                st.total_accesses += 1
                if p in res:
                    res[p] += w
                elif p in dead:
                    st.misses += 1
                else:
                    res[p] = w

        if (t + 1) % delta == 0:
            for layer in range(n_layers):
                units = range(layer * n_heads, (layer + 1) * n_heads)
                pooled = sorted(
                    (score for u in units for score in resident[u].values()), reverse=True
                )
                tau = pooled[b_layer - 1] if 0 < b_layer <= len(pooled) else -float("inf")
                raw_demand = [
                    float(sum(1 for score in resident[u].values() if score >= tau))
                    for u in units
                ]
                new_budgets = _reasonalloc_head_budgets(raw_demand, b_layer, mu)
                for u, nb in zip(units, new_budgets):
                    res = resident[u]
                    if len(res) > nb:
                        excess = len(res) - nb
                        weakest = sorted(res.keys(), key=res.__getitem__)[:excess]
                        for p in weakest:
                            del res[p]
                            destroyed[u].add(p)
                    for p in res:
                        res[p] = 0.0  # reset the trailing-importance window
        _tally_memory(resident, st)
    return st


def sweep(
    stream: Stream,
    n_units: int,
    budgets: list[int],
    scored_stream: ScoredStream | None = None,
    n_heads: int | None = None,
    rkv_buffer: int = 128,
    reasonalloc_delta: int = 128,
    reasonalloc_mu: float = 0.25,
    **frozen_kw,
) -> list[SimStats]:
    """Full once, plus reactive, evict, and frozen at each budget. When
    `scored_stream` (see `page_score_stream_from_log`) is also supplied,
    additionally runs the three Phase-3 baseline reimplementations that
    need per-page attention SCORES, not just wanted/not-wanted sets:
    snapkv, rkv_uniform, and (if `n_heads` is known too) reasonalloc.
    Skipped when scores aren't available -- e.g. the synthetic page-only
    streams in tests/test_residency.py -- since there's no attention
    weight to select or threshold on. `rkv_buffer`/`reasonalloc_delta`/
    `reasonalloc_mu` forward to those two policies' own parameters
    (see their docstrings); irrelevant (and unused) when scores aren't
    supplied.
    """
    out = [simulate_full(stream, n_units)]
    for b in budgets:
        out.append(simulate_reactive(stream, n_units, b))
        out.append(simulate_evict(stream, n_units, b))
        out.append(simulate_frozen(stream, n_units, b, **frozen_kw))
        if scored_stream is not None:
            out.append(simulate_snapkv(stream, n_units, b, scored_stream[0]))
            out.append(simulate_rkv_uniform(scored_stream, n_units, b, buffer=rkv_buffer))
            if n_heads:
                out.append(simulate_reasonalloc(
                    scored_stream, n_units, n_heads, b,
                    delta=reasonalloc_delta, mu=reasonalloc_mu,
                ))
    return out
