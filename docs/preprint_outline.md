# WakeKV — Paper Outline (Phase 0–2a snapshot)

*Companion to `paper/wakekv.tex`. Per user decision 2026-07-09: **do not
publish on arXiv** — keep the LaTeX draft in-repo as a raw internal
artifact while Phase 2b is built. Rawness is expected; this outline
tells you what the draft covers and what still needs a pass.*

**Working title:** *WakeKV: Reactive, Reversible KV Residency for Heads
That Change Their Minds*

**Length:** 4–6 pages + refs (arXiv preprint, later target = NeurIPS 2026
Efficient ML workshop or ICLR 2027 workshop). This is deliberately a
*measurement + design-motivation* paper — the system evaluation belongs
to a follow-up.

**Explicit framing:** we do **not** claim a superior serving system. We
claim (i) heads shift during decoding, (ii) predicting the shifts cheaply
does not work, (iii) simulated reactive residency dominates simulated
frozen residency on the memory/miss Pareto — and (iv) this motivates a
reactive real system, whose construction and evaluation is future work.

---

## Abstract (draft)

Recent KV-cache optimizations classify attention heads once — offline
(DuoAttention, FlexiCache) or at prefill (Ada-KV, LAVa, HeteroCache) —
and then treat the classification as fixed for the rest of the
generation. We measure per-decode-step head activity across three
regimes (needle retrieval, long chain-of-thought, multi-turn recall) on
Qwen2.5-3B and DeepSeek-R1-Distill-Qwen-1.5B and find 63–85% of heads
shift their reading habits at least once within a single generation,
with adjacent-step Jaccard around 0.5 — replicating the retrieval-heads-
are-dynamic phenomenon on our hardware. Motivated by this, we then ask
whether cheap runtime signals can *predict* wake-ups with enough lead
time to prefetch the head's KV cache from CPU. Under a fair evaluation
(one global threshold per signal, causal per-head normalization) they
cannot: precision 0.06–0.20 at recall 0.41–0.72 across four candidates.
Wake-ups are temporally bursty (mean concentration z +9.7 vs. shuffle
null on CoT) but not tightly detectable enough for a coarse trigger
either. We therefore argue for a *reactive* design — demote cooling
heads to CPU and fetch on demand — and validate it against a
FlexiCache-style frozen policy in a page-granularity residency
simulator over the same logged attention. Reactive dominates frozen on
the miss-vs-memory Pareto at every operating point in both regimes,
achieving roughly half the miss rate at matched memory. Real-hardware
throughput/quality evaluation is left to a follow-up; the negative
prediction result and the reactive-vs-frozen Pareto stand on their own.

---

## Section plan

**1. Introduction (1 col)**
- Hook: KV-cache methods assume stable head roles. Recent work
  (2602.11162) shows they aren't. What follows?
- Contribution list: (i) churn replication + regime coverage nobody else
  has, (ii) honest negative result on cheap prediction, (iii) burstiness
  characterization, (iv) matched-memory simulation-Pareto motivating
  reactive residency.
- **Explicit non-contribution:** no serving-system results in this paper.

**2. Setup (0.5 col)**
- Models, tasks, hardware (RTX 2080 Ti, 11 GB), instrumentation
  (per-step per-head top-k logging, memory-safe design).

**3. Do heads shift during decoding? (1 col + Figure 1)**
- Adjacent-Jaccard (binary + continuous), FlexiCache-style RCO temporal
  stability, HeteroCache-style drift events — three metrics, one story.
- Regime table: NIAH / CoT / multi-turn.
- **Figure 1** — head-activity heatmap over decode steps (the "office
  building" picture from the explainer, cleaned up).

**4. Can we predict the shifts cheaply? (1.5 col + Figure 2)**
- The **honesty ladder**: per-head best θ (optimistic) → fixed global θ
  (over-fires) → causal per-head z-score (deployable).
- Four candidate signals × the ladder × transfer bar.
- **Table 1** — headline P/R/F1 at deployable threshold; brief mention
  that per-head-best looks perfect and why that's misleading.
- **Figure 2** — Pareto of precision vs recall at lead ≥ transfer bar,
  reactive vs proactive design implication.

**5. Are wake-ups clustered? (0.75 col + Figure 3)**
- Per-step histogram, concentration z vs shuffle null, boundary-recall
  vs top-frac.
- Real burstiness (z +9.7) but not detectable enough for coarse
  prefetch. Also motivates reactive.
- **Figure 3** — CoT wake-up timeline for one representative run.

**6. Reactive residency wins on the memory/miss Pareto (1.25 col + Figure 4)**
- Simulator description (Full / Frozen / Reactive at page granularity).
- **Figure 4** — matched-memory Pareto curve, both regimes.
- **Table 2** — headline numbers: at frozen's operating memory, reactive
  miss rate is ~half.
- The **miss-asymmetry footnote**: reactive miss = paid fetch stall
  (quality preserved); frozen miss = quality gap. Reactive trades memory
  for stalls, not for accuracy — the honest framing.

**7. Related work (0.5 col)**
- Paste the paragraph from RESEARCH_PLAN.md §8 verbatim (static role
  methods, adaptive prefill-only allocators, decode-time compressors,
  offload systems, ReasonAlloc).

**8. Limitations & future work (0.5 col — DO NOT SKIP)**
- Scout-scale models (1.5–3B); Phase 2b/3 confirm at 7–8B.
- Simulator counts misses, not real PCIe stall time.
- Simulator vs. FlexiCache-style policy, not FlexiCache's code.
- No quality measurement yet — reactive assumed to preserve quality by
  fetch-on-demand construction; needs empirical confirmation on
  LongBench/RULER (future work).
- Small n (5 CoT + 1 multi-turn); reviewer will want more.
- Future work: real system on ≥24 GB GPU, real PCIe measurements, real
  quality benchmarks. State this explicitly — it's Phase 2b/3.

---

## Figures the paper needs

| # | Figure | Data source | Status |
|---|---|---|---|
| 1 | Per-step head activity heatmap (CoT run) | Phase 0 logs | need to generate |
| 2 | Precision-recall Pareto of signals @ deployable threshold | Phase 1 z-score data | need to generate |
| 3 | CoT wake-up timeline: per-step count + fitted null | Clustering analysis | need to generate |
| 4 | Matched-memory Pareto: reactive vs frozen, both regimes | Phase 2a simulator | need to generate |

All four are pure-numpy scripts against existing logs; no GPU needed.

## Tables

| # | Table | Content |
|---|---|---|
| 1 | Regime × churn-metric | Sec 3 headline |
| 2 | Deployable signal P/R/F1 | Sec 4 headline |
| 3 | Matched-memory simulator | Sec 6 headline |

---

## Concurrent-work checks (do THE DAY of preprint submission)

- arXiv sweep: "KV cache", "attention head", "reactive", "offload",
  "reversible", cited-by tree of 2602.11162, HeteroCache, FlexiCache,
  ReasonAlloc, ThinKV.
- Especially watch: any paper claiming "dynamic head classification" or
  "reactive KV residency" appearing after May 2026.

---

## What we intentionally leave out (for Phase 2b paper)

- Real-system throughput / tokens/sec.
- Real PCIe stall time on measured hardware.
- Quality metrics on LongBench/RULER/SCBench.
- Head-to-head against FlexiCache's real code.
- Ablations on demotion policy variants (LRU vs. LFU vs. attention-score).

These are the *next* paper, not this one.

---

## Timeline (light-touch)

- **Day 1–2:** figures generation (all four scripts) + section drafting
  in a fresh `docs/preprint/` directory.
- **Day 3–4:** self-review, tighten abstract, related-work paragraph,
  limitations section (the most important section — reviewers read it
  before the intro).
- **Day 5:** re-run the arXiv concurrent-work sweep, then submit.

Total ~1 week wall-clock of writing while Phase 2b setup happens in parallel.
