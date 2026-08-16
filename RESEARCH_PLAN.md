# Research Plan — WakeKV (working title)

**Reversible, decode-time head-priority management for the KV cache:
demote cooling heads by offloading to CPU, detect heads waking up, and
promote them back — instead of evicting and losing their history.**

*Plan drafted 2026-07-06. Private — contains unpublished research strategy.*

---

## 0. STATUS (updated 2026-08-03)

The plan below was drafted around a **proactive** design (predict wake-ups,
prefetch ahead of need). Phases 0–1 are now done and **changed the design**:
the premise holds, but cheap wake-up prediction does not work, so WakeKV is
now a **reactive** system (demote to CPU, fetch on demand, measure the
stall). Read this section first; where the older sections below still say
"predict/prefetch," §0 overrides them.

**Done:**
- **Phase 0 — G0 PASS.** Heads churn during decoding on our models
  (Qwen2.5-3B, R1-Distill-1.5B, confirmed at 8B on R1-Distill-Llama-8B):
  63–85% of heads shift at least once, adjacent-step Jaccard ~0.5
  (continuous score), strongest on long CoT. Premise confirmed. →
  `notes/phase01_results.md`.
- **Phase 1 — G1 resolved to REACTIVE.** Cheap per-head wake-up signals
  can't predict well enough to prefetch (deployable causal-z-score:
  precision 0.06–0.20, recall 0.41–0.72). Wake-ups *are* temporally bursty
  (CoT mean z +9.7 vs. null) but not tightly/detectably enough for a coarse
  boundary trigger either. **Both proactive routes ruled out** → build
  reactive. The prediction negative + burstiness become supporting analysis.
- **Phase 2a — reactive-residency simulator.** (`wakekv/residency.py`)
  Reactive beats frozen (FlexiCache-style) at matched memory in every
  regime and scale tested — CoT, multi-turn, 1.5B/3B/8B all agree, roughly
  half the miss rate. C2 supported *in principle*. The simulator also now
  models destructive eviction (`simulate_evict`, ReasonAlloc-style) at the
  *same* matched budget as reactive — no interpolation needed, since both
  share the identical LRU eviction schedule and only diverge in what
  happens to a page after demotion. This gives a first, simulator-level C3
  read (reversible offload beats destructive eviction in principle) ahead
  of the real-system head-to-head in Phase 3.
- **Phase 2b — shim + preliminary real-system numbers (M2b-1a/1b/M2b-2).**
  `wakekv/flexicache_shim.py` turns FlexiCache into WakeKV reactive via two
  config overrides; verified correctness-preserving (M2b-1b, `identity`
  mode). On an A30 (Mistral-7B, LEval 8k→1000, 24 prompts), reactive beats
  stock FlexiCache's throughput at every rerank interval (1.34–2.33×) while
  holding 95–99% of LongBench quality, no monotonic erosion. →
  `notes/wakekv_m2b2_sweep_a30.md`, paper §8/Table 4.

**In progress / not yet started:**
- **Phase 2b remainder.** Real PCIe stall time is still only inferred
  through end-to-end throughput, not measured directly (M2c). Only one
  model (Mistral-7B) and one input/output regime have been tested — no 8B,
  no multi-turn. The matched sparse-only baseline ablation (isolating
  "less attention work per step" from "better memory management" in the
  throughput win) has a first reading at native rerank cadence ($R{=}16$,
  reusing existing sweep rows — no new run): dropping from 64 dense heads
  to 0 more than doubles throughput (2.33×) for a 5-point LongBench cost,
  so sparsity alone already accounts for most of the throughput headline
  and essentially all of the quality cost. →
  `notes/wakekv_m2b2_sweep_a30.md`. Still missing: the complementary cell
  (64 dense heads reranked every step) that would isolate
  rerank-frequency's own contribution independent of dense-head count.
- **Phase 3 (real eval) — not started.** No baseline reimplementations
  (ReasonAlloc, FullKV, SnapKV, uniform R-KV), no floor ablation (μ→0), no
  *real-system* evict-vs-offload head-to-head (C3 currently has a
  simulator-level reading only — see Phase 2a above), no full
  LongBench/RULER/SCBench harness beyond the narrow M2b-2 eval. This is
  where C3 and C5 get resolved on real systems and where the paper's
  final headline numbers come from.

### Evidence ladder — what each stage establishes

| Stage | Establishes | Status |
|---|---|---|
| Phase 0 | Premise: heads churn during decoding | ✅ done (G0 pass, incl. 8B) |
| Phase 1 | Cheap prediction fails → reactive design | ✅ done (G1 → reactive) |
| Phase 2a: simulator | Reactive beats frozen *in principle* (C2) and beats destructive eviction *in principle* (C3), on logged attention, at page granularity | ✅ done |
| Phase 2b: real system | Measured memory saved, PCIe **stall time**, throughput in vLLM | ⏳ preliminary (M2b-2 done incl. first sparse-only reading; stall time, scale, real-system C3, full sparse×rerank ablation remain) |
| Phase 3: evaluation | 7–8B models, LongBench/RULER/SCBench **quality**, real baselines | ❌ not started |

**The simulator is a gate, not a result.** It counts *misses*, not stall
*time*; works at page granularity approximated from top-k, not the full KV
cache; measures no *quality* (assumes reactive preserves accuracy by
fetching on demand); compares against a FlexiCache-*style* policy, not their
code; scout-scale models. Phase 2b's M2b-2 sweep has since supplied a first
real-system throughput+quality reading that confirms the direction — but
it's narrow (one model, one regime, one A30) and explicitly preliminary;
the full headline numbers — matched baselines, broader scale, measured
stall time — still depend on the rest of Phase 2b plus Phase 3.

---

## 1. One-line pitch and positioning

> Prior decode-time budget reallocation is evict-only (ReasonAlloc); prior
> reversible per-head offloading is static-role (FlexiCache, HeteroCache).
> WakeKV makes head priority both **dynamic** and **reversible**.

**Premise (already established, cite don't re-prove):** attention-head
importance is non-stationary within a single generation — the active
retrieval-head set churns step-to-step (adjacent-step Jaccard 0.28–0.51,
"Retrieval Heads are Dynamic", arXiv:2602.11162, ACL 2026), and that paper
explicitly names per-head dynamic KV retention/fetch as future work.

**Target:** workshop paper (4–9 pages). Primary: NeurIPS 2026 workshops
(deadlines ~late Aug–early Sep 2026). Backup: ICLR 2027 workshops
(~Feb 2027). arXiv preprint as soon as results hold — the window is
closing (three near-misses Nov 2025–Jun 2026).

**Compute envelope:** single 24–48 GB GPU + ≥64 GB host RAM throughout.
HeteroCache's measured envelope (18 GB CPU RAM, 21 GB/s PCIe @128K, runs
on RTX 4090 / PCIe Gen3) is the feasibility proof.

## 2. Claims the paper will make (and the experiment that backs each)

| # | Claim | Experiment | Status |
|---|-------|-----------|--------|
| C1 | Head priority shifts materially during decoding in our target regimes (long CoT, multi-turn) | Phase 0 churn measurement | ✅ supported (G0) |
| C2 | Frozen head classification leaves quality/efficiency on the table in those regimes | 2a simulator (miss-vs-memory Pareto), then 2b/E3a-b real system | ⏳ simulator |
| C3 | Reversible demotion (offload) beats evict-based demotion at equal GPU-resident budget | E3c: offload vs. evict, same budget | ❌ Phase 3 |
| ~~C4~~ | ~~Wake-ups detectable early enough to hide PCIe latency~~ | Phase 1 signal study | ❌ **refuted** → reactive |
| C5 | Reversibility permits aggressive budget floors that eviction cannot afford | E4 floor ablation (μ → 0) | ❌ Phase 3 |
| C6 | Overhead is acceptable (reactive fetch + accounting) | E5 systems accounting: stall time, PCIe, tokens/sec, block-level GPU bytes | ❌ Phase 2b/3 |

**C4 is refuted** (Phase 1): no cheap signal predicts wake-ups well enough
to prefetch, so WakeKV promotes *reactively*. The prediction negative
result is now itself a contribution (motivates the reactive design).

Minimum publishable unit = C1 + C2 + C3 on one model, one battleground
regime, with honest systems accounting. Everything else strengthens.

## 3. Method sketch (REACTIVE — revised after Phase 1)

*Supersedes the original proactive sketch. Phase 1 refuted C4, so there is
no prediction/prefetch step.*

State per KV head (GQA group): a GPU-resident set capped at `budget` pages +
a `reservoir` (full KV mirrored in pinned host memory — FlexiCache-style, so
demotion is never destructive).

Reactive loop, every decode step:

1. Each head attends its top-k pages (sparse decode, Quest/FlexiCache-style).
2. **Fetch on demand:** a wanted page sitting in the reservoir (CPU) is
   pulled to GPU now — a promotion, and a measured stall. Quality is
   preserved by construction (we always fetch what's needed); the cost is
   the stall, not accuracy.
3. **Demote** by recent demand: when a head's resident set exceeds `budget`,
   evict the least-recently-wanted pages to the reservoir. Reversible —
   nothing is lost.

Contrast with the two competitors this beats:
- **FlexiCache** fixes each head's residency by *offline* classification;
  it can't demote a head that cools mid-run or promote one that heats up.
- **ReasonAlloc** reallocates head budgets during decode but by *eviction*
  — a re-grown budget can't recover destroyed KV. Reactive keeps it on CPU.

Reversibility is what makes the LRU demotion safe (evicted ≠ lost) and lets
`budget` go aggressively low (a starved head is recoverable on demand).

GQA note: all decisions at KV-group granularity (Llama-3.1-8B: 8 groups ×
32 layers = 256 units; small models have as few as 4–8 groups — handled
explicitly, see risk R4).

## 4. Phases, milestones, decision gates

### Phase 0 — Premise validation on OUR models (Week 1)
- Reproduce 2602.11162's per-step head measurement on target models
  (Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Llama-8B; optionally
  Qwen2.5-3B) and target regimes (NIAH + one long-CoT task + one
  multi-turn trace). Use a continuous score variant too (their binary
  score has a thresholding-artifact caveat).
- Also log per-head top-K page sets over decode (FlexiCache's RCO
  harness) — same runs serve Phase 1.
- **Gate G0 — ✅ PASSED.** 63–85% of heads drift; adjacent Jaccard ~0.5.
  Ran on the 11 GB scout card (Qwen2.5-3B, R1-Distill-1.5B) rather than
  7–8B — those confirm at Phase 3. Details: `notes/phase01_results.md`.

### Phase 1 — Signal study, offline (Weeks 2–3)
- On Phase-0 logged traces, evaluate candidate wake-up signals (§5) for:
  precision/recall of predicting "head needs pages it doesn't hold",
  **lead time** (steps of advance warning) vs. PCIe transfer time for a
  typical head's working set, and compute cost.
- Characterize churn timing: correlated with reasoning transitions /
  turn boundaries? (Novel measurement on top of 2602.11162 — publishable
  content regardless.)
- **Gate G1 — ✅ RESOLVED → REACTIVE.** No signal predicts wake-ups well
  enough (causal z-score P 0.06–0.20, R 0.41–0.72); wake-ups are bursty but
  not tightly/detectably clustered (CoT z +9.7, but 2%-budget trigger
  catches only 54%). Both proactive routes ruled out → reactive design.
  Ran offline on Phase-0 logs (no extra GPU). Details:
  `notes/phase01_results.md`.

### Phase 2a — Reactive-residency simulator (this branch, PR #3) ⏳
- Cheap, no-GPU go/no-go gate before touching vLLM. `wakekv/residency.py`
  replays logged attention through Full / Frozen (FlexiCache-style) /
  Reactive (WakeKV LRU + fetch-on-demand), sweeping per-head budget →
  miss-rate-vs-memory Pareto.
- **Gate G2a:** at matched mean memory, does reactive miss less than frozen
  on the real CoT/multi-turn logs? Synthetic shifting-role check already
  confirms the harness (reactive 0.28@48pg vs frozen 0.75@146pg).
  - **Yes** → C2 supported in principle; proceed to 2b.
  - **No** → iterate the demotion policy in the fast simulator loop (not in
    vLLM) before committing engineering. Informative either way, ~0 GPU.
- Caveats (why this is a gate, not a result): counts misses not stall
  *time*; page granularity from top-k; no *quality* measured; FlexiCache-
  *style* not FlexiCache; scout scale. See §0 evidence ladder.

### Phase 2b — Real system (needs ≥24 GB GPU) ❌
- Fork FlexiCache (Apache 2.0, vLLM): keep MinMax score cache, per-head
  block tables, UVA transfer kernels. Replace the frozen 25% split with the
  reactive fetch-on-demand promotion + LRU demotion.
- Milestones: M2a outputs match dense within sparse-top-K tolerance; M2b
  end-to-end long-CoT run; M2c reactive overhead + real PCIe **stall time**
  measured.
- **Gate G2b:** measured memory saved and stall time in a real engine. If
  FlexiCache proves unworkable → HuggingFace-level harness with simulated
  paging + *measured* (not simulated) PCIe transfers; report both. ~40
  GPU-hours.

### Phase 3 — Evaluation (Weeks 6–9)
Battlegrounds (regimes where ALL frozen-role systems are untested):
- **E3a Long CoT:** R1-Distill-Llama-8B on MATH-500 + AIME 2024, budgets
  {128, 256, 512, 1024} — ReasonAlloc's own tables enable direct
  comparison.
- **E3b Multi-turn:** SCBench subset (Qwen2.5-7B); measure role churn +
  quality at turn boundaries.
- **E3c Reversibility head-to-head:** same scoring/cadence, demotion =
  {evict (ReasonAlloc-style), offload (ours)}; equal GPU-resident bytes.
- **E3d Sanity:** LongBench subset — goal is parity (expect ~no headroom
  there; say so explicitly rather than get caught).
- Ablations: signal choice (E2 winners), Δ cadence, floor μ (E4),
  reactive vs. predictive promotion.
- Systems accounting (E5): block-level GPU-resident bytes (tension #6 —
  never report token-level "savings"), PCIe GB, stall ms/token,
  tokens/sec, incident analysis ("head woke up; pages arrived in time?").
- Baselines: FullKV, SnapKV, uniform R-KV, ReasonAlloc-reimpl (no code
  released — reimplement its allocator on R-KV, validate against their
  published numbers), FlexiCache (their code, their profile), static-
  offline variant of our own system (controller frozen after prefill —
  the cleanest apples-to-apples static-vs-dynamic ablation).
  HeteroCache: compare via protocol/cited numbers; running their stack is
  optional stretch. ~120–150 GPU-hours.

### Phase 4 — Writing (Weeks 9–11)
- 4–6 page workshop draft. Figure 1 = head-priority trajectory plot with
  wake-up events and what each system does to that head's cache.
- Fresh arXiv sweep for concurrent work immediately before submission
  (watch: HeteroCache v3, FlexiCache follow-ups, anything citing
  2602.11162 + KV-cache).
- Post arXiv preprint as soon as internal results hold, even before the
  workshop deadline.

## 5. Candidate wake-up signals (Phase 1 menu — HISTORICAL)

*Kept for the record. Phase 1 tested these; none predicts well enough for
prefetch (§0), so the reactive design uses none of them as a trigger. The
k-step-ahead probe (last row) remains the only untested option and is
noted as possible future work, not the current plan.*


| Signal | Origin | Cost | Notes |
|---|---|---|---|
| Online RCO (overlap of consecutive top-K page sets) | FlexiCache (offline) | ~free at rerank | online use is novel |
| Windowed-median drift vs. baseline, debounced | HeteroCache (content refresh) | fires ~0.8% of steps | repurpose as role trigger |
| Layer-pooled utility count (importance+redundancy) | ReasonAlloc / R-KV | computed anyway | budget-sizing signal |
| Per-head attention entropy trend | EntropyInfer et al. | cheap | worth one column in E2 |
| Hidden-state MLP probe, trained k-step-ahead | 2602.11162 (k=0 only) | tiny MLP/step | k>0 probe is unbuilt anywhere → biggest novelty + gives explicit lead time; medium risk |

Plan: ship v1 with the cheapest signal that passes G1; the k-step-ahead
probe is the stretch goal / second paper if time runs out.

## 6. Risks and kill criteria

- **R1 Scoop (HIGH).** 2602.11162 names this as future work; field moves
  monthly. Mitigation: preprint fast; re-sweep arXiv at every phase gate;
  if scooped on the mechanism, pivot to the parts nobody will have —
  churn-timing characterization (Phase 1) + reversible-vs-evict ablation.
- **R2 No usable lead time (MEDIUM).** Wake-ups may be too sudden.
  Fallback: reactive promotion + honest stall accounting (FlexiCache
  already pauses requests ~1/16 of steps; matching that is acceptable).
- **R3 Headroom too small (MEDIUM).** If long-CoT/multi-turn show <1–2
  pt gains over frozen roles at matched budgets → kill C2 as headline,
  reframe around C3+C5 (reversibility enables aggressive floors /
  robustness), which needs only ReasonAlloc as the foil.
- **R4 GQA granularity (LOW-MED).** 4–8 KV groups on small models make
  per-head control coarse. Demo models chosen with ≥8 groups; report the
  granularity limit explicitly (BaKlaVa flags it — cite).
- **R5 Engineering sink (MED).** vLLM internals eat weeks. Mitigation:
  strict Phase-2 timebox (3 weeks), fallback harness pre-planned at G2.
- **Kill switch:** if G0 fails (no churn in our regimes) or both C2 and
  C3 come back negative by end of Week 7 → stop, write up the negative
  result + measurement study (still workshop-viable), and switch to T8
  (small-model stress test), which is fully shaped in the ideation notes.

## 7. Compute budget

| Phase | GPU-hours (est.) |
|---|---|
| P0 premise | ~10 |
| P1 signals | ~20 |
| P2 system dev | ~40 |
| P3 evaluation | ~120–150 |
| **Total** | **~200–220** (single 24–48 GB GPU, ~5–6 weeks wall-clock of partial utilization) |

## 8. Related-work positioning (one paragraph, ready to paste)

Static head-role methods (DuoAttention, RazorAttention, MoA, HeadKV,
FlexiCache, HeteroCache) fix each head's cache treatment offline or at
prefill; adaptive budget allocators (Ada-KV, CAKE, LAVa, DynamicKV) are
prefill-only despite their names; decode-time compressors (R-KV, G-KV,
ThinKV, SCOPE) evict under uniform per-head budgets; ReasonAlloc
reallocates head budgets during decoding but demotion permanently
destroys KV; offload systems (ArkVale, ShadowKV, InfiniGen, HeadInfer,
FlexiCache, HeteroCache) keep KV recoverable but never change head
priority at runtime — and 2602.11162 shows head roles churn within a
single generation and names dynamic per-head KV retention as an open
direction. WakeKV closes the intersection.

## 9. Supporting documents

- `notes/phase01_results.md` — **Phase 0–1 numeric record**: G0 tables, the
  G1 honesty ladder, the clustering finding, and the resolved reactive
  decision. The current-state-of-truth companion to §0.
- `docs/phase01_explainer.html` — plain-language interactive walkthrough of
  Phases 0–1.
- `wakekv/residency.py` + `scripts/simulate_residency.py` — the Phase 2a
  reactive-residency simulator.
- `notes/big_four_deep_read.md` — close reads of FlexiCache, HeteroCache,
  ReasonAlloc, Retrieval-Heads-are-Dynamic: attack surfaces, reusable
  components, all key numbers with sources.
- `notes/kv_cache_ideation_notes.txt` — ideation trail: the 8 field
  tensions, idea evolution v0→v1, idea tracker, lit-check verdict and
  competitor table.
