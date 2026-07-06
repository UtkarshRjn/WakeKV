# Research Plan — WakeKV (working title)

**Reversible, decode-time head-priority management for the KV cache:
demote cooling heads by offloading to CPU, detect heads waking up, and
promote them back — instead of evicting and losing their history.**

*Plan drafted 2026-07-06. Private — contains unpublished research strategy.*

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

| # | Claim | Experiment |
|---|-------|-----------|
| C1 | Head priority shifts materially during decoding in our target regimes (long CoT, multi-turn) | E0 churn measurement + E1 trajectory plots (nobody has published one) |
| C2 | Frozen head classification leaves quality/efficiency on the table in those regimes | E3a/E3b: FlexiCache-style static split vs. our controller, long-CoT + multi-turn |
| C3 | Reversible demotion beats evict-based dynamic reallocation at equal GPU-resident budget | E3c: WakeKV vs. ReasonAlloc-style eviction (same utility score, same cadence) |
| C4 | Wake-ups are detectable early enough to hide PCIe latency | E2 signal study: lead-time vs. transfer-time analysis |
| C5 | Reversibility permits aggressive budget floors that eviction cannot afford | E4 floor ablation (μ → 0) |
| C6 | Overhead is negligible (controller + transfers) | E5 systems accounting: stall time, PCIe traffic, tokens/sec, block-level GPU bytes |

Minimum publishable unit = C1 + C2 + C3 on one model, one battleground
regime, with honest systems accounting. Everything else strengthens.

## 3. Method sketch (v0 design, to be refined in Phase 1)

State per KV head (GQA group): `resident_budget` (pages on GPU) +
`reservoir` (full KV mirrored in pinned host memory — FlexiCache-style,
so demotion is never destructive).

Controller loop, every Δ decode steps (start Δ=128 per ReasonAlloc's
proven ~0%-overhead cadence; ablate {32, 64, 128, 256}):

1. **Score heads** with the layer-pooled utility used by ReasonAlloc
   (R-KV importance+redundancy, KthLargest threshold, count-above-τ) —
   reused verbatim for apples-to-apples comparison.
2. **Detect drift/wake-ups** with the cheap signals that come free at
   rerank time (candidates in §5; final choice from E2).
3. **Demote** cooling heads: shrink resident budget, offload surplus pages
   (async, low-priority stream). Nothing is lost — reservoir keeps all.
4. **Promote** warming heads: prefetch their high-score pages from
   reservoir ahead of need (lead time from the early-warning signal);
   grow resident budget.
5. Budget floor μ can be near zero — a starved head is recoverable.

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
- **Gate G0:** meaningful churn on our models/regimes (e.g., adjacent-step
  overlap well below 1, and churn concentrated at detectable events).
  If heads are static in our regimes → idea dies cheaply; pivot to T8
  (small-model study) from the ideation notes. ~10 GPU-hours.

### Phase 1 — Signal study, offline (Weeks 2–3)
- On Phase-0 logged traces, evaluate candidate wake-up signals (§5) for:
  precision/recall of predicting "head needs pages it doesn't hold",
  **lead time** (steps of advance warning) vs. PCIe transfer time for a
  typical head's working set, and compute cost.
- Characterize churn timing: correlated with reasoning transitions /
  turn boundaries? (Novel measurement on top of 2602.11162 — publishable
  content regardless.)
- **Gate G1:** ≥1 signal whose lead time × decode-step time exceeds the
  transfer time of a typical promotion at our budgets. If no signal has
  enough lead time → fall back to reactive promotion + measure the stall
  cost honestly (the paper weakens but survives; FlexiCache pauses
  requests the same way). ~20 GPU-hours.

### Phase 2 — Minimal system (Weeks 3–6)
- Fork FlexiCache (Apache 2.0, vLLM): keep MinMax score cache, per-head
  block tables, UVA transfer kernels. Replace the frozen 25% split with
  the online controller hung off their existing rerank hook.
- Milestones: M2a controller runs (correctness: outputs match dense
  attention within sparse-top-K tolerance); M2b end-to-end long-CoT run;
  M2c controller overhead <2% tokens/sec at batch 8.
- Simplification permitted for a workshop: Python-level controller,
  custom kernels only where FlexiCache already provides them.
- **Gate G2:** M2a–M2c pass. If FlexiCache codebase proves unworkable →
  fallback harness: HuggingFace-level implementation with simulated
  paging + measured (not simulated) PCIe transfers; report both. ~40
  GPU-hours (mostly dev iterations).

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

## 5. Candidate wake-up signals (Phase 1 menu)

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

- `notes/big_four_deep_read.md` — close reads of FlexiCache, HeteroCache,
  ReasonAlloc, Retrieval-Heads-are-Dynamic: attack surfaces, reusable
  components, all key numbers with sources.
- `notes/kv_cache_ideation_notes.txt` — ideation trail: the 8 field
  tensions, idea evolution v0→v1, idea tracker, lit-check verdict and
  competitor table.
