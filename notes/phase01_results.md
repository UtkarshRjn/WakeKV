# Phase 0–1 Results (record of numbers)

*The durable numeric record. Raw per-run `summary.md` / `signal_study.md`
live under gitignored `runs/` on the cluster; the numbers are transcribed
here so they survive. Plain-language walkthrough: `docs/phase01_explainer.html`.*

Hardware: single RTX 2080 Ti (11 GB), fp16 (float32 for R1-Distill —
bf16-native, fp16 overflows to NaN on Turing). Small-model scout scale.

---

## Gate G0 — does head importance shift during decoding?  **PASS**

Per-head churn measured with three metrics: adjacent-step Jaccard of the
active head set (2602.11162), FlexiCache RCO temporal stability (fraction
of heads with mean RCO < 0.5), HeteroCache drift events vs. prefill baseline.

| regime | model | runs | adj-Jaccard (binary) | unique heads | unstable frac (RCO<0.5) | heads w/ ≥1 drift event |
|---|---|---|---|---|---|---|
| NIAH | Qwen2.5-3B | 1 (shakedown, 64 steps) | 0.71 | 78 | 0.064 | 0.63 |
| CoT | R1-Distill-Qwen-1.5B | 5 (≤2048 steps) | 0.82 | 95 | 0.023 | **0.85** |
| Multi-turn | Qwen2.5-3B | 1 (long-recall) | 0.79 | 69 | 0.085 | 0.76 |

Notes:
- Under the **continuous** needle score (robust to the binary-threshold
  artifact), NIAH adjacent-step Jaccard is **0.45–0.57** — squarely inside
  2602.11162's reported 0.28–0.51 churn range, on our own model/hardware.
- Binary Jaccard reads higher on CoT/multi-turn because it measures literal
  copy of the *problem/fact* tokens (a narrow behavior); the drift metric —
  which is what a residency controller reacts to — shows 63–85% of heads
  shift at least once.
- The unstable-head fraction (~2–9%) echoes HeteroCache's ~7% "volatile"
  class, found independently with a different metric.

**Read:** heads are non-stationary within a single generation, strongest in
CoT — exactly the regime where frozen-role competitors (FlexiCache,
HeteroCache) are untested. Premise holds.

---

## Gate G1 — can a cheap runtime signal predict wake-ups early enough to prefetch?  **NOT with cheap signals**

A "wake-up" = a head's continuous needle score jumps quiet→active
(hysteresis: ≥8 steps below lo, then above hi). A signal "predicts" it if
it crosses threshold within the lead window before the event. Transfer bar
(64 pages, 1 KV head, 21 GB/s, 30 ms/step) rounds to **< 0.01 decode steps**,
so any lead ≥ 1 clears the *latency* bar — the question is purely predictive
quality.

Four candidate signals: `online_rco` (FlexiCache statistic, online),
`drift` (HeteroCache trigger), `entropy_trend`, `needle_mass_delta`.

### The honesty ladder (same signal, three gradings)

1. **Per-head best threshold** (oracle — tunes a cutoff per head *after*
   seeing labels; not deployable): drift looks perfect — P 1.00 / R 1.00 at
   lead 8 on CoT. Mirage.
2. **Single fixed global threshold** (one raw cutoff for all heads — too
   strict; heads churn at different baseline rates): precision-starved, no
   signal clears a useful bar. F1 ≈ 0.15 (CoT) / 0.26 (multi-turn) best.
3. **Causal per-head z-score** (each head normalized vs. its own trailing
   window=32, then one global z>2.0 — the deployable middle): the real number.

### Deployable result (causal z-score, z>2.0, lead 32, best-F1 signal)

**CoT — 407 events:**

| signal | precision | recall | alarms |
|---|---|---|---|
| online_rco | 0.06 | 0.69 | 10703 |
| **drift** | **0.08** | **0.41** | **10305** |
| entropy_trend | 0.06 | 0.45 | 4377 |
| needle_mass_delta | 0.06 | 0.72 | 8383 |

Winner: drift, F1 **0.14**.

**Multi-turn — 73 events:**

| signal | precision | recall | alarms |
|---|---|---|---|
| online_rco | 0.13 | 0.67 | 687 |
| **drift** | **0.20** | **0.70** | **1208** |
| entropy_trend | 0.20 | 0.56 | 353 |
| needle_mass_delta | 0.10 | 0.52 | 779 |

Winner: drift, F1 **0.31**.

**Read:** the best deployable signal is a smoke detector that both cries
wolf (precision 0.06–0.20 → 80–94% of alarms false) *and* misses fires
(recall 0.41–0.72). Threshold sweeps won't rescue a precision of 0.06.
For WakeKV specifically this bites twice: low recall → stalls on missed
wake-ups; low precision → constant over-promotion → heads stay GPU-resident
→ the CPU-offload memory saving evaporates.

---

## Verdict and the fork

- **G0 PASS**: reversible dynamic per-head residency targets a real
  phenomenon. Core novelty (vs. FlexiCache frozen roles / ReasonAlloc
  evict-only) is intact.
- **G1 no-cheaply**: proactive predict-and-prefetch fails with cheap
  statistics — a clean, publishable negative result.

Two ways forward:
1. **Reactive (R2 fallback)** — demote to CPU, fetch on demand when a head
   reaches for offloaded pages, measure the stall honestly. Fully supported
   by data in hand; still occupies the novel gap.
2. **Coarse proactive** — pending check: are wake-ups *temporally clustered*
   (many heads waking at reasoning-phase boundaries)? If yes, one boundary
   trigger could recover high recall at low alarm count where per-head
   z-scores drowned. ~20 min, no GPU, on existing logs. Decides the fork.

## Caveats
- Scout scale: 1–3 model sizes, n=1 for NIAH/multi-turn, n=5 for CoT
  (407 events gives the CoT G1 numbers real weight; multi-turn's 73 are
  coarser). Confirm on 7–8B before any paper claim.
- All on an 11 GB Turing card; PCIe/throughput numbers are the *bar*, not
  measured system performance (that's Phase 2/3).
