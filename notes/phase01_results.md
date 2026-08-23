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

### Follow-up: are wake-ups temporally clustered? (decides proactive vs reactive)

`analyze_clustering.py` pools per-head wake events into a per-step histogram
and tests concentration in the busiest steps against a per-head uniform-
shuffle null.

CoT (5 runs), per-run z of concentration vs null:

| run | events | z | Fano | max co-wake |
|---|---|---|---|---|
| math500-0 | 28 | +0.1 | 1.0 | 2 |
| math500-1 | 217 | **+29.4** | 6.7 | 22 |
| math500-2 | 55 | +3.9 | 1.3 | 4 |
| math500-3 | 38 | +5.6 | 1.6 | 5 |
| math500-4 | 69 | +9.5 | 1.5 | 3 |

Mean z **+9.7** — wake-ups **are** genuinely bunched in time beyond chance
(4/5 runs significant; only the smallest run is Poisson). The earlier
"not clustered" read was a metric artifact.

But clustering isn't *tight* enough for a coarse trigger to be a clean win:

| trigger budget (top-frac of steps) | mean z | concentration ratio vs null | boundary-recall |
|---|---|---|---|
| 2% of steps | +9.7 | 1.65 | 0.54 |
| 5% of steps | +9.7 | 1.29 | 0.86 |

Fire narrow (2%) → catch only 54% of wake-ups. Fire wide (5%) → catch 86%
but the edge over random shrinks to 1.29x (5% of a long trace is a big
chunk). And the "busiest steps" are picked *after* seeing events — at
runtime you'd need a detectable boundary signal aligned with the bursts,
which is unshown. So a coarse boundary-prefetch trigger is **borderline,
not a clean win**.

### Fork resolved: REACTIVE

Both cheap proactive routes are ruled out:
- per-head signal (G1 z-score): precision 0.06–0.20 — no usable predictor.
- coarse boundary trigger (clustering): real bursts, but not tight or
  runtime-detectable enough.

**Decision: build WakeKV as a reactive system** — demote cooling heads to
CPU, fetch on demand when a head reaches for offloaded pages, measure the
promotion stall honestly (cf. FlexiCache, which pauses ~1/16 of the batch).
This still occupies the novel gap (dynamic per-head residency vs. frozen
FlexiCache / evict-only ReasonAlloc). The prediction negative result and
the burstiness finding become supporting analysis in the paper, motivating
the reactive choice.

The actual value of WakeKV is still unproven — it now rests on Phase 2/3
showing reactive dynamic residency beats FlexiCache in shifting-role
regimes (long CoT, multi-turn) at an acceptable stall cost.

## Caveats
- Scout scale: 1–3 model sizes, n=1 for NIAH/multi-turn, n=5 for CoT
  (407 events gives the CoT G1 numbers real weight; multi-turn's 73 are
  coarser). Confirm on 7–8B before any paper claim.
- All on an 11 GB Turing card; PCIe/throughput numbers are the *bar*, not
  measured system performance (that's Phase 2/3).

---

## Post-G1 follow-up candidates: ensemble vote + data-driven token correlation

**Status: RUN on real logs** (`bash scripts/wakekv_analyze_all_phase1.sh`,
4 model/task combos, 0 failures — full run reported in
[PR #18](https://github.com/UtkarshRjn/dynamic-head-kv/pull/18)). Method
recap below, then the actual result.

The G1 verdict (deployable causal-z-score precision 0.06–0.20 for every
individual signal) motivated the reactive design. Two follow-ups tested
whether that verdict leaves anything on the table: do the four failed
signals fail for *correlated* reasons (so combining them helps), and does
anything about *what the model is generating* — as opposed to what any one
head is attending to — predict a wake burst?

- **Ensemble vote** (`wakekv.signals.ensemble_vote`): each of the four
  signals is causally z-scored (as in the G1 deployable grading) and fires
  at `z > 2.0`; the ensemble is the per-step count of how many agree,
  graded at "≥2 of 4 agree."
- **Data-driven token/wake correlation** (`wakekv.signals.token_wake_stats`):
  replaces a hand-picked discourse-marker word list (rejected as
  unscientific — see PR discussion) with an unbiased scan of every
  generated token id. **Discovery** (first half of each run's steps, pooled
  across runs): every token id occurring ≥5 times is tested with a
  two-proportion z-test against the run's base "wake burst in the next 8
  steps" rate, Bonferroni-corrected across every distinct token tested.
  **Held-out evaluation** (second half, never seen during discovery):
  precision/recall of firing on the discovered set only.

### Result

| model / task | events | best attention signal (causal z, F1) | ensemble_vote (causal z, F1) | token markers (held-out P/R/F1 @ lead 32) |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct / multiturn | 73 | drift **0.31** | 0.17 | 0 markers found |
| Qwen2.5-3B-Instruct / niah | 236 | entropy_trend **0.26** | 0.18 | 0 markers found |
| DeepSeek-R1-Distill-Llama-8B / cot | 1137 | drift/online_rco **0.10** | 0.09 | **4 markers → P 0.48, R 0.89, F1 ≈ 0.62** |
| DeepSeek-R1-Distill-Qwen-1.5B / cot | 407 | drift **0.14** | 0.13 | **1 marker → P 0.43, R 0.68, F1 ≈ 0.53** |

**Ensemble vote: negative.** It never beats the single best attention signal
in any of the 4 combos (0.17<0.31, 0.18<0.26, 0.09<0.10, 0.13<0.14) — the
four signals' false positives aren't decorrelated enough for voting to help;
requiring agreement mostly just drags toward the pack's middle. Consistent
with three of them sharing the same top-k-overlap machinery.

**Token/wake correlation: real, but CoT-only.** Zero markers survive the
Bonferroni bar on multiturn or NIAH (Qwen2.5-3B, non-reasoning tasks). On
both CoT runs (DeepSeek-R1-Distill, reasoning traces) it finds markers whose
*held-out* F1 (0.53–0.62) is 4–5x every other signal in this document,
attention-based or ensembled, on the same data. Because of the discovery/
held-out split this isn't an oracle-tuned number — it's what a controller
could actually achieve online. This is the first candidate anywhere in
Phase 1 that clears a genuinely useful bar, and it does so exactly where the
original hypothesis expected it: reasoning traces, not short-context
retrieval or multi-turn chat.

Marker token ids found (8B/cot): `1396` (n=8, lift 4.6, z=5.3), `20597`
(n=5, lift 4.6, z=4.2), `220` (n=168, lift 1.6, z=4.2), `10461` (n=9, lift
3.6, z=4.1). 1.5B/cot: `220` (n=93, lift 1.5, z=3.4). **Caveat:** these are
raw token ids — the study intentionally needs no tokenizer to run, but two
of the 8B markers have small discovery counts (n=8, n=5, right at the
`min_count` floor) and none have been decoded to text yet, so what they
*are* (a discourse marker like "wait", vs. punctuation/whitespace, vs.
something else) is still unconfirmed. `token_id 220` recurring across both
CoT runs is notable but the two models use different tokenizers (Qwen vs.
Llama), so it's very unlikely to be the same literal token — decoding both
independently (`tokenizer.decode([220])` per model) is a cheap, valuable
next step before reading anything into the overlap.

One more open item: this NIAH run (236 events, 9 runs pooled) is larger than
the single 64-step shakedown run in the G0 table above — worth reconciling
whether G0's NIAH row should be updated with the fuller run set.

### Reproducing / extending

```bash
bash scripts/wakekv_analyze_all_phase1.sh          # all model/task combos
python scripts/analyze_phase1.py runs/<model>/<task>  # one combo
```

Unit tests: `tests/test_signals.py` (`test_token_wake_stats_*`,
`test_bonferroni_z_bar_*`, `test_ensemble_vote_*`, `test_event_horizon_mask_*`,
`test_signal_discovered_tokens_*`).
