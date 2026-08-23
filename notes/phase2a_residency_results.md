# Phase 2a — Reactive-Residency Simulation Results (record of numbers)

*The durable numeric record for `scripts/simulate_residency.py`
(`wakekv/residency.py`). Raw per-run `residency.md` output lives on the
cluster / in [PR #19](https://github.com/UtkarshRjn/dynamic-head-kv/pull/19)'s
discussion; the numbers are transcribed here so they survive.*

**Status: RUN on real Phase 0/1 logs** — `bash
scripts/wakekv_simulate_residency_all.sh`, all 4 model/task combos that
have real logs, page size 16, budgets {8, 16, 32, 64}.

No GPU: this replays *logged* attention (top-k page sets already recorded
by Phase 0) through three residency policies in pure NumPy — it doesn't
predict serving throughput or measure real PCIe stall time (that's Phase 2b/3).

## What's being compared

- **frozen** — FlexiCache-style: heads classified stable/unstable once,
  offline; unstable heads stay fully resident, stable heads keep a fixed
  top-B set refreshed every 16 steps.
- **reactive** — WakeKV: every head capped at B resident pages by recent
  demand (LRU); a wanted page on CPU is fetched on demand (one stall) and
  the page it evicts goes to a CPU reservoir, recoverable later.
- **evict** — ReasonAlloc-style: identical LRU cap and eviction rule to
  reactive, same budget, but demotion is destructive — an evicted page is
  gone for good, every later want of it misses again, forever.

## C2 — reactive vs. frozen, at matched memory

| model / task | frozen operating points compared | reactive wins |
|---|---|---|
| Qwen2.5-3B-Instruct / multiturn | 4 | **4/4** |
| Qwen2.5-3B-Instruct / niah | 1 (see note) | **1/1** |
| DeepSeek-R1-Distill-Llama-8B / cot | 4 | **4/4** |
| DeepSeek-R1-Distill-Qwen-1.5B / cot | 4 | **4/4** |

**13/13 — a clean sweep.** At every matched-memory point across every
model/task combo tested, reactive's miss rate is lower than frozen's — in
several cases by 2x or more (e.g. 8B/cot at ~18k pages: frozen 0.455 vs.
reactive 0.269). NIAH only has 1 comparable point because reactive's
memory footprint stayed below all but the smallest frozen budget in that
run — not a partial result, just fewer points fall inside reactive's
interpolation range.

*Recall the miss asymmetry: a reactive miss is a paid fetch **stall**
(quality preserved); a frozen miss is a **quality gap** (page unavailable
until the next scheduled rerank). So this result is "reactive trades memory
for stalls, not for accuracy" — Phase 2b/3 is what puts a real number on
the stall cost.*

## C3 — reactive (offload) vs. evict (destroy), same budget

| model / task | budgets compared | reactive wins |
|---|---|---|
| Qwen2.5-3B-Instruct / multiturn | 4 | **4/4** |
| Qwen2.5-3B-Instruct / niah | 4 | **4/4** |
| DeepSeek-R1-Distill-Llama-8B / cot | 4 | 3/4 |
| DeepSeek-R1-Distill-Qwen-1.5B / cot | 4 | **4/4** |

**15/16.** Reversibility wins essentially everywhere the budget is tight
enough for demotion to matter at all (at the largest budgets both policies
converge to ~0 miss rate, since almost nothing needs to be evicted).

**The one exception:** 8B/cot at budget=8 (the tightest budget, on the
largest/highest-churn combo — 1137 wake events): reactive 0.632 vs. evict
0.624 — evict very slightly better (0.008 absolute, ~1.3% relative).

*Why this can happen even though reactive and evict share the same LRU
rule:* the eviction **schedule** is only *provably* identical between the
two up to the first "recovery" event (a re-want of a page reactive brought
back but evict left destroyed). After that, reactive's resident set
literally contains a page evict's doesn't — so the two policies' "oldest N"
sets can diverge, and a *later* eviction round can legitimately choose a
different page under each policy. At a very tight budget on a
high-churn trace, recovery events are frequent, so this composition drift
has the most room to compound — which is exactly the one place it shows
up. This isn't a bug; it's a real, understood second-order effect of what
"same schedule" actually means once a page comes back from the dead on one
side and not the other. It doesn't undermine the 15/16 result — it's the
expected shape of noise at the strictest operating point.

## Read

**C2 and C3 both hold on real data**, not just the earlier synthetic
verification. This is the first real-log confirmation for C3 specifically
— it previously only had a synthetic-data check
(`tests/test_residency.py::test_evict_*`). Together with Phase 0/1
(`notes/phase01_results.md`), the simulator-level case for WakeKV's two
central design choices — reactive dynamism (C2) and reversible demotion
(C3) — is now supported across 4 real model/task combos (2 model families,
3B–8B, CoT/NIAH/multi-turn).

**Caveats (same as noted in `wakekv/residency.py` and RESEARCH_PLAN.md):**
counts *misses*, not stall *time*; page granularity approximated from
logged top-k, not full KV; compares against a FlexiCache-*style* policy,
not their actual code; scout-scale models (1.5B–8B). The real-system,
matched-stall-time head-to-head (Phase 2b/3) is still what the paper's
headline numbers depend on — this is the gate that says it's worth running
that experiment, not a replacement for it.

### Reproducing / extending

```bash
bash scripts/wakekv_simulate_residency_all.sh          # all model/task combos
python scripts/simulate_residency.py runs/<model>/<task>  # one combo
```
