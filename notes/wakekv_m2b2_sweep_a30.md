# M2b-2 — WakeKV reactive rerank-interval sweep (A30, Mistral-7B-v0.2)

The Phase 2b headline: WakeKV reactive residency (every head sparse, none
kept fully resident) vs stock FlexiCache (64 unstable heads kept dense),
swept over rerank interval. Run through `scripts/wakekv_rerank_sweep.sh` on
the A30, output-len 1000 / input 8k / 24 prompts (the regime where stock
FlexiCache showed its biggest speedup over vanilla vLLM).

## Throughput

| config | rerank interval | gen tok/s | elapsed (s) | vs stock |
|---|---:|---:|---:|---:|
| stock (FlexiCache) | 16 (native) | 114.4 | 209.8 | 1.00× (baseline) |
| reactive | 1 | 153.6 | 156.2 | 1.34× |
| reactive | 2 | 193.3 | 124.1 | 1.69× |
| reactive | 4 | 226.3 | 106.1 | 1.98× |
| reactive | 8 | 254.2 | 94.4 | 2.22× |
| reactive | 16 | 266.6 | 90.0 | **2.33×** |

## Read

- **Reactive beats stock FlexiCache at every interval**, 1.34×→2.33× as
  reranking gets less frequent. Two compounding effects: (1) reactive keeps
  **0 heads dense**, so its GPU KV is distributed uniformly across layers
  (`[11834]×32`) instead of stock's lumpy `[4096, 20707, …]` where
  dense-unstable-head layers starve others — the uniform layout admits a
  larger effective decode batch; (2) larger rerank interval = fewer
  promote/demote passes = less per-step overhead.
- The curve is monotone in R: overhead falls as R grows, so throughput rises
  and asymptotes toward the no-rerank ceiling. R=16 is the Pareto knee here.

## Quality axis (LongBench) — full sweep

Stock FlexiCache vs reactive at every rerank interval, 4 tasks × 30 samples,
scored with the harness `eval.py` (via `scripts/wakekv_quality_pareto.sh`,
which scores each interval in an isolated pred dir so reactive-r16 doesn't
collide with stock's rerank-16 predictions):

| rerank int | gen tok/s | vs stock | LongBench mean | % of stock |
|---:|---:|---:|---:|---:|
| stock (16) | 114.4 | 1.00× | 41.73 | — |
| 1 | 153.6 | 1.34× | 40.66 | 97.4% |
| 2 | 193.3 | 1.69× | 40.54 | 97.1% |
| 4 | 226.3 | 1.98× | 41.36 | 99.1% |
| 8 | 254.2 | 2.22× | 40.87 | 97.9% |
| 16 | 266.6 | 2.33× | 39.63 | 95.0% |

Per-task:

| config | multi_news | qasper | 2wikimqa | triviaqa |
|---|---:|---:|---:|---:|
| stock | 25.63 | 37.67 | 16.58 | 87.06 |
| reactive-r1 | 26.16 | 35.42 | 14.00 | 87.06 |
| reactive-r2 | 26.50 | 33.32 | 15.27 | 87.06 |
| reactive-r4 | 26.18 | 36.04 | 16.16 | 87.06 |
| reactive-r8 | 26.12 | 33.47 | 16.81 | 87.06 |
| reactive-r16 | 25.82 | 32.30 | 13.33 | 87.06 |

**Read.** Quality is essentially **flat at 95–99% of stock across the whole
sweep — no monotonic erosion.** Even the fastest point (R=16, 2.33×) holds
95%. TriviaQA is identical everywhere; the spread is dominated by the two
small-N QA tasks (qasper/2wikimqa bounce ±2–4 pts, noise at N=30). So reactive
residency buys **up to 2.33× throughput at ≥95% LongBench quality** — the
throughput×quality tradeoff M2b-2 set out to measure.

Caveat: N=30/task makes per-task deltas noisy (see the non-monotone qasper
column); the *mean* is stable but a full-LongBench (~200/task) run would tighten
the per-task numbers before any strong per-task claim.

Other caveats: single output length (1000), 24 prompts, A30 24GB with
`max-model-len 10240` and a 16GB host KV pool — ranks policies on this box,
does not predict absolute serving numbers. Small N → some noise.

## Reproduce

```bash
bash scripts/wakekv_rerank_sweep.sh          # idempotent; skips completed runs
python scripts/wakekv_sweep_table.py \
    /home/utranjan/FlexiCache/benchmarks/FlexiCache/Throughput/Results_M2b2
```

Raw JSONs: `FlexiCache/benchmarks/FlexiCache/Throughput/Results_M2b2/`.
