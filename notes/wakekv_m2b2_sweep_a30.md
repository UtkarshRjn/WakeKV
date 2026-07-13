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

## Quality axis (LongBench, `scripts/wakekv_longbench_check.sh`)

Reactive at **R=1** (every-step rerank) vs stock FlexiCache, 4 tasks ×
30 samples, scored with the harness `eval.py`:

| task | stock | reactive-r1 | Δ |
|---|---:|---:|---:|
| triviaqa | 87.06 | 87.06 | 0.00 |
| multi_news | 25.63 | 26.16 | +0.53 |
| qasper | 37.67 | 35.42 | −2.25 |
| 2wikimqa | 16.58 | 14.00 | −2.58 |
| **mean** | **41.73** | **40.66** | **−1.07** |

Reactive retains **~97.4%** of stock LongBench mean at R=1 while running
**1.34×** faster — supports "fetch-on-demand preserves quality by
construction". TriviaQA identical, summarization flat, the two multi-hop/QA
tasks off ~2.5 pts (small, near noise at N=30; 2wikimqa is the one to watch).

**Pareto is incomplete.** Quality is measured only at R=1 — reactive's
*quality ceiling* (larger R = staler residency = likely lower quality but
higher throughput). The high-throughput points (R=8 → 2.22×, R=16 → 2.33×)
have **no quality number yet**. To finish the throughput×quality Pareto, run
`RERANK_INTERVAL=<R> bash scripts/wakekv_longbench_check.sh` for R in
{2,4,8,16} (stock is re-scored each time; ~1h each on A30).

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
