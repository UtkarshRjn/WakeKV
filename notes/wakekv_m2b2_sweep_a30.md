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

## Important caveat — this is throughput only

The benchmark measures tokens/s, **not** output quality. Reactive makes
*every* head sparse (top-64 pages), where stock keeps 64 heads fully dense.
Some of the speedup is reactive simply doing less attention work. Whether
reactive retains accuracy at these intervals is a **separate axis not tested
here** — it needs the LongBench-style eval (see
`notes/flexicache_benchmark_a30.md`) run under `--mode reactive` across the
interval sweep. Expect a quality/throughput tradeoff: high R = fast but
staler residency. The real Pareto is throughput × accuracy; this file is one
axis of it.

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
