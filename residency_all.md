# Combined Phase 2a residency simulation — Sun Aug 23 06:29:00 PDT 2026

<!-- ============ Qwen__Qwen2.5-3B-Instruct/multiturn ============ -->
# Residency simulation — /home/utranjan/dynamic-head-kv/runs/Qwen__Qwen2.5-3B-Instruct/multiturn

- runs: 1  |  page size: 16 tokens  |  frozen: 25% heads kept full, stable refreshed every 16 steps

Miss = wanted page not GPU-resident (reactive: a fetch stall; frozen: static classification mis-served). Memory = mean GPU pages resident across the run.

| policy | budget | miss rate | mean resident pages | peak | fetches | stall steps |
|---|---|---|---|---|---|---|
| full | - | 0.000 | 18621 | 22213 | 0 | 0 |
| frozen | 8 | 0.491 | 7918 | 9065 | 14023 | 0 |
| reactive | 8 | 0.631 | 4608 | 4608 | 1164480 | 171 |
| evict | 8 | 0.651 | 4608 | 4608 | 0 | 0 |
| frozen | 16 | 0.294 | 11039 | 12488 | 21537 | 0 |
| reactive | 16 | 0.300 | 9187 | 9216 | 553613 | 171 |
| evict | 16 | 0.369 | 9187 | 9216 | 0 | 0 |
| frozen | 32 | 0.116 | 15828 | 18611 | 22241 | 0 |
| reactive | 32 | 0.010 | 17373 | 18386 | 19093 | 106 |
| evict | 32 | 0.025 | 17373 | 18386 | 0 | 0 |
| frozen | 64 | 0.110 | 16042 | 19299 | 21687 | 0 |
| reactive | 64 | 0.000 | 18621 | 22213 | 0 | 0 |
| evict | 64 | 0.000 | 18621 | 22213 | 0 | 0 |

## Read (matched-memory Pareto)
Reactive misses <= frozen at matched memory at **4/4** frozen operating points.
  - at ~7918 pages: frozen 0.491 vs reactive 0.392 (reactive better)
  - at ~11039 pages: frozen 0.294 vs reactive 0.234 (reactive better)
  - at ~15828 pages: frozen 0.116 vs reactive 0.065 (reactive better)
  - at ~16042 pages: frozen 0.110 vs reactive 0.057 (reactive better)

A win here supports C2 (frozen classification leaves quality on the table in shifting regimes). NOTE the miss asymmetry: a reactive miss is a paid fetch STALL (quality preserved), a frozen miss is a quality GAP (page unavailable until rerank). So reactive trades memory for stalls, not for accuracy.

Caveats: reactive fetches far more than frozen (see column) — the stall COUNT here is a proxy for stall TIME, which only Phase 2b measures on real PCIe. Page granularity from logged top-k (not full KV); scout-scale models. This sim ranks policies on the memory/miss frontier; it does not predict serving throughput.

## C3 read (reversible vs. destructive demotion, same budget)
Reactive (offload) misses <= evict (destroy) at **4/4** matched budgets.
  - budget 8 (~4608 vs ~4608 pages): reactive 0.631 vs evict 0.651 (reactive better)
  - budget 16 (~9187 vs ~9187 pages): reactive 0.300 vs evict 0.369 (reactive better)
  - budget 32 (~17373 vs ~17373 pages): reactive 0.010 vs evict 0.025 (reactive better)
  - budget 64 (~18621 vs ~18621 pages): reactive 0.000 vs evict 0.000 (reactive better)

Unlike frozen, reactive and evict share the exact same LRU eviction schedule — what stays resident and when something gets pushed out is identical between them, since that's driven only by demand and the budget cap. The only thing that differs is what happens to a page AFTER eviction: reactive can get it back with one paid stall; evict never gets it back at all. A win here isolates reversibility itself as the source of the advantage, holding dynamism, budget, and eviction order fixed on both sides.

<!-- ============ Qwen__Qwen2.5-3B-Instruct/niah ============ -->
# Residency simulation — /home/utranjan/dynamic-head-kv/runs/Qwen__Qwen2.5-3B-Instruct/niah

- runs: 9  |  page size: 16 tokens  |  frozen: 25% heads kept full, stable refreshed every 16 steps

Miss = wanted page not GPU-resident (reactive: a fetch stall; frozen: static classification mis-served). Memory = mean GPU pages resident across the run.

| policy | budget | miss rate | mean resident pages | peak | fetches | stall steps |
|---|---|---|---|---|---|---|
| full | - | 0.000 | 83945 | 113094 | 0 | 0 |
| frozen | 8 | 0.403 | 34047 | 44203 | 16829 | 0 |
| reactive | 8 | 0.655 | 4608 | 4608 | 1881204 | 255 |
| evict | 8 | 0.760 | 4608 | 4608 | 0 | 0 |
| frozen | 16 | 0.276 | 37204 | 47604 | 33168 | 0 |
| reactive | 16 | 0.409 | 9209 | 9216 | 1176024 | 255 |
| evict | 16 | 0.617 | 9209 | 9216 | 0 | 0 |
| frozen | 32 | 0.197 | 42203 | 53012 | 70602 | 0 |
| reactive | 32 | 0.187 | 18337 | 18421 | 538426 | 255 |
| evict | 32 | 0.398 | 18337 | 18421 | 0 | 0 |
| frozen | 64 | 0.155 | 47221 | 59125 | 112102 | 0 |
| reactive | 64 | 0.076 | 35638 | 36606 | 220181 | 254 |
| evict | 64 | 0.215 | 35638 | 36606 | 0 | 0 |

## Read (matched-memory Pareto)
Reactive misses <= frozen at matched memory at **1/1** frozen operating points.
  - at ~34047 pages: frozen 0.403 vs reactive 0.086 (reactive better)

A win here supports C2 (frozen classification leaves quality on the table in shifting regimes). NOTE the miss asymmetry: a reactive miss is a paid fetch STALL (quality preserved), a frozen miss is a quality GAP (page unavailable until rerank). So reactive trades memory for stalls, not for accuracy.

Caveats: reactive fetches far more than frozen (see column) — the stall COUNT here is a proxy for stall TIME, which only Phase 2b measures on real PCIe. Page granularity from logged top-k (not full KV); scout-scale models. This sim ranks policies on the memory/miss frontier; it does not predict serving throughput.

## C3 read (reversible vs. destructive demotion, same budget)
Reactive (offload) misses <= evict (destroy) at **4/4** matched budgets.
  - budget 8 (~4608 vs ~4608 pages): reactive 0.655 vs evict 0.760 (reactive better)
  - budget 16 (~9209 vs ~9209 pages): reactive 0.409 vs evict 0.617 (reactive better)
  - budget 32 (~18337 vs ~18337 pages): reactive 0.187 vs evict 0.398 (reactive better)
  - budget 64 (~35638 vs ~35638 pages): reactive 0.076 vs evict 0.215 (reactive better)

Unlike frozen, reactive and evict share the exact same LRU eviction schedule — what stays resident and when something gets pushed out is identical between them, since that's driven only by demand and the budget cap. The only thing that differs is what happens to a page AFTER eviction: reactive can get it back with one paid stall; evict never gets it back at all. A win here isolates reversibility itself as the source of the advantage, holding dynamism, budget, and eviction order fixed on both sides.

<!-- ============ deepseek-ai__DeepSeek-R1-Distill-Llama-8B/cot ============ -->
# Residency simulation — /home/utranjan/dynamic-head-kv/runs/deepseek-ai__DeepSeek-R1-Distill-Llama-8B/cot

- runs: 5  |  page size: 16 tokens  |  frozen: 25% heads kept full, stable refreshed every 16 steps

Miss = wanted page not GPU-resident (reactive: a fetch stall; frozen: static classification mis-served). Memory = mean GPU pages resident across the run.

| policy | budget | miss rate | mean resident pages | peak | fetches | stall steps |
|---|---|---|---|---|---|---|
| full | - | 0.000 | 49569 | 90264 | 0 | 0 |
| frozen | 8 | 0.455 | 18388 | 28721 | 160122 | 0 |
| reactive | 8 | 0.632 | 8088 | 8192 | 19062207 | 1229 |
| evict | 8 | 0.624 | 8088 | 8192 | 0 | 0 |
| frozen | 16 | 0.257 | 23749 | 34858 | 236262 | 0 |
| reactive | 16 | 0.338 | 15298 | 16384 | 11963040 | 1128 |
| evict | 16 | 0.373 | 15298 | 16384 | 0 | 0 |
| frozen | 32 | 0.102 | 31372 | 46488 | 329616 | 0 |
| reactive | 32 | 0.090 | 26362 | 32562 | 3944975 | 895 |
| evict | 32 | 0.145 | 26362 | 32562 | 0 | 0 |
| frozen | 64 | 0.045 | 37321 | 58145 | 386939 | 0 |
| reactive | 64 | 0.014 | 39178 | 55088 | 663062 | 544 |
| evict | 64 | 0.035 | 39178 | 55088 | 0 | 0 |

## Read (matched-memory Pareto)
Reactive misses <= frozen at matched memory at **4/4** frozen operating points.
  - at ~18388 pages: frozen 0.455 vs reactive 0.269 (reactive better)
  - at ~23749 pages: frozen 0.257 vs reactive 0.149 (reactive better)
  - at ~31372 pages: frozen 0.102 vs reactive 0.060 (reactive better)
  - at ~37321 pages: frozen 0.045 vs reactive 0.025 (reactive better)

A win here supports C2 (frozen classification leaves quality on the table in shifting regimes). NOTE the miss asymmetry: a reactive miss is a paid fetch STALL (quality preserved), a frozen miss is a quality GAP (page unavailable until rerank). So reactive trades memory for stalls, not for accuracy.

Caveats: reactive fetches far more than frozen (see column) — the stall COUNT here is a proxy for stall TIME, which only Phase 2b measures on real PCIe. Page granularity from logged top-k (not full KV); scout-scale models. This sim ranks policies on the memory/miss frontier; it does not predict serving throughput.

## C3 read (reversible vs. destructive demotion, same budget)
Reactive (offload) misses <= evict (destroy) at **3/4** matched budgets.
  - budget 8 (~8088 vs ~8088 pages): reactive 0.632 vs evict 0.624 (evict better)
  - budget 16 (~15298 vs ~15298 pages): reactive 0.338 vs evict 0.373 (reactive better)
  - budget 32 (~26362 vs ~26362 pages): reactive 0.090 vs evict 0.145 (reactive better)
  - budget 64 (~39178 vs ~39178 pages): reactive 0.014 vs evict 0.035 (reactive better)

Unlike frozen, reactive and evict share the exact same LRU eviction schedule — what stays resident and when something gets pushed out is identical between them, since that's driven only by demand and the budget cap. The only thing that differs is what happens to a page AFTER eviction: reactive can get it back with one paid stall; evict never gets it back at all. A win here isolates reversibility itself as the source of the advantage, holding dynamism, budget, and eviction order fixed on both sides.

<!-- ============ deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/cot ============ -->
# Residency simulation — /home/utranjan/dynamic-head-kv/runs/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/cot

- runs: 5  |  page size: 16 tokens  |  frozen: 25% heads kept full, stable refreshed every 16 steps

Miss = wanted page not GPU-resident (reactive: a fetch stall; frozen: static classification mis-served). Memory = mean GPU pages resident across the run.

| policy | budget | miss rate | mean resident pages | peak | fetches | stall steps |
|---|---|---|---|---|---|---|
| full | - | 0.000 | 12118 | 21326 | 0 | 0 |
| frozen | 8 | 0.395 | 4980 | 7356 | 33473 | 0 |
| reactive | 8 | 0.542 | 2637 | 2688 | 3083855 | 831 |
| evict | 8 | 0.543 | 2637 | 2688 | 0 | 0 |
| frozen | 16 | 0.191 | 6627 | 9355 | 46499 | 0 |
| reactive | 16 | 0.221 | 4899 | 5376 | 1580789 | 730 |
| evict | 16 | 0.263 | 4899 | 5376 | 0 | 0 |
| frozen | 32 | 0.066 | 8459 | 12607 | 60184 | 0 |
| reactive | 32 | 0.040 | 8014 | 10548 | 413095 | 507 |
| evict | 32 | 0.074 | 8014 | 10548 | 0 | 0 |
| frozen | 64 | 0.042 | 9233 | 14390 | 65292 | 0 |
| reactive | 64 | 0.005 | 10692 | 15913 | 62493 | 255 |
| evict | 64 | 0.013 | 10692 | 15913 | 0 | 0 |

## Read (matched-memory Pareto)
Reactive misses <= frozen at matched memory at **4/4** frozen operating points.
  - at ~4980 pages: frozen 0.395 vs reactive 0.216 (reactive better)
  - at ~6627 pages: frozen 0.191 vs reactive 0.121 (reactive better)
  - at ~8459 pages: frozen 0.066 vs reactive 0.034 (reactive better)
  - at ~9233 pages: frozen 0.042 vs reactive 0.024 (reactive better)

A win here supports C2 (frozen classification leaves quality on the table in shifting regimes). NOTE the miss asymmetry: a reactive miss is a paid fetch STALL (quality preserved), a frozen miss is a quality GAP (page unavailable until rerank). So reactive trades memory for stalls, not for accuracy.

Caveats: reactive fetches far more than frozen (see column) — the stall COUNT here is a proxy for stall TIME, which only Phase 2b measures on real PCIe. Page granularity from logged top-k (not full KV); scout-scale models. This sim ranks policies on the memory/miss frontier; it does not predict serving throughput.

## C3 read (reversible vs. destructive demotion, same budget)
Reactive (offload) misses <= evict (destroy) at **4/4** matched budgets.
  - budget 8 (~2637 vs ~2637 pages): reactive 0.542 vs evict 0.543 (reactive better)
  - budget 16 (~4899 vs ~4899 pages): reactive 0.221 vs evict 0.263 (reactive better)
  - budget 32 (~8014 vs ~8014 pages): reactive 0.040 vs evict 0.074 (reactive better)
  - budget 64 (~10692 vs ~10692 pages): reactive 0.005 vs evict 0.013 (reactive better)

Unlike frozen, reactive and evict share the exact same LRU eviction schedule — what stays resident and when something gets pushed out is identical between them, since that's driven only by demand and the budget cap. The only thing that differs is what happens to a page AFTER eviction: reactive can get it back with one paid stall; evict never gets it back at all. A win here isolates reversibility itself as the source of the advantage, holding dynamism, budget, and eviction order fixed on both sides.
