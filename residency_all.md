# Combined Phase 2a residency simulation — Mon Aug 24 11:17:54 PDT 2026

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
| snapkv | 8 | 0.428 | 7938 | 10926 | 0 | 0 |
| rkv_uniform | 8 | 0.163 | 14479 | 20409 | 0 | 0 |
| reasonalloc | 8 | 0.163 | 14479 | 20409 | 0 | 0 |
| frozen | 16 | 0.294 | 11039 | 12488 | 21537 | 0 |
| reactive | 16 | 0.300 | 9187 | 9216 | 553613 | 171 |
| evict | 16 | 0.369 | 9187 | 9216 | 0 | 0 |
| snapkv | 16 | 0.247 | 11519 | 14507 | 0 | 0 |
| rkv_uniform | 16 | 0.093 | 15684 | 20409 | 0 | 0 |
| reasonalloc | 16 | 0.090 | 15684 | 20409 | 0 | 0 |
| frozen | 32 | 0.116 | 15828 | 18611 | 22241 | 0 |
| reactive | 32 | 0.010 | 17373 | 18386 | 19093 | 106 |
| evict | 32 | 0.025 | 17373 | 18386 | 0 | 0 |
| snapkv | 32 | 0.201 | 12475 | 15463 | 0 | 0 |
| rkv_uniform | 32 | 0.010 | 18072 | 20409 | 0 | 0 |
| reasonalloc | 32 | 0.009 | 18096 | 20409 | 0 | 0 |
| frozen | 64 | 0.110 | 16042 | 19299 | 21687 | 0 |
| reactive | 64 | 0.000 | 18621 | 22213 | 0 | 0 |
| evict | 64 | 0.000 | 18621 | 22213 | 0 | 0 |
| snapkv | 64 | 0.201 | 12475 | 15463 | 0 | 0 |
| rkv_uniform | 64 | 0.000 | 18621 | 22213 | 0 | 0 |
| reasonalloc | 64 | 0.000 | 18621 | 22213 | 0 | 0 |

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

## Reactive vs. SnapKV (frozen prefill selection), same budget
Reactive misses <= snapkv at **2/4** matched budgets.
  - budget 8 (~4608 vs ~7938 pages): reactive 0.631 vs snapkv 0.428 (snapkv better)
  - budget 16 (~9187 vs ~11519 pages): reactive 0.300 vs snapkv 0.247 (snapkv better)
  - budget 32 (~17373 vs ~12475 pages): reactive 0.010 vs snapkv 0.201 (reactive better)
  - budget 64 (~18621 vs ~12475 pages): reactive 0.000 vs snapkv 0.201 (reactive better)

## Reactive vs. uniform R-KV (importance-ranked destructive pruning), same budget
Reactive misses <= rkv_uniform at **1/4** matched budgets.
  - budget 8 (~4608 vs ~14479 pages): reactive 0.631 vs rkv_uniform 0.163 (rkv_uniform better)
  - budget 16 (~9187 vs ~15684 pages): reactive 0.300 vs rkv_uniform 0.093 (rkv_uniform better)
  - budget 32 (~17373 vs ~18072 pages): reactive 0.010 vs rkv_uniform 0.010 (rkv_uniform better)
  - budget 64 (~18621 vs ~18621 pages): reactive 0.000 vs rkv_uniform 0.000 (reactive better)

## Reactive vs. ReasonAlloc (per-head reallocation, destructive), same budget
Reactive misses <= reasonalloc at **1/4** matched budgets.
  - budget 8 (~4608 vs ~14479 pages): reactive 0.631 vs reasonalloc 0.163 (reasonalloc better)
  - budget 16 (~9187 vs ~15684 pages): reactive 0.300 vs reasonalloc 0.090 (reasonalloc better)
  - budget 32 (~17373 vs ~18096 pages): reactive 0.010 vs reasonalloc 0.009 (reasonalloc better)
  - budget 64 (~18621 vs ~18621 pages): reactive 0.000 vs reasonalloc 0.000 (reactive better)

Caveats specific to these three (see `wakekv/residency.py`'s docstrings for exact simplifications and confidence levels): "same budget" is a NOMINAL target for rkv_uniform/reasonalloc, not continuous capping like reactive/evict -- they only prune at periodic buffer/delta boundaries (default every 128 steps), so realized mean resident pages (shown above) can run well above the budget between prunings; compare the printed page counts, not just the budget label. SnapKV's observation window is degraded to a single query (this log's step 0), not the paper's multi-token window; R-KV/ReasonAlloc's importance-vs-redundancy joint score is importance-only, since raw key vectors (needed for redundancy) aren't logged; ReasonAlloc's offline per-layer budget calibration is substituted with an equal split, not the paper's Reasoning-Wave allocation. Each is a real, cited reimplementation with a declared gap, not a guess.

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
| snapkv | 8 | 0.499 | 9040 | 13607 | 0 | 0 |
| rkv_uniform | 8 | 0.250 | 42407 | 86445 | 0 | 0 |
| reasonalloc | 8 | 0.250 | 42407 | 86445 | 0 | 0 |
| frozen | 16 | 0.276 | 37204 | 47604 | 33168 | 0 |
| reactive | 16 | 0.409 | 9209 | 9216 | 1176024 | 255 |
| evict | 16 | 0.617 | 9209 | 9216 | 0 | 0 |
| snapkv | 16 | 0.429 | 13506 | 18073 | 0 | 0 |
| rkv_uniform | 16 | 0.179 | 44727 | 86445 | 0 | 0 |
| reasonalloc | 16 | 0.177 | 44729 | 86445 | 0 | 0 |
| frozen | 32 | 0.197 | 42203 | 53012 | 70602 | 0 |
| reactive | 32 | 0.187 | 18337 | 18421 | 538426 | 255 |
| evict | 32 | 0.398 | 18337 | 18421 | 0 | 0 |
| snapkv | 32 | 0.374 | 18995 | 23562 | 0 | 0 |
| rkv_uniform | 32 | 0.128 | 49354 | 86445 | 0 | 0 |
| reasonalloc | 32 | 0.126 | 49373 | 86445 | 0 | 0 |
| frozen | 64 | 0.155 | 47221 | 59125 | 112102 | 0 |
| reactive | 64 | 0.076 | 35638 | 36606 | 220181 | 254 |
| evict | 64 | 0.215 | 35638 | 36606 | 0 | 0 |
| snapkv | 64 | 0.368 | 19611 | 24178 | 0 | 0 |
| rkv_uniform | 64 | 0.080 | 58374 | 86445 | 0 | 0 |
| reasonalloc | 64 | 0.074 | 58659 | 86445 | 0 | 0 |

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

## Reactive vs. SnapKV (frozen prefill selection), same budget
Reactive misses <= snapkv at **3/4** matched budgets.
  - budget 8 (~4608 vs ~9040 pages): reactive 0.655 vs snapkv 0.499 (snapkv better)
  - budget 16 (~9209 vs ~13506 pages): reactive 0.409 vs snapkv 0.429 (reactive better)
  - budget 32 (~18337 vs ~18995 pages): reactive 0.187 vs snapkv 0.374 (reactive better)
  - budget 64 (~35638 vs ~19611 pages): reactive 0.076 vs snapkv 0.368 (reactive better)

## Reactive vs. uniform R-KV (importance-ranked destructive pruning), same budget
Reactive misses <= rkv_uniform at **1/4** matched budgets.
  - budget 8 (~4608 vs ~42407 pages): reactive 0.655 vs rkv_uniform 0.250 (rkv_uniform better)
  - budget 16 (~9209 vs ~44727 pages): reactive 0.409 vs rkv_uniform 0.179 (rkv_uniform better)
  - budget 32 (~18337 vs ~49354 pages): reactive 0.187 vs rkv_uniform 0.128 (rkv_uniform better)
  - budget 64 (~35638 vs ~58374 pages): reactive 0.076 vs rkv_uniform 0.080 (reactive better)

## Reactive vs. ReasonAlloc (per-head reallocation, destructive), same budget
Reactive misses <= reasonalloc at **0/4** matched budgets.
  - budget 8 (~4608 vs ~42407 pages): reactive 0.655 vs reasonalloc 0.250 (reasonalloc better)
  - budget 16 (~9209 vs ~44729 pages): reactive 0.409 vs reasonalloc 0.177 (reasonalloc better)
  - budget 32 (~18337 vs ~49373 pages): reactive 0.187 vs reasonalloc 0.126 (reasonalloc better)
  - budget 64 (~35638 vs ~58659 pages): reactive 0.076 vs reasonalloc 0.074 (reasonalloc better)

Caveats specific to these three (see `wakekv/residency.py`'s docstrings for exact simplifications and confidence levels): "same budget" is a NOMINAL target for rkv_uniform/reasonalloc, not continuous capping like reactive/evict -- they only prune at periodic buffer/delta boundaries (default every 128 steps), so realized mean resident pages (shown above) can run well above the budget between prunings; compare the printed page counts, not just the budget label. SnapKV's observation window is degraded to a single query (this log's step 0), not the paper's multi-token window; R-KV/ReasonAlloc's importance-vs-redundancy joint score is importance-only, since raw key vectors (needed for redundancy) aren't logged; ReasonAlloc's offline per-layer budget calibration is substituted with an equal split, not the paper's Reasoning-Wave allocation. Each is a real, cited reimplementation with a declared gap, not a guess.

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
| snapkv | 8 | 0.021 | 46512 | 87201 | 0 | 0 |
| rkv_uniform | 8 | 0.508 | 11764 | 19442 | 0 | 0 |
| reasonalloc | 8 | 0.514 | 11764 | 19442 | 0 | 0 |
| frozen | 16 | 0.257 | 23749 | 34858 | 236262 | 0 |
| reactive | 16 | 0.338 | 15298 | 16384 | 11963040 | 1128 |
| evict | 16 | 0.373 | 15298 | 16384 | 0 | 0 |
| snapkv | 16 | 0.007 | 48212 | 88900 | 0 | 0 |
| rkv_uniform | 16 | 0.314 | 18353 | 25805 | 0 | 0 |
| reasonalloc | 16 | 0.317 | 18353 | 25805 | 0 | 0 |
| frozen | 32 | 0.102 | 31372 | 46488 | 329616 | 0 |
| reactive | 32 | 0.090 | 26362 | 32562 | 3944975 | 895 |
| evict | 32 | 0.145 | 26362 | 32562 | 0 | 0 |
| snapkv | 32 | 0.005 | 48533 | 89221 | 0 | 0 |
| rkv_uniform | 32 | 0.132 | 28381 | 39144 | 0 | 0 |
| reasonalloc | 32 | 0.129 | 28381 | 39145 | 0 | 0 |
| frozen | 64 | 0.045 | 37321 | 58145 | 386939 | 0 |
| reactive | 64 | 0.014 | 39178 | 55088 | 663062 | 544 |
| evict | 64 | 0.035 | 39178 | 55088 | 0 | 0 |
| snapkv | 64 | 0.005 | 48533 | 89221 | 0 | 0 |
| rkv_uniform | 64 | 0.031 | 40267 | 59934 | 0 | 0 |
| reasonalloc | 64 | 0.027 | 40267 | 59934 | 0 | 0 |

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

## Reactive vs. SnapKV (frozen prefill selection), same budget
Reactive misses <= snapkv at **0/4** matched budgets.
  - budget 8 (~8088 vs ~46512 pages): reactive 0.632 vs snapkv 0.021 (snapkv better)
  - budget 16 (~15298 vs ~48212 pages): reactive 0.338 vs snapkv 0.007 (snapkv better)
  - budget 32 (~26362 vs ~48533 pages): reactive 0.090 vs snapkv 0.005 (snapkv better)
  - budget 64 (~39178 vs ~48533 pages): reactive 0.014 vs snapkv 0.005 (snapkv better)

## Reactive vs. uniform R-KV (importance-ranked destructive pruning), same budget
Reactive misses <= rkv_uniform at **2/4** matched budgets.
  - budget 8 (~8088 vs ~11764 pages): reactive 0.632 vs rkv_uniform 0.508 (rkv_uniform better)
  - budget 16 (~15298 vs ~18353 pages): reactive 0.338 vs rkv_uniform 0.314 (rkv_uniform better)
  - budget 32 (~26362 vs ~28381 pages): reactive 0.090 vs rkv_uniform 0.132 (reactive better)
  - budget 64 (~39178 vs ~40267 pages): reactive 0.014 vs rkv_uniform 0.031 (reactive better)

## Reactive vs. ReasonAlloc (per-head reallocation, destructive), same budget
Reactive misses <= reasonalloc at **2/4** matched budgets.
  - budget 8 (~8088 vs ~11764 pages): reactive 0.632 vs reasonalloc 0.514 (reasonalloc better)
  - budget 16 (~15298 vs ~18353 pages): reactive 0.338 vs reasonalloc 0.317 (reasonalloc better)
  - budget 32 (~26362 vs ~28381 pages): reactive 0.090 vs reasonalloc 0.129 (reactive better)
  - budget 64 (~39178 vs ~40267 pages): reactive 0.014 vs reasonalloc 0.027 (reactive better)

Caveats specific to these three (see `wakekv/residency.py`'s docstrings for exact simplifications and confidence levels): "same budget" is a NOMINAL target for rkv_uniform/reasonalloc, not continuous capping like reactive/evict -- they only prune at periodic buffer/delta boundaries (default every 128 steps), so realized mean resident pages (shown above) can run well above the budget between prunings; compare the printed page counts, not just the budget label. SnapKV's observation window is degraded to a single query (this log's step 0), not the paper's multi-token window; R-KV/ReasonAlloc's importance-vs-redundancy joint score is importance-only, since raw key vectors (needed for redundancy) aren't logged; ReasonAlloc's offline per-layer budget calibration is substituted with an equal split, not the paper's Reasoning-Wave allocation. Each is a real, cited reimplementation with a declared gap, not a guess.

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
| snapkv | 8 | 0.032 | 11065 | 20264 | 0 | 0 |
| rkv_uniform | 8 | 0.399 | 3911 | 6423 | 0 | 0 |
| reasonalloc | 8 | 0.401 | 3911 | 6423 | 0 | 0 |
| frozen | 16 | 0.191 | 6627 | 9355 | 46499 | 0 |
| reactive | 16 | 0.221 | 4899 | 5376 | 1580789 | 730 |
| evict | 16 | 0.263 | 4899 | 5376 | 0 | 0 |
| snapkv | 16 | 0.014 | 11577 | 20776 | 0 | 0 |
| rkv_uniform | 16 | 0.201 | 5872 | 8507 | 0 | 0 |
| reasonalloc | 16 | 0.199 | 5872 | 8507 | 0 | 0 |
| frozen | 32 | 0.066 | 8459 | 12607 | 60184 | 0 |
| reactive | 32 | 0.040 | 8014 | 10548 | 413095 | 507 |
| evict | 32 | 0.074 | 8014 | 10548 | 0 | 0 |
| snapkv | 32 | 0.012 | 11657 | 20856 | 0 | 0 |
| rkv_uniform | 32 | 0.063 | 8495 | 12293 | 0 | 0 |
| reasonalloc | 32 | 0.058 | 8495 | 12295 | 0 | 0 |
| frozen | 64 | 0.042 | 9233 | 14390 | 65292 | 0 |
| reactive | 64 | 0.005 | 10692 | 15913 | 62493 | 255 |
| evict | 64 | 0.013 | 10692 | 15913 | 0 | 0 |
| snapkv | 64 | 0.012 | 11657 | 20856 | 0 | 0 |
| rkv_uniform | 64 | 0.012 | 10864 | 16908 | 0 | 0 |
| reasonalloc | 64 | 0.009 | 10864 | 16908 | 0 | 0 |

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

## Reactive vs. SnapKV (frozen prefill selection), same budget
Reactive misses <= snapkv at **1/4** matched budgets.
  - budget 8 (~2637 vs ~11065 pages): reactive 0.542 vs snapkv 0.032 (snapkv better)
  - budget 16 (~4899 vs ~11577 pages): reactive 0.221 vs snapkv 0.014 (snapkv better)
  - budget 32 (~8014 vs ~11657 pages): reactive 0.040 vs snapkv 0.012 (snapkv better)
  - budget 64 (~10692 vs ~11657 pages): reactive 0.005 vs snapkv 0.012 (reactive better)

## Reactive vs. uniform R-KV (importance-ranked destructive pruning), same budget
Reactive misses <= rkv_uniform at **2/4** matched budgets.
  - budget 8 (~2637 vs ~3911 pages): reactive 0.542 vs rkv_uniform 0.399 (rkv_uniform better)
  - budget 16 (~4899 vs ~5872 pages): reactive 0.221 vs rkv_uniform 0.201 (rkv_uniform better)
  - budget 32 (~8014 vs ~8495 pages): reactive 0.040 vs rkv_uniform 0.063 (reactive better)
  - budget 64 (~10692 vs ~10864 pages): reactive 0.005 vs rkv_uniform 0.012 (reactive better)

## Reactive vs. ReasonAlloc (per-head reallocation, destructive), same budget
Reactive misses <= reasonalloc at **2/4** matched budgets.
  - budget 8 (~2637 vs ~3911 pages): reactive 0.542 vs reasonalloc 0.401 (reasonalloc better)
  - budget 16 (~4899 vs ~5872 pages): reactive 0.221 vs reasonalloc 0.199 (reasonalloc better)
  - budget 32 (~8014 vs ~8495 pages): reactive 0.040 vs reasonalloc 0.058 (reactive better)
  - budget 64 (~10692 vs ~10864 pages): reactive 0.005 vs reasonalloc 0.009 (reactive better)

Caveats specific to these three (see `wakekv/residency.py`'s docstrings for exact simplifications and confidence levels): "same budget" is a NOMINAL target for rkv_uniform/reasonalloc, not continuous capping like reactive/evict -- they only prune at periodic buffer/delta boundaries (default every 128 steps), so realized mean resident pages (shown above) can run well above the budget between prunings; compare the printed page counts, not just the budget label. SnapKV's observation window is degraded to a single query (this log's step 0), not the paper's multi-token window; R-KV/ReasonAlloc's importance-vs-redundancy joint score is importance-only, since raw key vectors (needed for redundancy) aren't logged; ReasonAlloc's offline per-layer budget calibration is substituted with an equal split, not the paper's Reasoning-Wave allocation. Each is a real, cited reimplementation with a declared gap, not a guess.
