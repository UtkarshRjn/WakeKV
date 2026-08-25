# Combined Phase 2a residency simulation — Tue Aug 25 11:26:50 AM PDT 2026

<!-- ============ mistralai__Mistral-7B-Instruct-v0.2/niah ============ -->
# Residency simulation — /home/intern/utranjan/RetroSpec-base/dynamic-head-kv/runs/mistralai__Mistral-7B-Instruct-v0.2/niah

- runs: 5  |  page size: 16 tokens  |  frozen: 25% heads kept full, stable refreshed every 16 steps

Miss = wanted page not GPU-resident (reactive: a fetch stall; frozen: static classification mis-served). Memory = mean GPU pages resident across the run.

| policy | budget | miss rate | mean resident pages | peak | fetches | stall steps |
|---|---|---|---|---|---|---|
| full | - | 0.000 | 83711 | 111490 | 0 | 0 |
| frozen | 8 | 0.578 | 35191 | 49905 | 7952 | 0 |
| reactive | 8 | 0.650 | 8192 | 8192 | 569539 | 35 |
| evict | 8 | 0.761 | 8192 | 8192 | 0 | 0 |
| frozen | 16 | 0.504 | 38763 | 55997 | 15623 | 0 |
| reactive | 16 | 0.443 | 16348 | 16366 | 388394 | 35 |
| evict | 16 | 0.613 | 16348 | 16366 | 0 | 0 |
| frozen | 32 | 0.441 | 45713 | 67413 | 31118 | 0 |
| reactive | 32 | 0.180 | 32170 | 32550 | 157697 | 35 |
| evict | 32 | 0.309 | 32170 | 32550 | 0 | 0 |
| frozen | 64 | 0.405 | 56297 | 85068 | 54584 | 0 |
| reactive | 64 | 0.048 | 57537 | 62743 | 42242 | 34 |
| evict | 64 | 0.091 | 57537 | 62743 | 0 | 0 |

## Read (matched-memory Pareto)
Reactive misses <= frozen at matched memory at **4/4** frozen operating points.
  - at ~35191 pages: frozen 0.578 vs reactive 0.164 (reactive better)
  - at ~38763 pages: frozen 0.504 vs reactive 0.146 (reactive better)
  - at ~45713 pages: frozen 0.441 vs reactive 0.110 (reactive better)
  - at ~56297 pages: frozen 0.405 vs reactive 0.055 (reactive better)

A win here supports C2 (frozen classification leaves quality on the table in shifting regimes). NOTE the miss asymmetry: a reactive miss is a paid fetch STALL (quality preserved), a frozen miss is a quality GAP (page unavailable until rerank). So reactive trades memory for stalls, not for accuracy.

Caveats: reactive fetches far more than frozen (see column) — the stall COUNT here is a proxy for stall TIME, which only Phase 2b measures on real PCIe. Page granularity from logged top-k (not full KV); scout-scale models. This sim ranks policies on the memory/miss frontier; it does not predict serving throughput.

## C3 read (reversible vs. destructive demotion, same budget)
Reactive (offload) misses <= evict (destroy) at **4/4** matched budgets.
  - budget 8 (~8192 vs ~8192 pages): reactive 0.650 vs evict 0.761 (reactive better)
  - budget 16 (~16348 vs ~16348 pages): reactive 0.443 vs evict 0.613 (reactive better)
  - budget 32 (~32170 vs ~32170 pages): reactive 0.180 vs evict 0.309 (reactive better)
  - budget 64 (~57537 vs ~57537 pages): reactive 0.048 vs evict 0.091 (reactive better)

Unlike frozen, reactive and evict share the exact same LRU eviction schedule — what stays resident and when something gets pushed out is identical between them, since that's driven only by demand and the budget cap. The only thing that differs is what happens to a page AFTER eviction: reactive can get it back with one paid stall; evict never gets it back at all. A win here isolates reversibility itself as the source of the advantage, holding dynamism, budget, and eviction order fixed on both sides.
