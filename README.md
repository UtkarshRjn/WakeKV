# dynamic-head-kv (private)

Research repo for **WakeKV** (working title): reversible, decode-time
head-priority management for the LLM KV cache — demote cooling attention
heads by offloading their KV to CPU, detect heads waking up, promote them
back. Targeting a workshop paper (NeurIPS 2026 workshops primary, ICLR
2027 workshops backup).

- [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) — claims, phases, decision gates,
  risks, compute budget.
- [`notes/`](notes/) — literature deep-reads and ideation trail.

**Private:** contains unpublished research strategy. Do not make public
before the preprint is out.
