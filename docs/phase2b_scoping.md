# Phase 2b Scoping — Reactive WakeKV in FlexiCache/vLLM

*What to build, what to measure, what to explicitly skip. Read before
writing any Phase-2b code. Fork strategy per user answer 2026-07-09:
**fork FlexiCache**, not the HF fallback.*

---

## 1. What the simulator (2a) proved, and did not prove

**Proved:** reactive LRU + fetch-on-demand dominates a FlexiCache-style
frozen policy on the miss-vs-memory Pareto — ~half the miss rate at
matched memory, in both CoT and multi-turn.

**Did not prove — this is what Phase 2b is for:**
1. **Stall *time* on real PCIe** (2a counted stall *events*).
2. **Quality preservation** — the simulator assumes fetch-on-demand
   preserves quality; only running actual benchmarks confirms it.
3. **Reactive vs. FlexiCache's *real code*** (not just its policy shape).
4. **End-to-end throughput** (tokens/sec) including controller overhead.

Phase 2b's job: measure all four on a real 7–8B model with real KV cache
sizes and real PCIe transfers.

---

## 2. Fork strategy: what stays, what's replaced

**FlexiCache repo:** https://github.com/NazmulTakbir/FlexiCache
(Apache 2.0, vLLM-based, published MLSys 2026).

### Keep as-is (this is the reuse budget)
- **Their vLLM integration** — paged attention, block manager, request
  scheduler. Don't touch.
- **Per-(layer, head) block tables** with dirty tracking. This is the
  hardest infrastructure they wrote; we use it directly.
- **UVA-based CUDA transfer kernels** (low-priority stream, chunked). The
  reactive controller needs the same transport.
- **MinMax score cache decoupled from KV pages** — pages that are on CPU
  can still be scored, which is what makes reactive fetch-on-demand
  possible. This IS the reversibility mechanism.
- **Modified Triton Flash-Decoding** for per-head top-K sparse decode.
- **Their evaluation harness** — LongBench + L-Eval task runners. Reuse
  verbatim so numbers are comparable.

### Replace (the actual WakeKV change)
- **Head classification** — their offline profile → **remove entirely.**
  Every head is treated identically at start: budget B pages, empty
  resident set.
- **Rerank hook (fires every 16 steps for stable heads)** → replace with
  **per-step LRU eviction + fetch-on-demand**:
  - After each decode step, if any head's resident set exceeds B,
    evict least-recently-wanted pages to the reservoir.
  - If any head wanted a page in the reservoir, that fetch already
    happened synchronously (via the same transfer kernel) — record it
    as a stall event with its wall-clock cost.

### Strip out (not needed for reactive)
- Their **stability profiling script** (offline pass over LongBench).
  Reactive doesn't need it.
- Their **classification-transfer machinery** — the "which task's
  profile to use for another task" bookkeeping.

---

## 3. Controller interface (minimal contract)

```python
class ReactiveController:
    """
    Sits between vLLM's attention and its paged block manager.
    Called from the modified Flash-Decoding hook after every decode step.
    """
    def on_step(self, layer: int, head: int, wanted_pages: set[int]) -> StepResult:
        """
        Returns:
          - fetches: list[int] — pages loaded from CPU this step (stalls)
          - evictions: list[int] — pages moved to reservoir this step
          - stall_ms: float — measured wall-clock time in fetch kernels
        """

    def stats(self) -> dict:
        """Aggregated: total fetches, evictions, mean/p95/p99 stall ms,
        resident-page-steps (memory integral)."""
```

Everything upstream of `on_step` is FlexiCache's existing code. Everything
downstream (LRU bookkeeping, reservoir management) is our ~200-line
controller. Small surface area, small scope for bugs.

---

## 4. Milestones

Time-boxed to force honesty. If a milestone slips past its box, we
diagnose why (real block? or scope creep?) before continuing.

| M | Description | Time-box | Success criterion |
|---|---|---|---|
| **M2b-0** | Reproduce FlexiCache's published number on our hardware | 2 days | Their LongBench avg within ±0.5 of Table 4 |
| **M2b-1** | Strip classification; run their code with "everyone treated stable" at B=1024 | 2 days | Runs end-to-end, matches full-cache quality (all pages resident because 1024 covers most cases) |
| **M2b-2** | Add LRU demotion (no fetch-on-demand yet — misses become quality drops) | 2 days | Miss rate matches simulator prediction ±20% |
| **M2b-3** | Add fetch-on-demand path with measured stall time | 3 days | Fetches recover the misses; wall-clock stall matches PCIe bandwidth math |
| **M2b-4** | End-to-end on a long-CoT run (R1-Distill-8B, budget sweep) | 2 days | tokens/sec, mean/p95/p99 stall ms, blocks-actually-freed all recorded |
| **M2b-5** | Quality check on LongBench subset at matched budget | 2 days | Parity or measured-and-explained gap vs full-cache |

**Total time-box: ~13 working days.** If we're at day 10 without M2b-3,
we stop and use the HF harness as fallback (still measures real PCIe,
just not in vLLM).

---

## 5. What to measure (the Phase 2b table)

Per (regime, budget, policy ∈ {full, FlexiCache-original, reactive}):

| metric | why it matters |
|---|---|
| Mean resident pages | memory footprint |
| Peak resident pages | memory ceiling (what you need to provision) |
| Blocks actually freed | the paged-attention-integration truth check |
| Fetches / decode step | reactive workload |
| Mean / p95 / p99 stall ms | the honest cost of reactive |
| Tokens/sec | end-to-end throughput including controller |
| LongBench score (subset) | quality preserved? |

Regimes: **long CoT** (R1-Distill-8B on MATH-500 slice) and **multi-turn**
(SCBench subset, Qwen2.5-7B). One "sanity" regime: **NIAH at 16K–32K**
where frozen should do fine (we should show reactive doesn't regress).

---

## 6. Hardware ask

**Need:** ≥24 GB CUDA card, PCIe Gen4 preferred (not Gen3), ≥64 GB host RAM
pinned. FlexiCache's own paper ran on H100 + PCIe 5.0 + 180 GB pinned; we
don't need that headroom.

**DSMLP `-v` types to check** (the answer from user question 2):
- A5000 (24 GB) — fine for 7–8B in fp16 with a reasonable reservoir.
- A10 (24 GB) — same class.
- A100 40/80 GB — ideal but probably queued.

**First thing on the new pod:** `nvidia-smi topo -m` + a memcpy microbenchmark
to measure actual PCIe bandwidth. FlexiCache reports 21 GB/s on H100 Gen4;
DSMLP's cards may be Gen3 (~10–13 GB/s effective). Update the transfer-bar
math in `wakekv/signals.py` with the measured number.

**Fallback if no ≥24 GB card is available:** UCSD Research Cluster (Blink
process, ~few days), then cloud rental as last resort.

---

## 7. Explicit non-goals for Phase 2b

Not in this phase (keeps scope tight):
- Multi-GPU / tensor parallel.
- Batched serving > batch 1. Phase 2b is single-request measurement.
  (FlexiCache's amortization argument for reload latency needs batching;
  we'll note this is future work.)
- Custom CUDA kernels beyond what FlexiCache provides.
- Comparison against ReasonAlloc (their code isn't released; the
  simulator's reversible-vs-evict is a fine substitute for the paper).
- HeteroCache comparison (they released code recently — worth a
  follow-up compare, but not blocking).

---

## 8. Risks specific to Phase 2b

| # | Risk | Mitigation |
|---|---|---|
| R2b-1 | FlexiCache codebase is fragile on our hardware | M2b-0 (repro) is the fail-fast check; fall back to HF harness if their code breaks on us |
| R2b-2 | Reactive per-step overhead eats the throughput win | Measure at M2b-4; if bad, batch the eviction check (every 4 steps instead of every step) |
| R2b-3 | Fetch stalls dominate the token budget | Measure stall ms; if the fraction is too high, try async prefetch of top-N most-recently-wanted-but-evicted pages |
| R2b-4 | Quality regressions in fetch-on-demand | Shouldn't happen (fetch = quality preserved by construction), but M2b-5 confirms empirically |
| R2b-5 | Scoop while we build | **Preprint the Phase 0–2a paper first** (per user answer 1) |
| R2b-6 | DSMLP big cards unavailable | Fall back per §6 |

---

## 9. Where the results go

- **Phase 2b results:** appended to `notes/phase01_results.md` under a new
  "Phase 2b — real system" heading, mirroring the existing structure.
- **Figures:** `docs/figures/phase2b_*.png`.
- **Post-2b paper:** a *follow-up* to the Phase 0–2a preprint, targeting a
  main conference (MLSys 2027 / EuroSys 2027) once the systems numbers hold.

The Phase 0–2a preprint (per §5 in `preprint_outline.md`) states this
follow-up explicitly as future work.
