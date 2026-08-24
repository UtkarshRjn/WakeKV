# E3c design doc — making FlexiCache's demotion destructive ("evict" mode)

*Investigation for claim C3 / experiment E3c (`RESEARCH_PLAN.md` Phase 3):
a real-system offload-vs-evict head-to-head, analogous to the
simulator-level result already in `notes/phase2a_residency_results.md`.
This is a **design doc, not a shipped patch** — see §5 for why.*

**Source investigated:** `https://github.com/NazmulTakbir/FlexiCache`,
cloned read-only to `/tmp/flexicache_investigation` (not part of this
repo). All file:line citations below are against that clone (vLLM 0.8.2
fork), current as of the commit fetched 2026-08-24.

---

## 1. What I was looking for

`wakekv/flexicache_shim.py`'s `"reactive"` mode gets reversibility "for
free" by not touching FlexiCache's promote/demote mechanism at all — it
just forces every head through FlexiCache's *own* top-B + reservoir path
(`unstable_heads = []`) at full cadence (`rerank_frequency` overridden).
The task was to find the analogous free lunch for **evict**: some single,
well-scoped patch point where flipping FlexiCache's *existing* reversible
mechanism into a destructive one is as clean as the two-field config
override the reactive mode already does.

I did not find one. The reversibility isn't a flag anywhere — it's an
emergent property of three cooperating, always-on data structures, and
disabling it correctly means adding new persistent state that has to be
threaded through vLLM's continuous-batching request lifecycle (row
add/remove/move/condense) in `gpu_input_batch.py`. That's real engineering
on live GPU-resident tensors, not a two-field override. Details below.

---

## 2. How FlexiCache's reversibility actually works

Three cooperating structures, none of which is a "reservoir set" you can
just swap an eviction policy into:

### 2a. GPU KV blocks — the only thing `free_pages_decode_phase` touches

`KVCacheManager.free_pages_decode_phase`
(`vllm/v1/core/kv_cache_manager.py:624-672`) is called once per decode
step. For every **stable** head (i.e. every head under WakeKV-reactive,
since the shim forces `unstable_heads = []`), it computes the logical
block range that just fell outside the rerank window
(`start_log = cdiv(seq_len - rerank_freq, B)`, `end_log = cdiv(seq_len, B) - 1`,
lines 636-638), replaces that request's block-table entries at those
logical positions with a **null "guard" block**
(`guard = block_pools[L].free_block_queue.null_block`, line 650), and
returns the real block objects to `block_pools[L].free_blocks(..., guard_check=True)`
(line 672) — i.e. `BlockPool.free_blocks`
(`vllm/v1/core/block_pool.py:242-258`), which just appends them back to
the **shared GPU free-block queue** (`free_block_queue.append_many`,
line 253). That queue has zero memory of which logical page a block used
to hold — it's a generic pool, reused for whatever's allocated next,
for any request, any head. There is no "this page can come back" bit
here at all: this step *only* decides "not resident right now."

### 2b. The CPU mirror — the actual reservoir, and it's unconditional

The "can this page come back" property lives entirely in a **permanent,
continuously-maintained CPU copy** of the full KV cache, decoupled from
GPU residency:

- At `KVCacheManager` construction (`enable_flexicache=True` branch,
  lines 67-101), a full second block-pool hierarchy is allocated per
  layer: `req_to_cpu_blocks_by_layer` (line 86-88), populated via
  `BlockPool.get_new_blocks_cpu` (`block_pool.py:160-161`) whenever a
  request needs new logical block slots (`kv_cache_manager.py:349-364`).
  This CPU block table is sized to the **full** logical sequence per
  head, not to a top-B budget.
- `GPUModelRunner.offload_kv_cache_d2h_selected`
  (`vllm/v1/worker/gpu_model_runner.py:2232-2292`) runs every step for
  every request that generated new tokens (`cnt_d2h`, driven from
  `last_tx_blk_gpu`/`total_full_blk_gpu`, lines 2240-2258) and D2H-copies
  the newly-produced KV into that CPU mirror via a raw CUDA extension
  (`kv_d2h_tx_gpu_mapped.cu`, loaded at line 335). This runs
  unconditionally, independent of which heads are "stable" vs
  "unstable" and independent of the rerank cadence — every token's KV
  gets mirrored to CPU as soon as it's produced.
- The CPU block table entries are **only ever freed at request
  completion** (`kv_cache_manager.py:476`, inside the request-cleanup
  path), never during decode-phase demotion.

So "demote" (§2a) never touches the data that would need to be destroyed
to make demotion destructive. It only ever touches an ephemeral GPU
free-list. The real, freeable, "must actually be destroyed for evict
semantics" copy is the CPU mirror in `req_to_cpu_blocks_by_layer`,
written continuously and independently of the top-B/reservoir
bookkeeping.

### 2c. MinMax scores — reversibility's other half, and why "just don't refetch" isn't enough either

A block that's been demoted to CPU can be **re-selected into the top-B
set on a later rerank** even without ever being GPU-resident in the
interim, because block *scores* are maintained for the **entire logical
range**, not just the currently-resident pages, via a **separate, tiny,
always-resident-on-GPU MinMax summary cache**
(`minmax_block_pools`, `kv_cache_manager.py:77-79`; kernel at
`vllm/attention/ops/flexi_cache_triton_kernels.py:818-947`,
`kernel_compute_block_scores_minmax`). That kernel computes a score for
**every logical block index `< num_blk`** each rerank step
(loop `for p in tl.static_range(PAGES_PER_TB)` scanning `blk_base..blk_base+PAGES_PER_TB`,
line 902, gated only by `in_range = blk < num_blk`, line 904 — not by
GPU residency).

The actual "who's in the top-B now" decision is pure PyTorch (not a
kernel), in `write_top_k_blocks` / `write_top_k_blocks_post`
(`flexi_cache_triton_kernels.py:1084-1134`):

```python
# write_top_k_blocks_post, lines 1125-1134
if any_needs_rerank:
    ...
    old_top_k_blocks[:, needs_rerank_gpu, :, :] = top_k_blocks[:, needs_rerank_gpu, :, :]
    _, idx = torch.topk(bs[:, needs_rerank_gpu[:, None], :, :], K, dim=-1,
                         largest=True, sorted=False)
    top_k_blocks[:, needs_rerank_gpu[:, None], :, :K] = idx
```

`bs` (`block_scores`) covers the full logical range, so `torch.topk`
here has no memory of "this block was evicted three reranks ago" — it
just picks the current best K. `GPUModelRunner.reload_kv_cache_h2d`
(`gpu_model_runner.py:2294-2364`) then diffs `old_top_k_blocks` vs.
`top_k_blocks` (via the `topk_swap_map` CUDA kernel,
`vllm/v1/flexicache/kernels/cuda/topk_swap_map.cu`) and H2D-transfers
whatever's newly "incoming," sourcing from the CPU mirror (§2b) via
`cpu_block_table`. So even if you separately solved §2b (stopped
mirroring evicted pages to CPU, or scrubbed them after eviction), a
plain re-run of `torch.topk` over an unmodified `block_scores` array
could still select that block's *index* into `top_k_blocks` — the H2D
transfer for it would then either read stale/undefined CPU memory or
need separate handling. Correctly cutting reversibility means gating
*this* selection step, not (only) the CPU mirror.

---

## 3. The patch point I'd propose — and why it doesn't clear the "clean, well-scoped" bar

The least-bad patch point, by elimination, is `write_top_k_blocks_post`
(and `write_top_k_blocks` for the first-decode case) in
`vllm/attention/ops/flexi_cache_triton_kernels.py`. It's pure PyTorch
(no Triton/CUDA to touch), and it's the single place that turns "current
scores" into "who's resident" — architecturally the closest analog to
how the shim already treats `FlexiCacheConfig._build` as the one hook
point that determines downstream behavior.

The idea: maintain a persistent boolean "banned" tensor, same shape
class as `block_scores`
(`[num_layers, num_req_slots, num_kv_heads, max_logical_blocks]`).
Every time `write_top_k_blocks_post` runs, before computing `idx`,
mask out already-banned block positions in `bs` (set to `-inf`) so they
can never be re-selected; after computing `idx`, mark newly-dropped
blocks (in `old_top_k_blocks` but not in the new `idx`) as banned going
forward. That's a real transcription of `ReactiveController`'s
`demotion="evict"` semantics (`wakekv/reactive_controller.py:194-199`,
"currently resident but no longer desired" → reservoir/gone, and never
resident again) into the top-K selection step.

This is exactly the "make the freed slot's identity permanently
unreachable" operation item 3 of the task asked for — applied at the
score-selection layer rather than the GPU-block layer, because that's
where the identity of "this logical page" actually lives (§2c).

**Why I'm not shipping this as `mode="evict"` in `flexicache_shim.py`:**

1. **It isn't monkeypatchable the way `_build` is.** The call sites live
   in `vllm/attention/ops/chunked_prefill_paged_decode.py:753` and
   `:829`, which imports via `from .flexi_cache_triton_kernels import *`
   (`chunked_prefill_paged_decode.py:15`). That's a *value* import — it
   binds a local name in `chunked_prefill_paged_decode`'s namespace at
   import time, so monkeypatching
   `flexi_cache_triton_kernels.write_top_k_blocks_post` (the pattern the
   existing shim uses against `FlexiCacheConfig._build`) would silently
   **not** take effect. The patch would have to target
   `chunked_prefill_paged_decode.write_top_k_blocks_post` instead — a
   different module than the one the "clean" mechanism lives in,
   already a step away from "wrap one method."
2. **The banned-mask tensor needs new request-lifecycle wiring, in a
   different file, that doesn't exist yet.** `top_k_blocks`,
   `old_top_k_blocks`, and `block_scores` are all kept correct across
   vLLM's continuous batching (new request takes over a freed batch
   slot; requests reorder within the batch) by `InputBatch.add_request`,
   `.remove_request`, `.swap_states`, `.condense`
   (`vllm/v1/worker/gpu_input_batch.py:314, 450, 490, 549`, plus the
   `move_row`/copy calls around lines 572-606 and 874-951 for the
   paused-request path). A new banned-mask tensor of the same shape
   would need matching resets/copies/moves added at **every** one of
   those call sites, or a completed request's "this page is banned"
   state leaks into whatever unrelated request the freed batch slot is
   reassigned to next — a silent correctness bug (wrong request's
   heads get spuriously starved), not a crash, so it wouldn't
   necessarily surface as an obvious failure in a short smoke test.
3. **No way to check tensor shape/dtype/broadcast correctness without a
   live vLLM+GPU run.** `bs[:, needs_rerank_gpu[:, None], :, :]` and the
   corresponding masked-assignment for the banned tensor involve
   boolean-mask fancy indexing over GPU tensors with several implicit
   broadcast rules; getting this wrong doesn't raise cleanly in every
   case (silent shape broadcast, or a `topk` over an all-`-inf` slice)
   — exactly the kind of failure that's easy to introduce untested and
   hard to catch without wolverine's actual FlexiCache/vLLM install.

None of this is true of the existing `"reactive"`/`"identity"` modes:
those touch two Python scalar fields, read at one call site
(`FlexiCacheConfig._build`), with no cross-request state and no tensor
math — which is exactly why they were safe to write and test with a
mock config object (`tests/test_flexicache_shim.py`). The evict
mechanism fails that bar on all three axes at once (wrong-module
monkeypatch target, new cross-file lifecycle state, live-tensor-only
verifiability). Per the project's "never fabricate, flag what's
unverified" ethos, I'm not landing a `mode="evict"` here that I can't
actually exercise against real FlexiCache — a subtly-wrong version of
this would look identical to a correct one until someone inspects
per-head miss curves on a real run, which is precisely the risk the
task asked me to avoid.

---

## 4. What a correct implementation needs (for whoever picks this up on wolverine)

1. **New persistent tensor** `evicted_mask: torch.Tensor[bool]`, shape
   `[num_layers, max_num_reqs, num_kv_heads, max_logical_blocks]` (mirror
   `block_scores`'s allocation in `gpu_input_batch.py` — search
   `self.block_scores = torch.zeros(` around line 255 for the sibling
   pattern to copy), living in `InputBatch`.
2. **Reset on slot reuse.** Zero the relevant slice in
   `InputBatch.add_request` (line 314) whenever a batch row is claimed
   for a **new** request (not a reordering of an existing one).
3. **Copy on reorder, not reset.** In `swap_states` (line 490) and
   `condense` (line 549), and in the paused-request copy path (around
   lines 874-951), move/copy `evicted_mask` alongside `top_k_blocks` —
   same treatment, same call sites, so a request's ban history survives
   being shuffled to a different batch row.
4. **Patch `write_top_k_blocks` and `write_top_k_blocks_post`** (or add
   evict-aware siblings) to (a) mask `bs` at already-banned positions to
   `-inf` before `torch.topk`, and (b) after computing the new `idx`,
   mark positions present in `old_top_k_blocks` but absent from `idx` as
   newly banned. Gate this behind a config flag (e.g.
   `FCC.demotion_mode == "evict"`) analogous to how `unstable_heads`
   already gates behavior, so `"reactive"`/`"identity"` are unaffected.
5. **Patch the monkeypatch target correctly** — `vllm.attention.ops
   .chunked_prefill_paged_decode.write_top_k_blocks` /
   `.write_top_k_blocks_post` (the star-imported local names), not the
   origin module, confirmed via `grep -rn write_top_k_blocks` against
   whatever FlexiCache commit is actually installed (double-check no
   other file also does `from .flexi_cache_triton_kernels import *`
   before assuming these two call sites are exhaustive).
6. **Verify on wolverine**, in order of ambition, mirroring the
   `docs/phase2b_m1b_smoke_test.md` pattern already used for `"reactive"`:
   a. A synthetic single-request, single-head trace where a hand-picked
      block is forced out of top-K and its score is then forced back
      above threshold — confirm it does **not** re-enter `top_k_blocks`,
      and confirm an *unrelated* concurrently-running request is
      unaffected (catches the slot-reuse leak from point 2/3 above).
   b. Confirm `evict` mode produces a **strictly non-increasing** GPU
      block-table residency per head relative to `offload`/reactive mode
      at the same budget (the whole point of evict — a page, once gone,
      never gets a slot back), and a measurable **quality drop**
      relative to reactive at matched budget (since real KV information
      is now actually gone, not just temporarily off-GPU) — the
      real-system counterpart to `notes/phase2a_residency_results.md`'s
      simulator finding.
   c. Only then wire it into `wakekv/flexicache_shim.py` as
      `mode="evict"`, following the existing `install()`/`_apply_patch()`
      structure, with its own `_apply_config_override`-style unit tests
      against a mock module (the two new call sites can be mocked the
      same way `test_install_patches_a_fake_config_class` already mocks
      `vllm.v1.flexicache.config`).

## 5. Bottom line

The mechanism exists and I can name it precisely, but it's genuinely
embedded in vLLM's continuous-batching internals (request-slot lifecycle
across four methods in a file the current shim never touches) plus a
star-import indirection that makes even the monkeypatch target
non-obvious. That combination is exactly what the task described as "too
deeply embedded to patch safely without live GPU/vLLM testing." I wrote
this doc instead of code so the next person (with wolverine access) has
the precise mechanism, the precise patch point, and the precise
lifecycle hazards up front, rather than rediscovering them after a
silently-wrong run.
