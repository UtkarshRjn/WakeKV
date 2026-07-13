# M2b-1b Smoke Test — Verify the Shim on Wolverine

Runs after PR #8 is merged. Confirms that the FlexiCache monkey-patch
does what we think it does on a real vLLM+FlexiCache install, before we
start measuring things in M2b-2.

## What we're checking

Three claims, in order of ambition:

1. **Patch takes effect.** ``ShimReport`` at engine startup shows
   ``unstable_heads`` collapsed to zero and ``rerank_frequency`` overridden.
2. **Identity mode is bit-identical to stock FlexiCache.** Running with
   ``--mode identity`` against a fixed prompt must produce the exact
   same tokens as stock. If it does, the patching mechanism itself
   introduces no artifacts.
3. **Reactive mode runs end-to-end.** Same prompt under reactive mode
   completes without errors. Output tokens are allowed to differ from
   stock (different residency policy) — we're just checking it works.

## Setup

```bash
cd ~/FlexiCache
source flexi-env/bin/activate      # whatever venv M2b-0 used
cd ~/dynamic-head-kv && git pull    # get the shim
export WAKEKV_ROOT=$(pwd)
export PYTHONPATH=$WAKEKV_ROOT:$PYTHONPATH
```

## Check 1 — patch takes effect

Uses a tiny script that just imports vLLM and prints the shim report,
without running the model:

```bash
python -c "
import wakekv.flexicache_shim as shim
shim.install(mode='reactive', rerank_interval=1)

# Force FlexiCacheConfig to initialize (as vLLM would).
from vllm.v1.flexicache.config import FlexiCacheConfig
FlexiCacheConfig.initialize(
    model_name='mistralai/Mistral-7B-Instruct-v0.2',
    num_unstable_heads=64,
    topk_budget=64,
    unstable_heads_profile_task='gov_report',
    # ...whatever other kwargs their initialize() takes
)

print(shim.last_report())
"
```

**Expected output:**

```
[wakekv shim] mode=reactive  unstable heads: <64ish> → 0  rerank frequency: 16 → 1
```

If ``unstable heads`` doesn't collapse to 0, the ``_build`` interception
missed. If it says ``0 → 0``, either FlexiCache changed layout or the
config never actually loaded the profile — investigate before continuing.

## Check 2 — identity mode == stock

Same 8-prompt LongBench-slice from M2b-0, run through the shim in
identity mode. Output must be bit-identical to stock FlexiCache.

**Note:** the M2b-1b smoke run on wolverine did this with an in-process
harness (`scratchpad/run_mode.py`), not `benchmarks/benchmark_throughput.py`,
because the throughput benchmark doesn't emit per-prompt token IDs to
diff. A repo-friendly equivalent is on the M2b-2 followup list; the
existing smoke verification is documented in
`notes/wakekv_smoke_test_a30.md`.

The in-process pattern (adapt to your local paths):

```python
# scratchpad/run_mode.py — install shim, run LLM offline, dump token IDs
import os, sys, json
os.environ["WAKEKV_MODE"] = sys.argv[1]         # stock|identity|reactive
os.environ["WAKEKV_RERANK_INTERVAL"] = "1"

if sys.argv[1] != "stock":
    from wakekv import flexicache_shim
    flexicache_shim.install(mode=sys.argv[1], rerank_interval=1)

from vllm import LLM, SamplingParams
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
          enable_flexicache=True, num_unstable_heads=64,
          rerank_frequency=16, topK_budget=64,
          unstable_heads_profile_task="gov_report",
          max_model_len=8192, gpu_memory_utilization=0.90)

prompts = [...]  # 8 prompts, 8k tokens each
outs = llm.generate(prompts, SamplingParams(max_tokens=100, temperature=0.0))
json.dump([o.outputs[0].token_ids for o in outs], open(f"{sys.argv[1]}.json", "w"))
```

Run twice (`stock`, `identity`), then diff the two JSONs.

**Expected: all generations match.** If they don't, the shim's mere
presence is perturbing something (e.g., we're re-running ``_build`` and
the underlying globals aren't idempotent). Diagnose before M2b-2.

## Check 3 — reactive mode runs

Same command as Check 2B but with ``--mode reactive --rerank-interval 1``:

```bash
python $WAKEKV_ROOT/scripts/run_wakekv.py --mode reactive --rerank-interval 1 -- \
    python benchmarks/FlexiCache/Throughput/run_benchmark.py \
    <same args>
```

**Expected:**
- Runs to completion without CUDA errors.
- Tokens per second reported; likely lower than stock at output=100 (we
  saw a 0.93× slowdown at that length even in stock FlexiCache — the
  overhead of frequent reranking should make this more pronounced).
- Output tokens can differ from stock — that's a different policy at work.

Record the throughput number in ``notes/wakekv_smoke_test_a30.md``.

## Rerank-interval sweep — the actual M2b-2 headline plot

Once all three checks pass, run the M2b-2 sweep. See
`docs/phase2b_m2_runbook.md` for the full recipe; the short version:

```bash
bash scripts/wakekv_rerank_sweep.sh        # throughput at rerank ∈ {1,2,4,8,16}
bash scripts/wakekv_longbench_check.sh     # quality at stock + reactive-r1
python scripts/wakekv_sweep_table.py <tp_dir> --quality-dir <q_dir>
```

## What passes M2b-1b

- Check 1 (report), Check 2 (identity == stock), Check 3 (reactive runs).
- Nothing more. Sweep is M2b-2.

## What fails M2b-1b

- Check 1 report shows 0 → 0 → the ``_build`` hook didn't catch the
  actual FlexiCache config. Update ``_flexicache_module_name`` in
  ``wakekv/flexicache_shim.py``, or open the FlexiCache install and
  find where ``self.unstable_heads`` really gets assigned.
- Check 2 diverges → the shim's presence has a side effect. Most likely
  cause: ``_populate_globals`` isn't being re-run after we mutate
  ``self.unstable_heads``. Fix in the shim.
- Check 3 CUDA errors → reactive rerank cadence exposed a bug in
  FlexiCache we didn't know about. Downgrade to identity mode and file
  upstream.
