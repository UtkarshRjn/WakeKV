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

Same 8-prompt LongBench-slice from M2b-0. Two runs, diff the outputs:

```bash
# A) stock FlexiCache, unchanged
cd ~/FlexiCache
python benchmarks/FlexiCache/Throughput/run_benchmark.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --enable-flexicache --num-unstable-heads 64 \
    --rerank-frequency 16 --topK-budget 64 \
    --unstable-heads-profile-task gov_report \
    --input-len 8000 --output-len 100 --num-prompts 8 \
    --output stock_output.json

# B) same command, but through the shim in identity mode
python $WAKEKV_ROOT/scripts/run_wakekv.py --mode identity -- \
    python benchmarks/FlexiCache/Throughput/run_benchmark.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --enable-flexicache --num-unstable-heads 64 \
    --rerank-frequency 16 --topK-budget 64 \
    --unstable-heads-profile-task gov_report \
    --input-len 8000 --output-len 100 --num-prompts 8 \
    --output identity_output.json

# Compare generated tokens
python - <<PY
import json
a = json.load(open('stock_output.json'))
b = json.load(open('identity_output.json'))
match = sum(1 for x, y in zip(a['generations'], b['generations']) if x == y)
print(f"{match}/{len(a['generations'])} generations match")
PY
```

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

Once all three checks pass, run the sweep:

```bash
for R in 1 2 4 8 16; do
    python $WAKEKV_ROOT/scripts/run_wakekv.py --mode reactive --rerank-interval $R -- \
        python benchmarks/FlexiCache/Throughput/run_benchmark.py \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --enable-flexicache --num-unstable-heads 64 \
        --rerank-frequency $R --topK-budget 64 \
        --unstable-heads-profile-task gov_report \
        --input-len 8000 --output-len 1000 --num-prompts 24 \
        --output wakekv_rerank_${R}.json
done
```

Five data points. The output-len=1000 setting hits the regime where
stock FlexiCache showed the largest speedup vs stock vLLM (1.67× on A30).
That's where reactive residency is expected to shine — if it does, we
have the paper's headline Table.

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
