# M2b-1b Smoke Test — Results (A30, Mistral-7B-v0.2)

Ran `docs/phase2b_m1b_smoke_test.md` on the A30 box against the built
vLLM+FlexiCache env. All three checks pass; Check 3 required a one-line
shim fix (included in this branch).

## Checks

| Check | Result | Evidence |
|---|---|---|
| 1. Patch takes effect | PASS | `unstable heads: 64 → 0  rerank frequency: 16 → 1` |
| 2. Identity == stock | PASS | 8/8 generations bit-identical to stock FlexiCache |
| 3. Reactive runs E2E | PASS (after fix) | 8/8 coherent, no CUDA/assertion errors |

## Fix applied (this branch)

Reactive mode crashed at engine init:

```
AssertionError: Expected total 352801 GPU blocks, got 352800
```

Root cause: the reactive override set `unstable_heads = []` but left the
scalar `num_unstable_heads = 64`. FlexiCache's block-count assertion
(`gpu_model_runner.py:1977`) branches on `num_unstable_heads > 0`; the
inconsistent state took the `>0` branch (expects un-rounded total) while
the block distribution had already allocated the 0-unstable-head total
(rounded down to a multiple of layers) — off by one. FlexiCache *has* a
correct `num_unstable_heads == 0` path; the shim just wasn't zeroing the
scalar. Fix in `_apply_config_override`:

```python
config.num_unstable_heads = 0
config.unstable_heads_portion = 0.0
```

## Two follow-ups needed before the M2b-2 sweep (NOT fixed here)

1. **`scripts/run_wakekv.py` cannot carry the shim.** It installs the
   monkey-patch, then `os.execvp`s a fresh `python` — which replaces the
   process image and discards the in-process patch. The launched
   benchmark therefore runs stock FlexiCache, unpatched. The docstring's
   "execvp shares our process" comment is incorrect. Checks 2/3 here were
   run **in-process** instead (install shim → run vLLM offline in the same
   process, `VLLM_ENABLE_V1_MULTIPROCESSING=0` to keep engine-core
   in-process). The runner must launch vLLM in-process (or have the child
   install the shim itself) before the sweep will measure anything real.

2. **Smoke-doc benchmark reference is stale.** The doc points at
   `benchmarks/FlexiCache/Throughput/run_benchmark.py` (does not exist)
   and diffs a `generations` field that the real `benchmark_throughput.py`
   never emits (it outputs throughput stats only). The in-process harness
   used here (`scratchpad/run_mode.py`) generates and compares token IDs
   directly; fold an equivalent into the repo for M2b-2.

## Repro (in-process, A30)

```bash
conda activate FlexiCache
export PYTHONPATH=~/dynamic-head-kv:$PYTHONPATH
export VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 \
       TORCH_CUDA_ARCH_LIST=8.0 VLLM_ENABLE_V1_MULTIPROCESSING=0
# run_mode.py {stock|identity|reactive}: installs shim in-process, runs LLM offline,
# dumps per-prompt token_ids for diffing.
```
