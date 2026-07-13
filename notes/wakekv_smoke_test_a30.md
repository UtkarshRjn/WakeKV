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

## Runner fix (this branch): `scripts/run_wakekv.py` now carries the shim

The old runner installed the monkey-patch, then `os.execvp`'d a fresh
`python` — which replaces the process image and discards the in-process
patch, so the launched benchmark ran **stock** FlexiCache, unpatched. (The
docstring's "execvp shares our process" comment was wrong.)

Fixed by installing the shim *inside the target's interpreter*: the runner
now execs the target through `python -m wakekv._shimmed_main`, a bootstrap
that installs the shim (from `WAKEKV_MODE` / `WAKEKV_RERANK_INTERVAL` env
vars) and then `runpy`'s the real target. Verified end-to-end on A30 — a
target that does *not* self-install the shim still shows
`unstable heads: 64 → 0` after launch through the runner, and generates
without the block-count crash. Handles both `python script.py` and
`python -m module` targets; rejects non-Python targets with a clear error.
Unit-tested in `tests/test_run_wakekv.py`.

## Remaining follow-up before M2b-2

**Smoke-doc benchmark reference is stale.** The doc points at
`benchmarks/FlexiCache/Throughput/run_benchmark.py` (does not exist) and
diffs a `generations` field that the real `benchmarks/benchmark_throughput.py`
never emits (throughput stats only). The in-process harness used for the
smoke checks (`scratchpad/run_mode.py`) generates and compares token IDs
directly; fold an equivalent into the repo, and point the sweep at the real
`benchmarks/benchmark_throughput.py`, for M2b-2.

## Repro (in-process, A30)

```bash
conda activate FlexiCache
export PYTHONPATH=~/dynamic-head-kv:$PYTHONPATH
export VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 \
       TORCH_CUDA_ARCH_LIST=8.0 VLLM_ENABLE_V1_MULTIPROCESSING=0
# run_mode.py {stock|identity|reactive}: installs shim in-process, runs LLM offline,
# dumps per-prompt token_ids for diffing.
```
