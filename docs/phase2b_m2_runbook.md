# M2b-2 Runbook — Reactive-vs-Frozen Measurement Sweep

The end-to-end recipe for the paper's headline table: reactive residency
against a stock-FlexiCache baseline on the A30, at five rerank intervals.
Runs on wolverine after M2b-1b passes (which it does — see
`notes/wakekv_smoke_test_a30.md`).

## What we're measuring

Two sweeps, same infrastructure:

1. **Throughput** — generation tokens per second at each rerank interval.
   Reveals whether reactive's every-step rerank cost outweighs the
   memory-fluidity benefit. This is the paper's C6 (overhead) claim on
   real hardware.
2. **Quality** — LongBench F1/Rouge at the same rerank intervals. Reveals
   whether the reactive residency preserves output quality. This is the
   paper's C2 (frozen classification leaves quality on the table)
   confirmation from the *other* side — reactive doesn't lose quality
   either.

## Prerequisites

- M2b-0 passed (FlexiCache reproduces on A30, see
  `notes/flexicache_benchmark_a30.md`).
- M2b-1b passed (shim installs correctly, see
  `notes/wakekv_smoke_test_a30.md`).
- PR #9's runner and tabulator on `main` (they are).

## Sweep A — throughput (~2 hours GPU)

```bash
# On wolverine, in the FlexiCache conda env
cd ~/dynamic-head-kv && git pull   # get PR #10 additions
bash scripts/wakekv_rerank_sweep.sh
```

Runs 6 configurations (stock + reactive at `rerank_interval ∈ {1,2,4,8,16}`)
against `benchmarks/benchmark_throughput.py` at input=8000 / output=1000 /
24 prompts (the regime where stock FlexiCache showed 1.67× vs vLLM baseline
in M2b-0). Each JSON goes to `Results_M2b2/wakekv-<label>.json`, skipping
completed runs on re-invocation.

Expected output when it finishes:

```
=== ALL RUNS DONE <timestamp> ===
# M2b-2 — WakeKV reactive rerank-interval sweep (Results_M2b2)

| config       | rerank interval | gen tok/s | elapsed (s) | vs stock |
|--------------|-----------------|-----------|-------------|----------|
| stock        | 16 (native)     |   154.7   |    155.1    | 1.00× ...|
| reactive-r1  | 1               |     ...   |     ...     | ...      |
| ...
```

Regime-of-interest reading:
- If **reactive-r16** matches stock — sanity check passes (0 unstable
  heads with rerank=16 is close to stock behavior; small delta expected).
- If **reactive-r1** trails stock by more than ~30% — the rerank-every-step
  overhead is real and the paper needs to say so honestly.
- If **reactive-r1** *beats* stock — reactive is a net win at this
  workload. Unlikely but worth reporting if it happens.

## Sweep B — quality (~2 hours GPU)

Runs LongBench at stock and reactive-r1 (the two endpoints). The
intermediate rerank intervals aren't needed for quality — if r1
matches stock, everything in between will too.

```bash
bash scripts/wakekv_longbench_check.sh
```

Uses the same LongBench harness M2b-0 used; if the file path differs on
your box, set `BENCH_SCRIPT=/path/to/run_benchmark.py` before invoking.
Runs the 4-task M2b-0 subset (`qasper`, `2wikimqa`, `triviaqa`,
`multi_news`) at 30 samples/task by default; override via `LONGBENCH_TASKS`
and `SAMPLES_PER_TASK`.

## Combined table

```bash
python scripts/wakekv_sweep_table.py \
    ~/FlexiCache/benchmarks/FlexiCache/Throughput/Results_M2b2 \
    --quality-dir ~/FlexiCache/benchmarks/FlexiCache/LongBench/Results_M2b2
```

Prints the throughput table with a `LongBench mean` column and a per-task
breakdown. This is the material for paper Table 4.

## What passes M2b-2

- Sweep A: all 6 configurations complete without crashes; throughput
  numbers plausibly ordered (stock ≈ reactive-r16 > reactive-r8 > ... > reactive-r1
  is the typical shape, but the specific values are the finding).
- Sweep B: reactive-r1 within ~2 points of stock on each task
  (small-sample noise; paper Table already shows FC/dense F1 ratio of
  1.05 at 30 samples/task in M2b-0 — same noise floor applies).

## What fails M2b-2

- Throughput at r1 catastrophically low (say <10% of stock) → the paper
  reframes: reactive isn't a serving win, but the memory-vs-quality
  tradeoff might still be. That's a real finding, not a defeat.
- Quality regresses > 3 points on any task → **stop and diagnose**. Either
  the shim isn't preserving semantics as identity mode suggested it does,
  or the FC's on-CPU MinMax path has a bug at rerank=1.
- Runs crash at engine init → back to the shim, not this runbook.

## What's NOT in this milestone

- Memory measurement (peak GPU bytes, blocks freed). Available from
  `nvidia-smi --loop` during the runs, but not automated here. Add
  post-hoc if the throughput numbers demand it for the paper.
- Rerank interval > 16 or < 1. Only defined values.
- Non-Mistral models. R1-Distill-Llama-8B repeat is a follow-up.

## Recording results

Once both sweeps land, copy the combined table into
`notes/wakekv_sweep_a30.md` and fold the headline number into paper §6 /
Table 4. The exact numbers determine whether the paper's Setup section
needs any wording adjustment.
