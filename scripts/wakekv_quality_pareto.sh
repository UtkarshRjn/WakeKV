#!/usr/bin/env bash
# M2b-2 quality PARETO: LongBench accuracy for stock + reactive at every
# rerank interval, so the quality column sits next to the full throughput
# sweep (wakekv_rerank_sweep.sh).
#
# Each reactive interval is scored in ISOLATION (its own eval pass) because
# run_benchmark.py derives the prediction filename from CLI args, so reactive
# at R=16 would otherwise collide with the stock (rerank=16) predictions.
# Idempotent: a config whose wakekv-longbench-<label>.json already exists is
# skipped, so the stock + reactive-r1 results from the earlier run are reused
# and a teardown just resumes.
#
# GPU-portable: TORCH_CUDA_ARCH_LIST -- the only hardware-specific value
# here -- is now auto-detected from whatever GPU is actually attached
# (see wakekv_gpu_lib.sh), with an explicit env var still taking
# precedence if you set one. WAKEKV_MAX_MODEL_LEN already defaults to an
# A30-safe value and is overridable if you want to use more of a bigger
# card's memory (H100: 80GB vs A30's 24GB) instead of reproducing A30
# exactly.
#
# Usage:  bash scripts/wakekv_quality_pareto.sh
# Env:    INTERVALS (default "1 2 4 8 16"), LONGBENCH_TASKS, SAMPLES_PER_TASK,
#         BATCH_SIZE, WAKEKV_MAX_MODEL_LEN, WAKEKV_ROOT, FLEXI_ROOT,
#         TORCH_CUDA_ARCH_LIST (auto-detected if unset -- see wakekv_gpu_lib.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/wakekv_gpu_lib.sh"

WAKEKV_ROOT="${WAKEKV_ROOT:-/home/utranjan/dynamic-head-kv}"
FLEXI_ROOT="${FLEXI_ROOT:-/home/utranjan/FlexiCache}"
INTERVALS="${INTERVALS:-1 2 4 8 16}"
LONGBENCH_TASKS="${LONGBENCH_TASKS:-qasper 2wikimqa triviaqa multi_news}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
export WAKEKV_MAX_MODEL_LEN="${WAKEKV_MAX_MODEL_LEN:-8192}"

source /opt/conda/etc/profile.d/conda.sh
conda activate FlexiCache
export PYTHONPATH="$WAKEKV_ROOT:${PYTHONPATH:-}"
export VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export TORCH_CUDA_ARCH_LIST="$(detect_cuda_arch)"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

MODEL="Mistral-7B-Instruct-v0.2"
BENCH_DIR="$FLEXI_ROOT/benchmarks/FlexiCache/Language_Modelling/LongBench"
OUT_DIR="$BENCH_DIR/Results_M2b2"
mkdir -p "$OUT_DIR"
cd "$BENCH_DIR"

# Score whatever single config sits in results/pred.json into a label JSON.
reshape_one() {   # $1 = cfg stem, $2 = out label
    python - "$MODEL" "$1" "$OUT_DIR/$2" <<'PY'
import json, os, sys
model, cfg, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
scores = json.load(open("results/pred.json"))[model]
per = {t: c[cfg] for t, c in scores.items() if cfg in c}
if per:
    json.dump({"per_task": per}, open(outpath, "w"), indent=2)
    print("wrote", os.path.basename(outpath), per)
else:
    print("NO SCORES for", cfg); sys.exit(1)
PY
}

run_isolated() {   # $1 = rerank, $2 = mode(off|reactive), $3 = out label
    local R="$1" mode="$2" label="$3"
    if [ -s "$OUT_DIR/$label" ]; then
        echo ">>> SKIP (present): $label"; return 0
    fi
    echo "==== $(date) $label : rerank=$R mode=$mode ===="
    rm -rf pred results 2>/dev/null || true
    local args=(run_benchmark.py --model "$MODEL" --dataset $LONGBENCH_TASKS
                --batch_size "$BATCH_SIZE" --limit "$SAMPLES_PER_TASK"
                --flexicache --num_unstable_heads 64 --rerank_frequency "$R"
                --topK_budget 64 --unstable_heads_profile_task gov_report)
    if [ "$mode" = "off" ]; then
        python "${args[@]}" || { echo "!!! GEN FAILED $label"; return 0; }
    else
        python "$WAKEKV_ROOT/scripts/run_wakekv.py" --mode reactive --rerank-interval "$R" -- \
            python "${args[@]}" || { echo "!!! GEN FAILED $label"; return 0; }
    fi
    python eval.py
    reshape_one "flexicache-64-unstable-${R}-rerank-64-topK" "$label" || echo "!!! RESHAPE FAILED $label"
}

# FullKV reference point: no FlexiCache at all (dense attention, full
# cache). Pairs with wakekv_rerank_sweep.sh's fullkv throughput row so
# Table 4 gets a genuine ceiling, not just a comparison among compressed
# policies.
run_fullkv_quality() {
    local label="wakekv-longbench-fullkv.json"
    if [ -s "$OUT_DIR/$label" ]; then
        echo ">>> SKIP (present): $label"; return 0
    fi
    echo "==== $(date) $label : dense (no FlexiCache) ===="
    rm -rf pred results 2>/dev/null || true
    python run_benchmark.py --model "$MODEL" --dataset $LONGBENCH_TASKS \
        --batch_size "$BATCH_SIZE" --limit "$SAMPLES_PER_TASK" \
        || { echo "!!! GEN FAILED $label"; return 0; }
    python eval.py
    # UNVERIFIED: without --flexicache, run_benchmark.py almost certainly
    # writes results/pred.json under a DIFFERENT config-string key than
    # FlexiCache's "flexicache-64-unstable-*-rerank-*-topK" pattern (that
    # pattern is FlexiCache-specific naming). "dense" below is a guess, not
    # something checked against the actual harness. If reshape fails, open
    # results/pred.json, find the real key for this run, and fix the string.
    reshape_one "dense" "$label" \
        || echo "!!! RESHAPE FAILED $label — inspect results/pred.json for the real config key and fix the cfg string in run_fullkv_quality()"
}
run_fullkv_quality

# stock + reactive-r1 already exist from wakekv_longbench_check.sh (skipped).
run_isolated 16 off      "wakekv-longbench-stock.json"
for R in $INTERVALS; do
    run_isolated "$R" reactive "wakekv-longbench-reactive-r${R}.json"
done

echo "==== quality pareto done $(date) ===="
python "$WAKEKV_ROOT/scripts/wakekv_sweep_table.py" \
       "$FLEXI_ROOT/benchmarks/FlexiCache/Throughput/Results_M2b2" \
       --quality-dir "$OUT_DIR" || true
