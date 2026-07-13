#!/usr/bin/env bash
# M2b-2 quality check: does reactive mode preserve accuracy?
#
# Companion to wakekv_rerank_sweep.sh (throughput). Runs FlexiCache's
# LongBench harness (the one M2b-0 already used) at:
#   - stock FlexiCache (num_unstable_heads=64, rerank_frequency=16), and
#   - WakeKV reactive at rerank_interval=1 (the extreme: every-step rerank),
# then prints per-task F1/Rouge deltas.
#
# Quality is expected to be preserved (fetch-on-demand keeps the "right"
# pages resident by construction — see §6 of the paper draft). If a task
# regresses more than a few points, that is a real finding and blocks
# M2b-2.
#
# Usage:  bash scripts/wakekv_longbench_check.sh
# Env:    WAKEKV_ROOT, FLEXI_ROOT, LONGBENCH_TASKS, SAMPLES_PER_TASK,
#         RERANK_INTERVAL (for the reactive run; default 1).
#
# The LongBench runner path in this script is the SAME one used by M2b-0.
# If your local checkout uses a different runner, override BENCH_SCRIPT.

set -euo pipefail

WAKEKV_ROOT="${WAKEKV_ROOT:-/home/utranjan/dynamic-head-kv}"
FLEXI_ROOT="${FLEXI_ROOT:-/home/utranjan/FlexiCache}"
LONGBENCH_TASKS="${LONGBENCH_TASKS:-qasper 2wikimqa triviaqa multi_news}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-30}"
RERANK_INTERVAL="${RERANK_INTERVAL:-1}"

source /opt/conda/etc/profile.d/conda.sh
conda activate FlexiCache
export PYTHONPATH="$WAKEKV_ROOT:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export TORCH_CUDA_ARCH_LIST="8.0"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
BENCH_DIR="$FLEXI_ROOT/benchmarks/FlexiCache/LongBench"
# Fall back to the M2b-0 runner path if the standard one isn't there.
BENCH_SCRIPT="${BENCH_SCRIPT:-$BENCH_DIR/run_benchmark.py}"
OUT_DIR="$BENCH_DIR/Results_M2b2"
mkdir -p "$OUT_DIR"
cd "$BENCH_DIR"

if [ ! -f "$BENCH_SCRIPT" ]; then
    cat >&2 <<EOF
[wakekv_longbench_check] Cannot find LongBench runner at:
    $BENCH_SCRIPT
The M2b-0 notes say it lives at a similar path but with local edits (see
notes/flexicache_benchmark_a30.md). Set BENCH_SCRIPT=/path/to/run_benchmark.py
and rerun.
EOF
    exit 2
fi

common_bench_args() {
    local rerank="$1" outfile="$2"
    echo --model "$MODEL" \
         --tasks "$LONGBENCH_TASKS" \
         --limit "$SAMPLES_PER_TASK" \
         --max-model-len 8192 \
         --gpu-memory-utilization 0.90 \
         --seed 42 \
         --rerank-frequency "$rerank" \
         --topK-budget 64 --num-unstable-heads 64 \
         --unstable_heads_profile_task gov_report \
         --enable-flexicache \
         --output-json "$outfile"
}

run_one() {
    local label="$1" mode="$2" rerank="$3"
    local outfile="$OUT_DIR/wakekv-longbench-${label}.json"
    if [ -s "$outfile" ]; then
        echo ">>> SKIP (done): $outfile"
        return 0
    fi
    echo "===================================================================="
    echo ">>> $(date) | label=$label mode=$mode rerank=$rerank"
    echo ">>> tasks=$LONGBENCH_TASKS  samples=$SAMPLES_PER_TASK"
    echo "===================================================================="
    if [ "$mode" = "off" ]; then
        python "$BENCH_SCRIPT" $(common_bench_args "$rerank" "$outfile") \
            || echo "!!! RUN FAILED ($label)"
    else
        python "$WAKEKV_ROOT/scripts/run_wakekv.py" \
            --mode "$mode" --rerank-interval "$rerank" -- \
            python "$BENCH_SCRIPT" $(common_bench_args "$rerank" "$outfile") \
            || echo "!!! RUN FAILED ($label)"
    fi
}

run_one "stock" off 16
run_one "reactive-r${RERANK_INTERVAL}" reactive "$RERANK_INTERVAL"

echo "=== quality check done $(date) ==="
python "$WAKEKV_ROOT/scripts/wakekv_sweep_table.py" \
       "$FLEXI_ROOT/benchmarks/FlexiCache/Throughput/Results_M2b2" \
       --quality-dir "$OUT_DIR" || true
