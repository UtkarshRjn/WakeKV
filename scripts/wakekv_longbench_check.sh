#!/usr/bin/env bash
# M2b-2 quality check: does reactive mode preserve LongBench accuracy?
#
# Companion to wakekv_rerank_sweep.sh (throughput). Runs FlexiCache's real
# LongBench harness (benchmarks/FlexiCache/Language_Modelling/LongBench) at:
#   - stock FlexiCache (num_unstable_heads=64, rerank_frequency=16), and
#   - WakeKV reactive at rerank_interval=$RERANK_INTERVAL (default 1),
# scores both with the harness's eval.py, and writes per-config quality JSONs
# ({"per_task": {task: score}}) that wakekv_sweep_table.py --quality-dir reads.
#
# Quality is expected to be preserved; a task regressing more than a few points
# is a real finding and blocks M2b-2.
#
# GPU-portable: TORCH_CUDA_ARCH_LIST is auto-detected from whatever GPU
# is actually attached (see wakekv_gpu_lib.sh); an explicit env var still
# takes precedence if you set one.
#
# Usage:  bash scripts/wakekv_longbench_check.sh
# Env:    WAKEKV_ROOT, FLEXI_ROOT, LONGBENCH_TASKS, SAMPLES_PER_TASK,
#         RERANK_INTERVAL, BATCH_SIZE, WAKEKV_MAX_MODEL_LEN,
#         TORCH_CUDA_ARCH_LIST, CONDA_SH (both auto-detected if unset --
#         see wakekv_gpu_lib.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/wakekv_gpu_lib.sh"

WAKEKV_ROOT="${WAKEKV_ROOT:-/home/utranjan/dynamic-head-kv}"
FLEXI_ROOT="${FLEXI_ROOT:-/home/utranjan/FlexiCache}"
LONGBENCH_TASKS="${LONGBENCH_TASKS:-qasper 2wikimqa triviaqa multi_news}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-30}"
RERANK_INTERVAL="${RERANK_INTERVAL:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
export WAKEKV_MAX_MODEL_LEN="${WAKEKV_MAX_MODEL_LEN:-8192}"

source "$(detect_conda_sh)"
conda activate FlexiCache
export PYTHONPATH="$WAKEKV_ROOT:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export TORCH_CUDA_ARCH_LIST="$(detect_cuda_arch)"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

MODEL="Mistral-7B-Instruct-v0.2"                     # run_benchmark.py model KEY (not HF path)
BENCH_DIR="$FLEXI_ROOT/benchmarks/FlexiCache/Language_Modelling/LongBench"
BENCH_SCRIPT="${BENCH_SCRIPT:-$BENCH_DIR/run_benchmark.py}"
OUT_DIR="$BENCH_DIR/Results_M2b2"
mkdir -p "$OUT_DIR"
cd "$BENCH_DIR"

if [ ! -f "$BENCH_SCRIPT" ]; then
    echo "[wakekv_longbench_check] LongBench runner not found: $BENCH_SCRIPT" >&2
    exit 2
fi

# Stock and reactive use DIFFERENT rerank_frequency, so their prediction
# filenames (flexicache-64-unstable-<R>-rerank-64-topK) don't collide.
STOCK_R=16
REACT_R="$RERANK_INTERVAL"
if [ "$REACT_R" = "$STOCK_R" ]; then
    echo "[wakekv_longbench_check] RERANK_INTERVAL must differ from stock's 16 to avoid a pred-file collision." >&2
    exit 2
fi

# Fresh predictions so eval.py only scores this run's configs.
rm -rf pred results 2>/dev/null || true

bench_cmd() {   # $1 = rerank_frequency
    echo "$BENCH_SCRIPT" --model "$MODEL" --dataset $LONGBENCH_TASKS \
         --batch_size "$BATCH_SIZE" --limit "$SAMPLES_PER_TASK" \
         --flexicache --num_unstable_heads 64 --rerank_frequency "$1" \
         --topK_budget 64 --unstable_heads_profile_task gov_report
}

echo "==== $(date) STOCK FlexiCache (unstable=64, rerank=$STOCK_R) ===="
python $(bench_cmd "$STOCK_R")

echo "==== $(date) REACTIVE (shim: unstable=0, rerank=$REACT_R) ===="
python "$WAKEKV_ROOT/scripts/run_wakekv.py" --mode reactive --rerank-interval "$REACT_R" -- \
    python $(bench_cmd "$REACT_R")

echo "==== $(date) scoring (eval.py) ===="
python eval.py

# Reshape results/pred.json -> per-config wakekv-longbench-<label>.json.
STOCK_CFG="flexicache-64-unstable-${STOCK_R}-rerank-64-topK"
REACT_CFG="flexicache-64-unstable-${REACT_R}-rerank-64-topK"
python - "$MODEL" "$STOCK_CFG" "$REACT_CFG" "$REACT_R" "$OUT_DIR" <<'PY'
import json, os, sys
model, stock_cfg, react_cfg, react_r, out_dir = sys.argv[1:6]
scores = json.load(open("results/pred.json"))[model]   # {task: {config: score}}
def reshape(cfg):
    return {"per_task": {task: c[cfg] for task, c in scores.items() if cfg in c}}
json.dump(reshape(stock_cfg), open(os.path.join(out_dir, "wakekv-longbench-stock.json"), "w"), indent=2)
json.dump(reshape(react_cfg), open(os.path.join(out_dir, f"wakekv-longbench-reactive-r{react_r}.json"), "w"), indent=2)
print("wrote wakekv-longbench-stock.json and wakekv-longbench-reactive-r%s.json" % react_r)
PY

echo "==== quality check done $(date) ===="
python "$WAKEKV_ROOT/scripts/wakekv_sweep_table.py" \
       "$FLEXI_ROOT/benchmarks/FlexiCache/Throughput/Results_M2b2" \
       --quality-dir "$OUT_DIR" || true
