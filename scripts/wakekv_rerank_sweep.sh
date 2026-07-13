#!/usr/bin/env bash
# M2b-2: WakeKV reactive rerank-interval sweep vs stock FlexiCache.
#
# Runs FlexiCache's real throughput benchmark (benchmarks/benchmark_throughput.py)
# through the WakeKV shim at several rerank intervals, plus a stock-FlexiCache
# baseline, and collects throughput JSONs. The shim is applied via
# scripts/run_wakekv.py, which installs it INSIDE the benchmark's interpreter
# (see wakekv/_shimmed_main.py) so it is live when vLLM initializes.
#
# Scaled for A30 (24GB): input 8k, 24 prompts, output 1000 (the regime where
# stock FlexiCache showed its largest speedup vs vLLM). Override via env.
#
# Usage:  bash scripts/wakekv_rerank_sweep.sh
# Env:    WAKEKV_ROOT, FLEXI_ROOT, INTERVALS, OUTPUT_LEN, NUM_PROMPTS, INPUT_LEN
set -euo pipefail

WAKEKV_ROOT="${WAKEKV_ROOT:-/home/utranjan/dynamic-head-kv}"
FLEXI_ROOT="${FLEXI_ROOT:-/home/utranjan/FlexiCache}"
INTERVALS="${INTERVALS:-1 2 4 8 16}"
INPUT_LEN="${INPUT_LEN:-8000}"
OUTPUT_LEN="${OUTPUT_LEN:-1000}"
NUM_PROMPTS="${NUM_PROMPTS:-24}"

source /opt/conda/etc/profile.d/conda.sh
conda activate FlexiCache
export PYTHONPATH="$WAKEKV_ROOT:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export TORCH_CUDA_ARCH_LIST="8.0"
# Keep vLLM's engine-core in this process so the in-process shim reaches it.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
BENCH_DIR="$FLEXI_ROOT/benchmarks/FlexiCache/Throughput"   # cwd for relative dataset path
BENCH="../../benchmark_throughput.py"                       # -> benchmarks/benchmark_throughput.py
DATASET="Prompts/prompts-Mistral-7B-Instruct-v0.2.json"
OUT_DIR="$BENCH_DIR/Results_M2b2"
mkdir -p "$OUT_DIR"
cd "$BENCH_DIR"

# Common benchmark args (FlexiCache profile loads 64 heads; in reactive mode the
# shim zeros them at runtime — num-unstable-heads must stay 64 so the profile key
# resolves).
common_bench_args() {
  local rerank="$1" outfile="$2"
  echo --dataset-name leval --dataset-path "$DATASET" \
       --model "$MODEL" \
       --gpu-memory-utilization 0.90 --tensor-parallel-size 1 --max-model-len 8192 \
       --no-enable-prefix-caching --disable-cascade-attn \
       --max-num-batched-tokens 8192 --max-num-seqs 16 \
       --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" --num-prompts "$NUM_PROMPTS" \
       --random-range-ratio-input 0.3333 --random-range-ratio-output 1 \
       --seed 42 --output-json "$outfile" \
       --rerank-frequency "$rerank" --topK-budget 64 --num-unstable-heads 64 \
       --unstable_heads_profile_task gov_report --enable-flexicache
}

run_one() {
  local label="$1" mode="$2" rerank="$3"
  local outfile="$OUT_DIR/wakekv-${label}.json"
  if [ -s "$outfile" ]; then
    echo ">>> SKIP (done): $outfile"
    return 0
  fi
  echo "=================================================================="
  echo ">>> $(date) | label=$label mode=$mode rerank=$rerank | in=$INPUT_LEN out=$OUTPUT_LEN n=$NUM_PROMPTS"
  echo "=================================================================="
  if [ "$mode" = "off" ]; then
    # Stock FlexiCache baseline: no shim, native rerank/unstable config.
    python "$BENCH" $(common_bench_args "$rerank" "$outfile") \
      || echo "!!! RUN FAILED ($label)"
  else
    python "$WAKEKV_ROOT/scripts/run_wakekv.py" --mode "$mode" --rerank-interval "$rerank" -- \
      python "$BENCH" $(common_bench_args "$rerank" "$outfile") \
      || echo "!!! RUN FAILED ($label)"
  fi
}

# Baseline: stock FlexiCache (64 unstable heads, native rerank freq 16).
run_one "stock" off 16

# Reactive sweep: 0 unstable heads, rerank every R steps.
for R in $INTERVALS; do
  run_one "reactive-r${R}" reactive "$R"
done

echo "=== ALL RUNS DONE $(date) ==="
python "$WAKEKV_ROOT/scripts/wakekv_sweep_table.py" "$OUT_DIR" || true
