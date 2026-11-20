#!/usr/bin/env bash
# Two hardware cells the A30 sweep did not include:
#   full KV (FlexiCache off)
#   stock's 64 dense heads with rerank interval 1
# Throughput only. LongBench is a separate, longer pass.
set -uo pipefail

WAKEKV_ROOT="${WAKEKV_ROOT:-$HOME/dynamic-head-kv}"
FLEXI_ROOT="${FLEXI_ROOT:-$HOME/FlexiCache}"
PY="${PY:-$HOME/.conda/envs/FlexiCache/bin/python}"
export PATH="$(dirname "$PY"):$PATH"
# nvcc invokes cicc by name. The notebook image keeps it in nvvm/bin,
# which is not on PATH, so FlexiCache's JIT extension fails with exit 127.
if [ -d /opt/conda/nvvm/bin ]; then
  export PATH="/opt/conda/nvvm/bin:$PATH"
fi
export LD_LIBRARY_PATH="$HOME/.conda/envs/FlexiCache/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WAKEKV_ROOT:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
BENCH_DIR="$FLEXI_ROOT/benchmarks/FlexiCache/Throughput"
BENCH="$FLEXI_ROOT/benchmarks/benchmark_throughput.py"
DATASET="$BENCH_DIR/Prompts/prompts-Mistral-7B-Instruct-v0.2.json"
OUT_DIR="$BENCH_DIR/Results_review"
mkdir -p "$OUT_DIR"
cd "$BENCH_DIR"

echo "=== $(date) host=$(hostname) gpu ==="
nvidia-smi -L || true
"$PY" -c "import vllm,torch; print('vllm', vllm.__file__); print('cuda', torch.cuda.get_device_name(0))"

common() {
  local outfile="$1"
  shift
  echo --dataset-name leval --dataset-path "$DATASET" \
    --model "$MODEL" \
    --gpu-memory-utilization 0.90 --tensor-parallel-size 1 --max-model-len 10240 \
    --no-enable-prefix-caching --disable-cascade-attn \
    --max-num-batched-tokens 8192 --max-num-seqs 16 \
    --input-len 8000 --output-len 1000 --num-prompts 24 \
    --random-range-ratio-input 0.3333 --random-range-ratio-output 1 \
    --seed 42 --output-json "$outfile" "$@"
}

run_fullkv() {
  local outfile="$OUT_DIR/wakekv-fullkv.json"
  if [ -s "$outfile" ]; then echo "SKIP $outfile"; return 0; fi
  echo "=== $(date) fullkv ==="
  "$PY" "$BENCH" $(common "$outfile") || echo "FAILED fullkv"
}

run_dense_r1() {
  local outfile="$OUT_DIR/wakekv-dense-r1.json"
  if [ -s "$outfile" ]; then echo "SKIP $outfile"; return 0; fi
  echo "=== $(date) 64 dense heads, R=1 ==="
  WAKEKV_MODE=cadence WAKEKV_RERANK_INTERVAL=1 \
    "$PY" -m wakekv._shimmed_main "$BENCH" \
    $(common "$outfile" \
      --rerank-frequency 1 --topK-budget 64 --num-unstable-heads 64 \
      --unstable_heads_profile_task gov_report --enable-flexicache) \
    || echo "FAILED dense-r1"
}

run_longbench() {
  local label="$1" mode="$2" rerank="$3"
  local bench_dir="$FLEXI_ROOT/benchmarks/FlexiCache/Language_Modelling/LongBench"
  local out="$bench_dir/Results_review/${label}.json"
  mkdir -p "$bench_dir/Results_review"
  if [ -s "$out" ]; then echo "SKIP $out"; return 0; fi
  echo "=== $(date) longbench $label mode=$mode R=$rerank ==="
  cd "$bench_dir"
  rm -rf pred results
  local args=(run_benchmark.py --model Mistral-7B-Instruct-v0.2
    --dataset qasper 2wikimqa triviaqa multi_news
    --batch_size 4 --limit 30)
  if [ "$mode" != "fullkv" ]; then
    args+=(--flexicache --num_unstable_heads 64 --rerank_frequency "$rerank"
      --topK_budget 64 --unstable_heads_profile_task gov_report)
  fi
  if [ "$mode" = "cadence" ]; then
    WAKEKV_MODE=cadence WAKEKV_RERANK_INTERVAL="$rerank" \
      "$PY" -m wakekv._shimmed_main "${args[@]}" || { echo "FAILED $label"; return 0; }
  else
    "$PY" "${args[@]}" || { echo "FAILED $label"; return 0; }
  fi
  "$PY" eval.py || { echo "FAILED eval $label"; return 0; }
  cp -a results/pred.json "$bench_dir/Results_review/${label}.pred.json" || true
  cd "$BENCH_DIR"
}

run_fullkv
run_dense_r1
run_longbench wakekv-longbench-fullkv fullkv 0
run_longbench wakekv-longbench-dense-r1 cadence 1
echo "=== $(date) gap cells done ==="
ls -l "$OUT_DIR" "$FLEXI_ROOT/benchmarks/FlexiCache/Language_Modelling/LongBench/Results_review" 2>/dev/null || true
