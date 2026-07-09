#!/usr/bin/env bash
# G1 smoke test driver — runs phase-0 collection + phase-0/1 analysis.
set -uo pipefail
cd /home/utranjan/dynamic-head-kv
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
  echo "=================================================================="
  echo ">>> $*"
  echo "=================================================================="
  "$@"
  echo "<<< exit=$? : $*"
  echo
}

run python scripts/run_phase0.py --model Qwen/Qwen2.5-3B-Instruct \
    --task niah --context-tokens 5000 --depths 0.25 0.5 0.75 --seeds 0 1 2 \
    --max-new-tokens 256

run python scripts/run_phase0.py --model Qwen/Qwen2.5-3B-Instruct --task multiturn

run python scripts/run_phase0.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --task cot --max-new-tokens 2048

run python scripts/analyze_phase0.py runs/Qwen__Qwen2.5-3B-Instruct/niah

run python scripts/analyze_phase1.py runs/Qwen__Qwen2.5-3B-Instruct/niah

echo "ALL_DONE"
