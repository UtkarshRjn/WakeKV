#!/usr/bin/env bash
# Commands behind the paper's five model/regime combinations.
#
#   bash scripts/reproduce_paper.sh log       # GPU: write runs/*/log.npz
#   bash scripts/reproduce_paper.sh analyze   # CPU: churn, prediction, simulator
#   bash scripts/reproduce_paper.sh system    # A30-style FlexiCache sweep; needs FLEXI_ROOT
#
# analyze reads gitignored runs/ and does not need a GPU. Fetch the
# paper's logs first with scripts/fetch_logs.sh. The archive has four
# combinations (Qwen2.5-3B NIAH and multi-turn,
# R1-Distill-Qwen-1.5B CoT, R1-Distill-Llama-8B CoT). Mistral-7B/NIAH is
# analyzed only when its log.npz files are present, with R-KV and
# ReasonAlloc pruning every 8 steps instead of 128. Needle depth for that
# Mistral run was not recorded; the log stage uses the harness default 0.5.
#
# Bit-exact table match needs the same logs the paper used. Regenerating
# them follows this matrix; it does not guarantee the same sampled prompts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "python not found" >&2
  exit 1
fi

# Pre-Ampere GPUs need float32 for R1-Distill. Ampere+ can set this to auto.
DTYPE_1_5B="${DTYPE_1_5B:-float32}"

log_one() {
  echo "=================================================================="
  echo ">>> $(date) | $*"
  echo "=================================================================="
  "$PY" scripts/attention/log_attention.py "$@"
}

stage_log() {
  log_one --model Qwen/Qwen2.5-3B-Instruct \
    --task niah --context-tokens 5000 --depths 0.25 0.5 0.75 --seeds 0 1 2
  log_one --model Qwen/Qwen2.5-3B-Instruct --task multiturn
  log_one --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --task cot --max-new-tokens 2048 --limit 5 --dtype "$DTYPE_1_5B"
  log_one --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --task cot --max-new-tokens 2048 --limit 5
  log_one --model mistralai/Mistral-7B-Instruct-v0.2 \
    --task niah --context-tokens 8000 --seeds 0 1 2 3 4
}

analyze_one() {
  local task_dir="$1"
  shift
  if [ ! -d "$task_dir" ]; then
    echo "missing $task_dir (run the log stage first)" >&2
    exit 1
  fi
  "$PY" scripts/attention/analyze_churn.py "$task_dir"
  "$PY" scripts/attention/analyze_wakeups.py "$task_dir"
  "$PY" scripts/attention/analyze_clustering.py "$task_dir"
  "$PY" scripts/attention/simulate_residency.py "$task_dir" "$@"
}

stage_analyze() {
  local combos=(
    runs/Qwen__Qwen2.5-3B-Instruct/niah
    runs/Qwen__Qwen2.5-3B-Instruct/multiturn
    runs/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/cot
    runs/deepseek-ai__DeepSeek-R1-Distill-Llama-8B/cot
  )
  local d
  for d in "${combos[@]}"; do
    analyze_one "$d"
  done
  # Paper appendix: this combo's R-KV and ReasonAlloc use buffer/delta = 8,
  # not the default 128, and those two columns are excluded from the win counts.
  local mistral=runs/mistralai__Mistral-7B-Instruct-v0.2/niah
  if compgen -G "$mistral"/*/log.npz > /dev/null; then
    analyze_one "$mistral" --rkv-buffer 8 --reasonalloc-delta 8
  else
    echo "skip $mistral (no log.npz in this tree)"
  fi
}

stage_system() {
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/system/wakekv_gpu_lib.sh"
  require_flexicache_checkout "$ROOT"
  bash "$SCRIPT_DIR/system/wakekv_rerank_sweep.sh"
  bash "$SCRIPT_DIR/system/wakekv_longbench_check.sh"
  bash "$SCRIPT_DIR/system/wakekv_quality_pareto.sh"
}

usage() {
  echo "usage: bash scripts/reproduce_paper.sh {log|analyze|system}" >&2
  exit 2
}

case "${1:-}" in
  log) stage_log ;;
  analyze) stage_analyze ;;
  system) stage_system ;;
  *) usage ;;
esac
