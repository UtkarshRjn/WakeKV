#!/usr/bin/env bash
# One needle run at 65,536 context tokens. Records whether the full KV fits.
set -uo pipefail
ROOT="${WAKEKV_ROOT:-$HOME/dynamic-head-kv}"
PY="${PY:-$HOME/.conda/envs/FlexiCache/bin/python}"
export PATH="$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$HOME/.conda/envs/FlexiCache/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
echo "=== $(date) long-context needle ==="
nvidia-smi -L || true
"$PY" scripts/log_attention.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --task niah --context-tokens 65536 --depths 0.5 --seeds 0 \
  --max-new-tokens 32 --dtype bfloat16 \
  --out runs/longctx \
  || echo "FAILED long-context needle"
echo "=== $(date) long-context done ==="
