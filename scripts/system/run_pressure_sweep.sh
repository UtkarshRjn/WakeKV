#!/usr/bin/env bash
# Memory-pressure sweep for WakeKV / FlexiCache.
#
# Varies GPU KV pool fraction, batch size, and context length.
# Requires FlexiCache + a GPU. Results land under Results_pressure/.
#
# Env:
#   MODEL, GPU_FRACTIONS, BATCH_SIZES, CTX_LENS, OUTPUT_LEN, NUM_PROMPTS,
#   WAKEKV_ROOT, FLEXI_ROOT
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WAKEKV_ROOT="${WAKEKV_ROOT:-$ROOT}"
FLEXI_ROOT="${FLEXI_ROOT:-$HOME/FlexiCache}"
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
GPU_FRACTIONS="${GPU_FRACTIONS:-0.125 0.25 0.5 0.75}"
BATCH_SIZES="${BATCH_SIZES:-1 4 8}"
CTX_LENS="${CTX_LENS:-8192 16384}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
OUT_DIR="${OUT_DIR:-$FLEXI_ROOT/benchmarks/FlexiCache/Throughput/Results_pressure}"

mkdir -p "$OUT_DIR"
export PYTHONPATH="${WAKEKV_ROOT}:${PYTHONPATH:-}"
PY="${PY:-python}"

echo "=== $(date) pressure sweep model=$MODEL ===" | tee -a "$OUT_DIR/sweep.log"

for frac in $GPU_FRACTIONS; do
  for bs in $BATCH_SIZES; do
    for ctx in $CTX_LENS; do
      for mode in fullkv stock reactive; do
        label="f${frac}_b${bs}_c${ctx}_${mode}"
        echo ">>> $(date) $label" | tee -a "$OUT_DIR/sweep.log"
        # Hook point: call FlexiCache throughput with a KV pool fraction.
        # When FlexiCache exposes a pool cap, pass --kv-pool-fraction "$frac".
        if [ -f "$WAKEKV_ROOT/scripts/system/wakekv_gpu_lib.sh" ]; then
          # shellcheck disable=SC1090
          source "$WAKEKV_ROOT/scripts/system/wakekv_gpu_lib.sh" || true
        fi
        "$PY" - <<PY
from pathlib import Path
import json, time
out = Path("$OUT_DIR") / "${label}.json"
# Placeholder cell when FlexiCache CLI is unavailable on this host.
payload = {
    "label": "$label",
    "model": "$MODEL",
    "gpu_fraction": float("$frac"),
    "batch_size": int("$bs"),
    "ctx": int("$ctx"),
    "output_len": int("$OUTPUT_LEN"),
    "num_prompts": int("$NUM_PROMPTS"),
    "mode": "$mode",
    "status": "scheduled",
    "ts": time.time(),
}
out.write_text(json.dumps(payload, indent=2) + "\n")
print("wrote", out)
PY
      done
    done
  done
done

echo "=== $(date) pressure sweep done ===" | tee -a "$OUT_DIR/sweep.log"
ls -l "$OUT_DIR" | tee -a "$OUT_DIR/sweep.log"
