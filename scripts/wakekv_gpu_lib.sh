#!/usr/bin/env bash
# Shared GPU-detection helper for the M2b-2 real-system scripts
# (wakekv_rerank_sweep.sh, wakekv_quality_pareto.sh, wakekv_longbench_check.sh).
#
# TORCH_CUDA_ARCH_LIST tells the CUDA build/JIT toolchain which compute
# capability to target. Getting it wrong doesn't error loudly -- it
# compiles kernels for the wrong architecture (a slow PTX-JIT fallback at
# best). Rather than hardcode one card's value, detect_cuda_arch() asks
# whatever GPU is actually attached, with an explicit override always
# taking precedence and a clearly-logged fallback if detection fails:
#
#   1. $TORCH_CUDA_ARCH_LIST, if already set in the environment  -- respected as-is
#   2. `nvidia-smi --query-gpu=compute_cap`                      -- fast, no Python needed
#   3. `python -c "import torch; ..."`                           -- works if step 2's
#      tool is unavailable but the target conda env has torch installed
#   4. 8.0 (Ampere), with a loud warning -- this is the only step that guesses
#
# Usage (after `conda activate` -- step 3 needs the target env's python):
#   source "$(dirname "${BASH_SOURCE[0]}")/wakekv_gpu_lib.sh"
#   export TORCH_CUDA_ARCH_LIST="$(detect_cuda_arch)"

detect_cuda_arch() {
  if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    echo "$TORCH_CUDA_ARCH_LIST"
    return
  fi

  local detected
  detected="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
  if [ -n "$detected" ]; then
    echo "[wakekv_gpu_lib] detected compute capability $detected via nvidia-smi" >&2
    echo "$detected"
    return
  fi

  detected="$(python -c 'import torch; c = torch.cuda.get_device_capability(); print(f"{c[0]}.{c[1]}")' 2>/dev/null)"
  if [ -n "$detected" ]; then
    echo "[wakekv_gpu_lib] detected compute capability $detected via torch.cuda.get_device_capability()" >&2
    echo "$detected"
    return
  fi

  echo "[wakekv_gpu_lib] WARNING: could not detect GPU compute capability" \
       "(no nvidia-smi, no importable torch) -- defaulting to 8.0 (Ampere)." \
       "Set TORCH_CUDA_ARCH_LIST explicitly if this host is not Ampere." >&2
  echo "8.0"
}
