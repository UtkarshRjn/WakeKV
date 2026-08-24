#!/usr/bin/env bash
# Shared host-portability helpers for the M2b-2 real-system scripts
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
# conda.sh's location is the other thing that varies per host (/opt/conda on
# one machine, ~/miniconda3 on another, ...). detect_conda_sh() finds it
# instead of assuming one path, with no safe guess to fall back on -- unlike
# GPU arch, a wrong conda path can't silently "sort of work", so this one
# errors loudly instead of defaulting to anything:
#
#   1. $CONDA_SH, if already set                                 -- respected as-is
#   2. $CONDA_EXE (set by an active conda install)                -- derives ../etc/profile.d/conda.sh
#   3. `conda info --base` (conda is on PATH but not activated)   -- same derivation
#   4. common install locations (/opt/conda, ~/miniconda3, ~/anaconda3, ~/miniforge3)
#   5. none found -- print every location checked and exit nonzero
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/wakekv_gpu_lib.sh"
#   source "$(detect_conda_sh)"
#   conda activate FlexiCache
#   export TORCH_CUDA_ARCH_LIST="$(detect_cuda_arch)"   # after activate -- step 3 needs the target env's python

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

detect_conda_sh() {
  if [ -n "${CONDA_SH:-}" ]; then
    if [ -f "$CONDA_SH" ]; then
      echo "$CONDA_SH"
      return
    fi
    echo "[wakekv_gpu_lib] WARNING: \$CONDA_SH=$CONDA_SH does not exist -- falling back to auto-detection" >&2
  fi

  local candidates=() base
  if [ -n "${CONDA_EXE:-}" ]; then
    candidates+=("$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh")
  fi
  if command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null)"
    [ -n "$base" ] && candidates+=("$base/etc/profile.d/conda.sh")
  fi
  candidates+=(
    "/opt/conda/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "$HOME/miniforge3/etc/profile.d/conda.sh"
  )

  local c
  for c in "${candidates[@]}"; do
    if [ -f "$c" ]; then
      echo "[wakekv_gpu_lib] found conda.sh at $c" >&2
      echo "$c"
      return
    fi
  done

  echo "[wakekv_gpu_lib] ERROR: could not find conda.sh. Checked \$CONDA_SH," \
       "\$CONDA_EXE, 'conda info --base', and: ${candidates[*]}." \
       "Set CONDA_SH explicitly to the conda.sh path for this host." >&2
  return 1
}
