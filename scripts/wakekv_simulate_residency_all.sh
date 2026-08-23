#!/usr/bin/env bash
# Run simulate_residency.py over EVERY (model, task) directory under runs/
# that actually has Phase-0 logs, so a single command produces the full
# residency simulation (C2: reactive vs frozen at matched memory, C3:
# reactive/offload vs evict/destroy at the same budget) across every
# model/task combo you've run so far.
#
# simulate_residency.py itself only takes ONE task_dir at a time (it pools
# every run/seed *within* that directory into one report) — this script
# just walks runs/<model>/<task>/ for you and invokes it once per combo
# found. No GPU needed (same as simulate_residency.py itself).
#
# Usage:  bash scripts/wakekv_simulate_residency_all.sh [RUNS_ROOT]
# Env:    RUNS_ROOT (default "runs")
# Output: <task_dir>/residency.md written as before by simulate_residency.py,
#         PLUS a combined residency_all.md at the repo root concatenating
#         every report (with a header naming the model/task) for easy diffing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAKEKV_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS_ROOT="${1:-$WAKEKV_ROOT/runs}"
OUT_SUMMARY="$WAKEKV_ROOT/residency_all.md"

if [ ! -d "$RUNS_ROOT" ]; then
    echo "no such directory: $RUNS_ROOT" >&2
    exit 1
fi

: > "$OUT_SUMMARY"
echo "# Combined Phase 2a residency simulation — $(date)" >> "$OUT_SUMMARY"

shopt -s nullglob
found_any=0
failed=()
for task_dir in "$RUNS_ROOT"/*/*/; do
    task_dir="${task_dir%/}"
    # Only process directories that look like run_phase0.py output: at
    # least one direct child with a log.npz (same test simulate_residency.py
    # itself uses to find run dirs).
    if ! compgen -G "$task_dir"/*/log.npz > /dev/null; then
        continue
    fi
    found_any=1
    rel="${task_dir#"$RUNS_ROOT"/}"
    echo "=================================================================="
    echo ">>> $(date) | $rel"
    echo "=================================================================="
    if python "$WAKEKV_ROOT/scripts/simulate_residency.py" "$task_dir"; then
        {
            echo ""
            echo "<!-- ============ $rel ============ -->"
            cat "$task_dir/residency.md"
        } >> "$OUT_SUMMARY"
    else
        echo "!!! FAILED: $rel" >&2
        failed+=("$rel")
    fi
done

if [ "$found_any" -eq 0 ]; then
    echo "no runs with log.npz found under $RUNS_ROOT" >&2
    exit 1
fi

echo "=== ALL DONE $(date) ==="
echo "combined report: $OUT_SUMMARY"
if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED (${#failed[@]}): ${failed[*]}" >&2
    exit 1
fi
