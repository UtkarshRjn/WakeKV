#!/usr/bin/env bash
# Run analyze_wakeups.py over EVERY (model, task) directory under runs/ that
# actually has attention logs, so a single command produces the full signal
# study (four attention signals + ensemble_vote + the data-driven
# token/wake correlation) across every model/task combo you've run so far.
#
# analyze_wakeups.py itself only takes ONE task_dir at a time (it pools every
# run/seed *within* that directory into one report) — this script just walks
# runs/<model>/<task>/ for you and invokes it once per combo found.
#
# Usage:  bash scripts/analyze_wakeups_all.sh [RUNS_ROOT]
# Env:    RUNS_ROOT (default "runs")
# Output: <task_dir>/signal_study.md written by analyze_wakeups.py, plus a
#         combined runs/signal_study_all.md concatenating every report.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAKEKV_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS_ROOT="${1:-$WAKEKV_ROOT/runs}"
OUT_SUMMARY="$RUNS_ROOT/signal_study_all.md"

if [ ! -d "$RUNS_ROOT" ]; then
    echo "no such directory: $RUNS_ROOT" >&2
    exit 1
fi

: > "$OUT_SUMMARY"
echo "# Wake-up prediction — $(date)" >> "$OUT_SUMMARY"

shopt -s nullglob
found_any=0
failed=()
for task_dir in "$RUNS_ROOT"/*/*/; do
    task_dir="${task_dir%/}"
    # Only process directories that look like log_attention.py output: at
    # least one direct child with a log.npz (same test analyze_wakeups.py
    # itself uses to find run dirs).
    if ! compgen -G "$task_dir"/*/log.npz > /dev/null; then
        continue
    fi
    found_any=1
    rel="${task_dir#"$RUNS_ROOT"/}"
    echo "=================================================================="
    echo ">>> $(date) | $rel"
    echo "=================================================================="
    if python "$WAKEKV_ROOT/scripts/analyze_wakeups.py" "$task_dir"; then
        {
            echo ""
            echo "<!-- ============ $rel ============ -->"
            cat "$task_dir/signal_study.md"
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
