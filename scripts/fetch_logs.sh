#!/usr/bin/env bash
# Download the paper's attention logs into runs/.
#
# The archive is the GitHub release attention-logs-v1: 20 log.npz files
# and the meta.json next to each one, for Qwen2.5-3B (NIAH and multi-turn)
# and the two R1-Distill CoT models. The Mistral-7B/NIAH attention logs
# could not be recovered and are not in the archive.
# Derived reports (summary.md, residency.md, plots) are not included; the
# analyze stage regenerates them.
#
# Skips the download when a log is already present, unless --force.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
URL="https://github.com/UtkarshRjn/dynamic-head-kv/releases/download/attention-logs-v1/wakekv-attention-logs.zip"
SHA256="c302b7308522bf995bce43d682aa5c17349668eb2b26a06fbda7b1bb0695f65a"
SENTINEL="$ROOT/runs/Qwen__Qwen2.5-3B-Instruct/multiturn/recall/log.npz"

if [ "${1:-}" != "--force" ] && [ -f "$SENTINEL" ]; then
  echo "attention logs already present ($SENTINEL); pass --force to re-download"
  exit 0
fi

tmp="$(mktemp "${TMPDIR:-/tmp}/wakekv-attention-logs.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
echo "downloading $URL"
curl -fL --retry 3 -o "$tmp" "$URL"
if command -v shasum >/dev/null 2>&1; then
  echo "$SHA256  $tmp" | shasum -a 256 -c -
elif command -v sha256sum >/dev/null 2>&1; then
  echo "$SHA256  $tmp" | sha256sum -c -
else
  echo "need shasum or sha256sum to check the archive" >&2
  exit 1
fi
unzip -o -q "$tmp" -d "$ROOT"
echo "extracted attention logs under $ROOT/runs"
