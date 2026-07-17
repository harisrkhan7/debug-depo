#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

OUTPUT="${OUTPUT:-$ROOT_DIR/data/processed/agentforge_swebench_verified/predictions.jsonl}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$ROOT_DIR/data/processed/agentforge_swebench_verified/predictions_summary.json}"

if (($# == 0)); then
  echo "Usage: $0 shard-*/predictions.jsonl ..." >&2
  exit 2
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$UV_BIN" run python -m debug_depo.metrics \
  --input "$@" \
  --output "$OUTPUT" \
  --summary-output "$SUMMARY_OUTPUT"
