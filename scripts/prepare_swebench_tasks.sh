#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data/splits}"
NAME="${NAME:-swebench_verified}"
LIMIT="${LIMIT:-}"
START_INDEX="${START_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --dataset "$DATASET"
  --split "$SPLIT"
  --output-dir "$OUTPUT_DIR"
  --name "$NAME"
  --start-index "$START_INDEX"
  --num-shards "$NUM_SHARDS"
  --shard-index "$SHARD_INDEX"
)

if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ -n "${TASK_IDS_FILE:-}" ]]; then
  args+=(--instance-ids-file "$TASK_IDS_FILE")
fi

"$UV_BIN" run python -m debug_depo.data "${args[@]}" "$@"
