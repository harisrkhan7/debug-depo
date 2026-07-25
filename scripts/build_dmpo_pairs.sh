#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "Set RUN_ROOT to the completed SWE-smith run directory." >&2
  exit 2
fi

OUTPUT_DIR="${DMPO_OUTPUT_DIR:-$RUN_ROOT/preference-data/dmpo}"
TOKEN_METRIC="${TOKEN_METRIC:-total_tokens}"
MIN_COST_RATIO="${MIN_COST_RATIO:-1.1}"
MAX_PAIRS_PER_TASK="${MAX_PAIRS_PER_TASK:-0}"
mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
args=(
  --run-root "$RUN_ROOT"
  --output "$OUTPUT_DIR/pairs.jsonl"
  --summary-output "$OUTPUT_DIR/summary.json"
  --token-metric "$TOKEN_METRIC"
  --min-cost-ratio "$MIN_COST_RATIO"
  --max-pairs-per-task "$MAX_PAIRS_PER_TASK"
)
if [[ "${INCLUDE_FAILURE_EFFICIENCY_PAIRS:-0}" == "1" ]]; then
  args+=(--include-failure-efficiency-pairs)
fi

"$UV_BIN" run python -m debug_depo.build_dmpo_pairs "${args[@]}" "$@"
