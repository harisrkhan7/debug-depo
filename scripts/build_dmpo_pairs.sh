#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/preference_defaults.sh"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "Set RUN_ROOT to the completed SWE-smith run directory." >&2
  exit 2
fi

OUTPUT_DIR="${DMPO_OUTPUT_DIR:-$RUN_ROOT/preference-data/dmpo}"
TOKEN_METRIC="${TOKEN_METRIC:-$PREFERENCE_TOKEN_METRIC_DEFAULT}"
MIN_COST_RATIO="${MIN_COST_RATIO:-1.1}"
MAX_PAIRS_PER_TASK="${MAX_PAIRS_PER_TASK:-0}"
PREFERENCE_MAX_ROLLOUTS="${PREFERENCE_MAX_ROLLOUTS:-$PREFERENCE_MAX_ROLLOUTS_DEFAULT}"
mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${REBUILD_PREFERENCE_DATA:-0}" != "1" ]] && \
  "$ROOT_DIR/scripts/validate_preference_data.sh" dmpo "$OUTPUT_DIR" >/dev/null 2>&1; then
  echo "Reusing complete immutable DMPO data: $OUTPUT_DIR"
  exit 0
fi
args=(
  --run-root "$RUN_ROOT"
  --output "$OUTPUT_DIR/pairs.jsonl"
  --summary-output "$OUTPUT_DIR/summary.json"
  --token-metric "$TOKEN_METRIC"
  --min-cost-ratio "$MIN_COST_RATIO"
  --max-pairs-per-task "$MAX_PAIRS_PER_TASK"
  --max-rollouts "$PREFERENCE_MAX_ROLLOUTS"
)
if [[ -n "${PREFERENCE_SAMPLE_INDICES:-}" ]]; then
  args+=(--sample-indices "$PREFERENCE_SAMPLE_INDICES")
fi
if [[ "${INCLUDE_FAILURE_EFFICIENCY_PAIRS:-0}" == "1" ]]; then
  args+=(--include-failure-efficiency-pairs)
fi

"$UV_BIN" run python -m debug_depo.build_dmpo_pairs "${args[@]}" "$@"
"$ROOT_DIR/scripts/validate_preference_data.sh" dmpo "$OUTPUT_DIR" >/dev/null
