#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "Set RUN_ROOT to the SWE-smith run directory." >&2
  exit 2
fi

OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-$RUN_ROOT/analysis}"
RUNS_PER_TEMPERATURE="${RUNS_PER_TEMPERATURE:-4}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-8}"
mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
args=(
  --run-root "$RUN_ROOT"
  --rollouts-csv "$OUTPUT_DIR/rollouts.csv"
  --tasks-csv "$OUTPUT_DIR/tasks.csv"
  --summary-output "$OUTPUT_DIR/summary.json"
  --runs-per-temperature "$RUNS_PER_TEMPERATURE"
  --total-samples "$TOTAL_SAMPLES"
)
if [[ -n "${EXPECTED_TASKS:-}" ]]; then
  args+=(--expected-tasks "$EXPECTED_TASKS")
fi

"$UV_BIN" run python -m debug_depo.swesmith_analyze "${args[@]}" "$@"
