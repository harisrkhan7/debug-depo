#!/usr/bin/env bash
set -euo pipefail

HYPERSTACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HYPERSTACK_DIR/common.sh"

FAMILY="${1:-verified}"
case "$FAMILY" in
  verified)
    RUN_NAME="${RUN_NAME:-agentforge-verified-hyperstack}"
    EXPECTED_TASKS="${EXPECTED_TASKS:-500}"
    ;;
  swesmith)
    RUN_NAME="${RUN_NAME:-swesmith-train-1000}"
    EXPECTED_TASKS="${EXPECTED_TASKS:-1000}"
    RUNS_PER_TEMPERATURE="${RUNS_PER_TEMPERATURE:-4}"
    TEMPERATURES="${TEMPERATURES:-0.6:0.7}"
    read -r -a analysis_temperature_values <<<"${TEMPERATURES//:/ }"
    TOTAL_SAMPLES="${TOTAL_SAMPLES:-$((${#analysis_temperature_values[@]} * RUNS_PER_TEMPERATURE))}"
    export RUNS_PER_TEMPERATURE TEMPERATURES TOTAL_SAMPLES
    ;;
  *)
    echo "Usage: bash hyperstack/analyze.sh verified|swesmith" >&2
    exit 2
    ;;
esac

require_run_name "$RUN_NAME"
RUN_ROOT="${RUN_ROOT:-$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME}"
export RUN_ROOT
export ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-$RUN_ROOT/analysis}"
export EXPECTED_TASKS
export EXPECTED_COUNT="${EXPECTED_COUNT:-$EXPECTED_TASKS}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run: would analyze $FAMILY run at $RUN_ROOT into $ANALYSIS_OUTPUT_DIR."
  exit 0
fi

require_project_environment
cd "$DEBUG_DEPO_ROOT"
if [[ "$FAMILY" == "verified" ]]; then
  scripts/analyze_run.sh
else
  scripts/analyze_swesmith.sh
fi
echo "$FAMILY analysis complete: $ANALYSIS_OUTPUT_DIR"
