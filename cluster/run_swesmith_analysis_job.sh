#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PBS_O_WORKDIR:-$ROOT_DIR}"
source cluster/load_job_environment.sh
source cluster/env/load.sh

MODE="${SWESMITH_MODE:-pilot}"
case "$MODE" in
  smoke) DEFAULT_RUN_NAME=swesmith-smoke ;;
  pilot) DEFAULT_RUN_NAME=swesmith-pilot ;;
  full) DEFAULT_RUN_NAME=swesmith-full ;;
  *)
    echo "SWESMITH_MODE must be smoke, pilot, or full, got: $MODE" >&2
    exit 2
    ;;
esac
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
export RUN_ROOT="${RUN_ROOT:-$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME}"
export ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-$RUN_ROOT/analysis}"
scripts/analyze_swesmith.sh
