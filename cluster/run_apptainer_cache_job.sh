#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PBS_O_WORKDIR:-$ROOT_DIR}"
if [[ ! -f pyproject.toml ]]; then
  echo "Submit this job from the debug-depo repository root." >&2
  exit 2
fi

source cluster/load_job_environment.sh
source cluster/env/load.sh

MODE="${CACHE_BUILD_MODE:-smoke}"
case "$MODE" in
  smoke)
    DEFAULT_WORKERS=2
    ;;
  full)
    DEFAULT_WORKERS=20
    ;;
  *)
    echo "CACHE_BUILD_MODE must be smoke or full, got: $MODE" >&2
    exit 2
    ;;
esac

export CACHE_BUILD_MODE="$MODE"
export CACHE_BUILD_MAX_WORKERS="${CACHE_BUILD_MAX_WORKERS:-$DEFAULT_WORKERS}"
export SWESMITH_TASK_IDS_FILE="${SWESMITH_TASK_IDS_FILE:-${TASK_IDS_FILE:-data/splits/swesmith_cache_5700_instance_ids.txt}}"
SUMMARY_ROOT="${CACHE_BUILD_SUMMARY_ROOT:-$DEBUG_DEPO_SCRATCH/cache-builds}"
mkdir -p "$SUMMARY_ROOT"
job_label="${PBS_JOBID:-manual}"
export CACHE_BUILD_SUMMARY_OUTPUT="${CACHE_BUILD_SUMMARY_OUTPUT:-$SUMMARY_ROOT/${MODE}-${job_label}.json}"

cat <<MSG
Starting Apptainer cache prebuild
  mode:                 $CACHE_BUILD_MODE
  datasets:             ${CACHE_BUILD_DATASETS:-both}
  workers:              $CACHE_BUILD_MAX_WORKERS
  SWE-bench SIFs:       $SWEBENCH_APPTAINER_SIF_DIR
  SWE-smith task IDs:   $SWESMITH_TASK_IDS_FILE
  SWE-smith SIFs:       $SWESMITH_APPTAINER_SIF_DIR
  summary:              $CACHE_BUILD_SUMMARY_OUTPUT
MSG

scripts/build_apptainer_cache.sh
