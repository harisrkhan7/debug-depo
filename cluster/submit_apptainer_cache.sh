#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${CACHE_BUILD_MODE:-smoke}"
case "$MODE" in
  smoke)
    PBS_SCRIPT="cluster/pbs/build_apptainer_cache_smoke.pbs"
    DEFAULT_WORKERS=2
    ;;
  full)
    PBS_SCRIPT="cluster/pbs/build_apptainer_cache_full.pbs"
    DEFAULT_WORKERS=20
    ;;
  *)
    echo "CACHE_BUILD_MODE must be smoke or full, got: $MODE" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-apptainer-cache-$MODE}"
source cluster/resolve_run_paths.sh

DATASETS="${CACHE_BUILD_DATASETS:-both}"
MAX_WORKERS="${CACHE_BUILD_MAX_WORKERS:-$DEFAULT_WORKERS}"
SWESMITH_TASK_IDS_FILE="${SWESMITH_TASK_IDS_FILE:-${TASK_IDS_FILE:-data/splits/swesmith_cache_5700_instance_ids.txt}}"
EXPECTED_SWEBENCH_TASKS="${EXPECTED_SWEBENCH_TASKS:-500}"

if [[ ! "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CACHE_BUILD_MAX_WORKERS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$EXPECTED_SWEBENCH_TASKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_SWEBENCH_TASKS must be a positive integer." >&2
  exit 2
fi
if [[ "$DATASETS" != "both" && "$DATASETS" != "swebench" && "$DATASETS" != "swesmith" ]]; then
  echo "CACHE_BUILD_DATASETS must be both, swebench, or swesmith." >&2
  exit 2
fi
if [[ "$DATASETS" != "swebench" && ! -f "$SWESMITH_TASK_IDS_FILE" ]]; then
  echo "SWE-smith task ID file not found: $SWESMITH_TASK_IDS_FILE" >&2
  exit 2
fi

pbs_variables="CACHE_BUILD_MODE=$MODE,CACHE_BUILD_DATASETS=$DATASETS,CACHE_BUILD_MAX_WORKERS=$MAX_WORKERS,SWESMITH_TASK_IDS_FILE=$SWESMITH_TASK_IDS_FILE,EXPECTED_SWEBENCH_TASKS=$EXPECTED_SWEBENCH_TASKS"
for variable_name in \
  SWEBENCH_TASK_IDS_FILE \
  SWEBENCH_CACHE_LIMIT \
  SWESMITH_CACHE_LIMIT \
  SWEBENCH_DATASET \
  SWEBENCH_DATASET_REVISION \
  SWEBENCH_SPLIT \
  SWESMITH_DATASET \
  SWESMITH_DATASET_REVISION \
  SWESMITH_SPLIT
do
  if [[ -n "${!variable_name:-}" ]]; then
    pbs_variables+=",${variable_name}=${!variable_name}"
  fi
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Dry run: no jobs submitted.\n\n'
  printf 'Cluster logs: %s\n\n' "$CLUSTER_LOG_DIR"
  printf 'Apptainer cache %s job\n' "$MODE"
  printf '  PBS script: %s\n' "$PBS_SCRIPT"
  printf '  Variables:\n'
  while IFS= read -r variable; do
    printf '    %s\n' "$variable"
  done <<<"${pbs_variables//,/$'\n'}"
  exit 0
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub is not available. Run this script on the cluster login node." >&2
  exit 127
fi

mkdir -p "$CLUSTER_LOG_DIR"
job_id="$(qsub \
  -N "apptainer-cache-$MODE" \
  -o "$CLUSTER_LOG_DIR/" \
  -e "$CLUSTER_LOG_DIR/" \
  -v "$pbs_variables" \
  "$PBS_SCRIPT")"
if [[ "${CACHE_SUBMIT_PRINT_JOB_ID_ONLY:-0}" == "1" ]]; then
  printf '%s\n' "$job_id"
else
  echo "Submitted Apptainer cache $MODE build: $job_id"
fi
