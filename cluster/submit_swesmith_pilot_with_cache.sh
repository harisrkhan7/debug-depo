#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-swesmith-pilot}"
source cluster/resolve_run_paths.sh
TASK_LIMIT="${TASK_LIMIT:-30}"
EXPECTED_TASKS="${EXPECTED_TASKS:-$TASK_LIMIT}"
NUM_SHARDS="${NUM_SHARDS:-3}"
TASK_IDS_FILE="${TASK_IDS_FILE:-data/splits/swesmith_train_5000_instance_ids.txt}"
DATASET="${DATASET:-SWE-bench/SWE-smith-py}"
SWESMITH_DATASET_REVISION="${SWESMITH_DATASET_REVISION:-77cab9055d42ab4a5c25c89a8f937096db13558e}"
SPLIT="${SPLIT:-train}"
CACHE_BUILD_MAX_WORKERS="${CACHE_BUILD_MAX_WORKERS:-8}"

if [[ ! -f "$TASK_IDS_FILE" ]]; then
  echo "Pilot task ID file not found: $TASK_IDS_FILE" >&2
  exit 2
fi

submit_cache() {
  CACHE_BUILD_MODE=full \
  CACHE_BUILD_DATASETS=swesmith \
  CACHE_BUILD_MAX_WORKERS="$CACHE_BUILD_MAX_WORKERS" \
  SWESMITH_TASK_IDS_FILE="$TASK_IDS_FILE" \
  SWESMITH_CACHE_LIMIT="$TASK_LIMIT" \
  SWESMITH_DATASET="$DATASET" \
  SWESMITH_DATASET_REVISION="$SWESMITH_DATASET_REVISION" \
  SWESMITH_SPLIT="$SPLIT" \
    cluster/submit_apptainer_cache_full.sh
}

submit_pipeline() {
  local cache_job_id="$1"
  SWESMITH_MODE=pilot \
  RUN_NAME="$RUN_NAME" \
  TASK_LIMIT="$TASK_LIMIT" \
  EXPECTED_TASKS="$EXPECTED_TASKS" \
  NUM_SHARDS="$NUM_SHARDS" \
  TASK_IDS_FILE="$TASK_IDS_FILE" \
  DATASET="$DATASET" \
  SWESMITH_DATASET_REVISION="$SWESMITH_DATASET_REVISION" \
  SPLIT="$SPLIT" \
  SUBMIT_EVAL=1 \
  SUBMIT_ANALYSIS=1 \
  AFTEROK_JOB_ID="$cache_job_id" \
    cluster/submit_swesmith_pilot.sh
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Pilot dependency chain: cache -> collect -> evaluate -> analyze\n\n'
  submit_cache
  printf '\n'
  submit_pipeline "<cache-job-id>"
  exit 0
fi

cache_job="$(
  CACHE_SUBMIT_PRINT_JOB_ID_ONLY=1 \
  submit_cache
)"
echo "Submitted pilot-scoped SWE-smith cache build: $cache_job"
submit_pipeline "$cache_job"
