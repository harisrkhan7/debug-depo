#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$CLOUD_DIR/common.sh"

: "${RUN_NAME:?Set RUN_NAME for this validation run.}"
require_run_name "$RUN_NAME"

: "${TASK_IDS_FILE:?Set TASK_IDS_FILE to the SWE-smith task IDs to validate.}"
if [[ -s "$TASK_IDS_FILE" ]]; then
  resolved_task_ids_file="$TASK_IDS_FILE"
else
  resolved_task_ids_file="$DEBUG_DEPO_ROOT/$TASK_IDS_FILE"
fi

task_count="$(awk 'NF { count++ } END { print count + 0 }' "$resolved_task_ids_file")"
EXPECTED_TASKS="${EXPECTED_TASKS:-$task_count}"

validation_model="${MODEL_PATH:-${AGENTFORGE_MODEL:-}}"
: "${validation_model:?Set MODEL_PATH or AGENTFORGE_MODEL to the model to validate.}"

export RUN_NAME TASK_IDS_FILE EXPECTED_TASKS
export AGENTFORGE_MODEL="$validation_model"
export VLLM_MODEL="${VLLM_MODEL:-$validation_model}"
export RUNS_PER_TEMPERATURE=1
export TEMPERATURES=0.0
export TOTAL_SAMPLES=1
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export MAX_STEPS="${MAX_STEPS:-200}"
export TOP_P="${TOP_P:-1.0}"
export BASE_SEED="${BASE_SEED:-42}"

cat <<MSG
Cloud deterministic SWE-smith validation
  run:               $RUN_NAME
  model:             $AGENTFORGE_MODEL
  task IDs:          $TASK_IDS_FILE
  expected tasks:    $EXPECTED_TASKS
  trajectories/task: 1
  temperature:       0.0
  context length:    $CONTEXT_LENGTH
  maximum steps:     $MAX_STEPS
MSG

exec bash "$CLOUD_DIR/pipeline.sh" swesmith
