#!/usr/bin/env bash
set -uo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-swesmith-train-1000-r2}"
AGENTFORGE_MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-100}"
failed_stages=()

run_stage() {
  local label="$1"
  shift
  echo "Starting $label"
  if "$@"; then
    echo "Completed $label"
  else
    failed_stages+=("$label")
    echo "Failed $label; continuing with the next stage." >&2
  fi
}

run_stage "1K trajectory pipeline" env \
RUN_NAME="$TRAIN_RUN_NAME" \
TASK_IDS_FILE=data/splits/swesmith_train_1000_instance_ids.txt \
EXPECTED_TASKS=1000 \
RUNS_PER_TEMPERATURE=2 \
TEMPERATURES=0.6:0.7 \
BASE_SEED=42 \
CONTEXT_LENGTH=65536 \
MAX_STEPS=200 \
TIMEOUT_SECONDS=21600 \
TOP_P=1.0 \
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}" \
EVAL_MAX_WORKERS="$EVAL_MAX_WORKERS" \
AGENTFORGE_MODEL="$AGENTFORGE_MODEL" \
  bash "$CLOUD_DIR/pipeline.sh" swesmith

for budget in 100 200 500; do
  run_stage "$budget-task SFT validation" env \
  RUN_NAME="$TRAIN_RUN_NAME-sft-validation-$budget" \
  TASK_IDS_FILE="data/splits/swesmith_validation_${budget}_instance_ids.txt" \
  AGENTFORGE_MODEL="$AGENTFORGE_MODEL" \
  CONTEXT_LENGTH=32768 \
  MAX_STEPS=200 \
  EVAL_MAX_WORKERS="$EVAL_MAX_WORKERS" \
    bash "$CLOUD_DIR/validate.sh"
done

if ((${#failed_stages[@]})); then
  echo "Trajectory suite finished with failed stages: ${failed_stages[*]}" >&2
  exit 1
fi

echo "Trajectory suite completed successfully."
