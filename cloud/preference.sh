#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$CLOUD_DIR/common.sh"

ACTION="${1:-all}"
case "$ACTION" in
  data|dmpo|depo|all|validate-data) ;;
  *)
    echo "Usage: bash cloud/preference.sh data|dmpo|depo|all|validate-data" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-swesmith-train-1000}"
require_run_name "$RUN_NAME"
RUN_ROOT="${RUN_ROOT:-$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME}"
export RUN_NAME RUN_ROOT

require_project_environment
cd "$DEBUG_DEPO_ROOT"

build_data() {
  local pids=()
  RUN_ROOT="$RUN_ROOT" scripts/build_dmpo_pairs.sh &
  pids+=("$!")
  RUN_ROOT="$RUN_ROOT" scripts/build_depo_data.sh &
  pids+=("$!")
  local failed=0
  local process_id
  for process_id in "${pids[@]}"; do
    if ! wait "$process_id"; then
      failed=1
    fi
  done
  if ((failed)); then
    echo "At least one preference-data builder failed." >&2
    return 1
  fi
  RUN_ROOT="$RUN_ROOT" scripts/validate_preference_data.sh dmpo
  RUN_ROOT="$RUN_ROOT" scripts/validate_preference_data.sh depo
}

validate_data() {
  RUN_ROOT="$RUN_ROOT" scripts/validate_preference_data.sh dmpo
  RUN_ROOT="$RUN_ROOT" scripts/validate_preference_data.sh depo
}

configure_training() {
  require_positive_integer NUM_PROCESSES "$NUM_PROCESSES"
  gpu_id_array
  if ((NUM_PROCESSES != NUM_SHARDS)); then
    echo "NUM_PROCESSES must equal the number of configured GPUs ($NUM_SHARDS), got: $NUM_PROCESSES." >&2
    return 2
  fi
  export CUDA_VISIBLE_DEVICES="${GPU_IDS// /,}"
  export NUM_PROCESSES
  export PACKAGE_MODEL="${PACKAGE_MODEL:-1}"
  export MAX_LENGTH="${MAX_LENGTH:-32768}"
}

train_dmpo() {
  configure_training
  export EXPERIMENT_ARM="${EXPERIMENT_ARM:-dmpo}"
  local stage_env=()
  if [[ -n "${DMPO_LEARNING_RATE:-}" ]]; then
    stage_env+=("LEARNING_RATE=$DMPO_LEARNING_RATE")
  fi
  if [[ -n "${DMPO_BETA:-}" ]]; then
    stage_env+=("BETA=$DMPO_BETA")
  fi
  if [[ -n "${DMPO_GAMMA:-}" ]]; then
    stage_env+=("GAMMA=$DMPO_GAMMA")
  fi
  env "${stage_env[@]}" scripts/train_dmpo.sh
}

train_depo() {
  configure_training
  export EXPERIMENT_ARM="${EXPERIMENT_ARM:-dmpo-depo}"
  local stage_env=()
  if [[ -n "${DEPO_LEARNING_RATE:-}" ]]; then
    stage_env+=("LEARNING_RATE=$DEPO_LEARNING_RATE")
  fi
  if [[ -n "${DEPO_BETA:-}" ]]; then
    stage_env+=("BETA=$DEPO_BETA")
  fi
  env "${stage_env[@]}" scripts/train_depo.sh
}

case "$ACTION" in
  data)
    build_data
    ;;
  validate-data)
    validate_data
    ;;
  dmpo)
    validate_data
    train_dmpo
    ;;
  depo)
    validate_data
    train_depo
    ;;
  all)
    build_data
    export EXPERIMENT_ARM="${EXPERIMENT_ARM:-dmpo-depo}"
    train_dmpo
    train_depo
    ;;
esac
