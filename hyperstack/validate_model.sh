#!/usr/bin/env bash
set -euo pipefail

HYPERSTACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HYPERSTACK_DIR/common.sh"

OBJECTIVE="${1:-}"
case "$OBJECTIVE" in
  dmpo|depo) ;;
  *)
    echo "Usage: bash hyperstack/validate_model.sh dmpo|depo" >&2
    exit 2
    ;;
esac

TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-swesmith-train-5000}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-$DEBUG_DEPO_SCRATCH/runs/$TRAIN_RUN_NAME}"
RUN_ROOT="$TRAIN_RUN_ROOT"
# shellcheck disable=SC1091
source "$DEBUG_DEPO_ROOT/scripts/preference_trial_paths.sh"

case "$OBJECTIVE" in
  dmpo)
    MODEL_PATH="${MODEL_PATH:-$DMPO_MODEL_DIR}"
    EVAL_RUN_NAME="${EVAL_RUN_NAME:-$TRAIN_RUN_NAME-dmpo-$DMPO_TRIAL_NAME-evaluation-500}"
    ;;
  depo)
    MODEL_PATH="${MODEL_PATH:-$DEPO_MODEL_DIR}"
    if [[ "$EXPERIMENT_ARM" == "depo" ]]; then
      EVAL_RUN_NAME="${EVAL_RUN_NAME:-$TRAIN_RUN_NAME-depo-$DEPO_TRIAL_NAME-evaluation-500}"
    else
      EVAL_RUN_NAME="${EVAL_RUN_NAME:-$TRAIN_RUN_NAME-dmpo-$DMPO_TRIAL_NAME-depo-$DEPO_TRIAL_NAME-evaluation-500}"
    fi
    ;;
esac

if [[ "${DRY_RUN:-0}" != "1" && ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Packaged model is missing: $MODEL_PATH" >&2
  exit 2
fi

export RUN_NAME="$EVAL_RUN_NAME"
unset RUN_ROOT
export AGENTFORGE_MODEL="$MODEL_PATH"
export VLLM_MODEL="$MODEL_PATH"
export EXPECTED_TASKS="${EXPECTED_TASKS:-500}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
bash "$HYPERSTACK_DIR/pipeline.sh" verified
