#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source scripts/preference_defaults.sh

OBJECTIVE="${PREFERENCE_OBJECTIVE:-}"
case "$OBJECTIVE" in
  dmpo) OBJECTIVE_LABEL=DMPO ;;
  depo) OBJECTIVE_LABEL=DEPO ;;
  *)
    echo "PREFERENCE_OBJECTIVE must be dmpo or depo." >&2
    exit 2
    ;;
esac

TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-swesmith-train}"
if [[ ! "$TRAIN_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "TRAIN_RUN_NAME may contain only letters, numbers, dots, underscores, and dashes." >&2
  exit 2
fi

if [[ -z "${TRAIN_RUN_ROOT:-}" ]]; then
  saved_run_name="${RUN_NAME:-}"
  RUN_NAME="$TRAIN_RUN_NAME"
  source cluster/resolve_run_paths.sh
  TRAIN_RUN_ROOT="$RUN_ROOT"
  RUN_NAME="$saved_run_name"
  unset RUN_ROOT
fi

saved_run_root="${RUN_ROOT:-}"
RUN_ROOT="$TRAIN_RUN_ROOT"
source scripts/preference_trial_paths.sh
if [[ -n "$saved_run_root" ]]; then
  RUN_ROOT="$saved_run_root"
else
  unset RUN_ROOT
fi
unset saved_run_root

case "$OBJECTIVE" in
  dmpo)
    if [[ "$EXPERIMENT_ARM" != "dmpo" && "$EXPERIMENT_ARM" != "dmpo-depo" ]]; then
      echo "DMPO evaluation requires EXPERIMENT_ARM=dmpo or dmpo-depo." >&2
      exit 2
    fi
    MODEL_PATH="${MODEL_PATH:-$DMPO_MODEL_DIR}"
    EVAL_RUN_NAME="${EVAL_RUN_NAME:-$TRAIN_RUN_NAME-dmpo-$DMPO_TRIAL_NAME-evaluation-500}"
    ;;
  depo)
    if [[ "$EXPERIMENT_ARM" != "depo" && "$EXPERIMENT_ARM" != "dmpo-depo" ]]; then
      echo "DEPO evaluation requires EXPERIMENT_ARM=depo or dmpo-depo." >&2
      exit 2
    fi
    MODEL_PATH="${MODEL_PATH:-$DEPO_MODEL_DIR}"
    if [[ "$EXPERIMENT_ARM" == "depo" ]]; then
      EVAL_RUN_NAME="${EVAL_RUN_NAME:-$TRAIN_RUN_NAME-depo-$DEPO_TRIAL_NAME-evaluation-500}"
    else
      EVAL_RUN_NAME="${EVAL_RUN_NAME:-$TRAIN_RUN_NAME-dmpo-$DMPO_TRIAL_NAME-depo-$DEPO_TRIAL_NAME-evaluation-500}"
    fi
    ;;
esac
TASK_IDS_FILE="${TASK_IDS_FILE:-}"

if [[ "${DRY_RUN:-0}" != "1" && ! -f "$MODEL_PATH/config.json" && -z "${AFTEROK_JOB_ID:-}" ]]; then
  echo "Packaged $OBJECTIVE model is missing: $MODEL_PATH" >&2
  echo "Set AFTEROK_JOB_ID when submitting before its package job completes." >&2
  exit 2
fi
if [[ -n "$TASK_IDS_FILE" && ! -s "$TASK_IDS_FILE" ]]; then
  echo "Evaluation task split is missing or empty: $TASK_IDS_FILE" >&2
  exit 2
fi

env_args=(
  "RUN_NAME=$EVAL_RUN_NAME"
  "AGENTFORGE_MODEL=$MODEL_PATH"
  "EXPECTED_COUNT=${EXPECTED_COUNT:-500}"
  "NUM_SHARDS=${EVAL_NUM_SHARDS:-10}"
  "ROLLOUT_WORKERS=${EVAL_ROLLOUT_WORKERS:-6}"
  "CONTEXT_LENGTH=${EVAL_CONTEXT_LENGTH:-$PREFERENCE_EVAL_CONTEXT_LENGTH_DEFAULT}"
  "TEMPERATURE=${EVAL_TEMPERATURE:-0.0}"
  "SUBMIT_EVAL=1"
  "SUBMIT_ANALYSIS=1"
  "COLLECTION_STAGE_LABEL=$OBJECTIVE_LABEL rollout generation job"
  "EVALUATION_STAGE_LABEL=$OBJECTIVE_LABEL evaluation job"
  "ANALYSIS_STAGE_LABEL=$OBJECTIVE_LABEL analysis job"
)
if [[ -n "$TASK_IDS_FILE" ]]; then
  env_args+=("TASK_IDS_FILE=$TASK_IDS_FILE")
fi
if [[ -n "${AFTEROK_JOB_ID:-}" ]]; then
  env_args+=("AFTEROK_JOB_ID=$AFTEROK_JOB_ID")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  env_args+=("DRY_RUN=1")
fi

env -u RUN_ROOT "${env_args[@]}" cluster/submit_verified_full.sh
