#!/usr/bin/env bash

# Resolve collision-free output paths for one preference-training experiment arm.
if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT must be set before sourcing scripts/preference_trial_paths.sh." >&2
  return 2 2>/dev/null || exit 2
fi

EXPERIMENT_ARM="${EXPERIMENT_ARM:-dmpo-depo}"
case "$EXPERIMENT_ARM" in
  dmpo|depo|dmpo-depo) ;;
  *)
    echo "EXPERIMENT_ARM must be dmpo, depo, or dmpo-depo." >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

DMPO_TRIAL_NAME="${DMPO_TRIAL_NAME:-default}"
DEPO_TRIAL_NAME="${DEPO_TRIAL_NAME:-default}"
for trial_name in "$DMPO_TRIAL_NAME" "$DEPO_TRIAL_NAME"; do
  if [[ ! "$trial_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Trial names may contain only letters, numbers, dots, underscores, and dashes: $trial_name" >&2
    return 2 2>/dev/null || exit 2
  fi
done

DMPO_TRIAL_ROOT="${DMPO_TRIAL_ROOT:-$RUN_ROOT/experiments/dmpo/$DMPO_TRIAL_NAME}"
case "$EXPERIMENT_ARM" in
  depo)
    default_depo_trial_root="$RUN_ROOT/experiments/depo/$DEPO_TRIAL_NAME"
    ;;
  *)
    default_depo_trial_root="$RUN_ROOT/experiments/dmpo-depo/$DMPO_TRIAL_NAME/depo/$DEPO_TRIAL_NAME"
    ;;
esac
DEPO_TRIAL_ROOT="${DEPO_TRIAL_ROOT:-$default_depo_trial_root}"
DMPO_TRAIN_OUTPUT_DIR="${DMPO_TRAIN_OUTPUT_DIR:-$DMPO_TRIAL_ROOT/training}"
DMPO_MODEL_DIR="${DMPO_MODEL_DIR:-$DMPO_TRIAL_ROOT/model}"
DEPO_TRAIN_OUTPUT_DIR="${DEPO_TRAIN_OUTPUT_DIR:-$DEPO_TRIAL_ROOT/training}"
DEPO_MODEL_DIR="${DEPO_MODEL_DIR:-$DEPO_TRIAL_ROOT/model}"

export EXPERIMENT_ARM DMPO_TRIAL_NAME DEPO_TRIAL_NAME
export DMPO_TRIAL_ROOT DEPO_TRIAL_ROOT
export DMPO_TRAIN_OUTPUT_DIR DMPO_MODEL_DIR
export DEPO_TRAIN_OUTPUT_DIR DEPO_MODEL_DIR
