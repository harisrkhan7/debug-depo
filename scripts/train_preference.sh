#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/preference_defaults.sh"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "Set RUN_ROOT to the evaluated SWE-smith run directory." >&2
  exit 2
fi
source "$ROOT_DIR/scripts/preference_trial_paths.sh"

OBJECTIVE="${PREFERENCE_OBJECTIVE:-}"
case "$OBJECTIVE" in
  dmpo)
    OBJECTIVE_LABEL=DMPO
    MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$PREFERENCE_BASE_MODEL_DEFAULT}"
    DATA_PATH="${DMPO_DATA_PATH:-$RUN_ROOT/preference-data/dmpo/pairs.jsonl}"
    TRAIN_OUTPUT_DIR="$DMPO_TRAIN_OUTPUT_DIR"
    PACKAGE_OUTPUT_DIR="$DMPO_MODEL_DIR"
    LEARNING_RATE="${LEARNING_RATE:-$PREFERENCE_DMPO_LEARNING_RATE_DEFAULT}"
    BETA="${BETA:-$PREFERENCE_DMPO_BETA_DEFAULT}"
    GAMMA="${GAMMA:-$PREFERENCE_DMPO_GAMMA_DEFAULT}"
    ;;
  depo)
    OBJECTIVE_LABEL=DEPO
    if [[ "$EXPERIMENT_ARM" == "depo" ]]; then
      default_depo_base="$PREFERENCE_BASE_MODEL_DEFAULT"
    elif [[ "$EXPERIMENT_ARM" == "dmpo-depo" ]]; then
      default_depo_base="$DMPO_MODEL_DIR"
    else
      echo "DEPO training requires EXPERIMENT_ARM=depo or dmpo-depo." >&2
      exit 2
    fi
    MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$default_depo_base}"
    DATA_PATH="${DEPO_DATA_PATH:-$RUN_ROOT/preference-data/depo/trajectories.jsonl}"
    TRAIN_OUTPUT_DIR="$DEPO_TRAIN_OUTPUT_DIR"
    PACKAGE_OUTPUT_DIR="$DEPO_MODEL_DIR"
    LEARNING_RATE="${LEARNING_RATE:-$PREFERENCE_DEPO_LEARNING_RATE_DEFAULT}"
    BETA="${BETA:-$PREFERENCE_DEPO_BETA_DEFAULT}"
    ALPHA_TOKENS="${ALPHA_TOKENS:-$PREFERENCE_ALPHA_TOKENS_DEFAULT}"
    ALPHA_STEPS="${ALPHA_STEPS:-$PREFERENCE_ALPHA_STEPS_DEFAULT}"
    DEPO_TOKEN_METRIC="${DEPO_TOKEN_METRIC:-$PREFERENCE_TOKEN_METRIC_DEFAULT}"
    ;;
  *)
    echo "PREFERENCE_OBJECTIVE must be dmpo or depo." >&2
    exit 2
    ;;
esac

NUM_PROCESSES="${NUM_PROCESSES:-${PBS_NGPUS:-$PREFERENCE_NUM_PROCESSES_DEFAULT}}"
MAX_LENGTH="${MAX_LENGTH:-$PREFERENCE_TRAIN_MAX_LENGTH_DEFAULT}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-$PREFERENCE_PER_DEVICE_BATCH_SIZE_DEFAULT}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$PREFERENCE_GRADIENT_ACCUMULATION_STEPS_DEFAULT}"
EPOCHS="${EPOCHS:-$PREFERENCE_EPOCHS_DEFAULT}"
SAVE_STEPS="${SAVE_STEPS:-$PREFERENCE_SAVE_STEPS_DEFAULT}"
if [[ ! "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_PROCESSES must be a positive integer." >&2
  exit 2
fi

if [[ ! -s "$DATA_PATH" ]]; then
  echo "$OBJECTIVE_LABEL training data is missing or empty: $DATA_PATH" >&2
  echo "Run the preference-data builders first." >&2
  exit 2
fi
if [[ "$OBJECTIVE" == "depo" && "$EXPERIMENT_ARM" == "dmpo-depo" && "${VALIDATE_ONLY:-0}" != "1" && ! -f "$MODEL_NAME_OR_PATH/config.json" ]]; then
  echo "Packaged DMPO initialization is missing: $MODEL_NAME_OR_PATH" >&2
  echo "Run DMPO training/package first or set MODEL_NAME_OR_PATH explicitly." >&2
  exit 2
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
args=(
  --model-name-or-path "$MODEL_NAME_OR_PATH"
  --data-path "$DATA_PATH"
  --output-dir "$TRAIN_OUTPUT_DIR"
  --max-rows "$MAX_TRAIN_ROWS"
  --max-length "$MAX_LENGTH"
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --epochs "$EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --beta "$BETA"
  --save-steps "$SAVE_STEPS"
  --expected-num-processes "$NUM_PROCESSES"
)
if [[ "$OBJECTIVE" == "dmpo" ]]; then
  args+=(--gamma "$GAMMA")
else
  args+=(
    --alpha-tokens "$ALPHA_TOKENS"
    --alpha-steps "$ALPHA_STEPS"
    --token-metric "$DEPO_TOKEN_METRIC"
  )
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  args+=(--trust-remote-code)
fi
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  "$UV_BIN" run "debug-depo-train-$OBJECTIVE" "${args[@]}" --validate-only "$@"
  exit 0
fi

accelerate_args=(launch --num_processes "$NUM_PROCESSES")
if ((NUM_PROCESSES > 1)); then
  accelerate_args+=(--multi_gpu)
fi
"$UV_BIN" run accelerate "${accelerate_args[@]}" \
  -m debug_depo.preference_training --objective "$OBJECTIVE" "${args[@]}" "$@"

if [[ "${PACKAGE_MODEL:-1}" == "1" ]]; then
  if [[ -f "$PACKAGE_OUTPUT_DIR/package_manifest.json" && -f "$PACKAGE_OUTPUT_DIR/config.json" ]]; then
    echo "Reusing complete $OBJECTIVE_LABEL package: $PACKAGE_OUTPUT_DIR"
  elif [[ -d "$PACKAGE_OUTPUT_DIR" ]] && find "$PACKAGE_OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty $OBJECTIVE_LABEL package: $PACKAGE_OUTPUT_DIR" >&2
    exit 2
  else
    package_args=(
      --base-model "$MODEL_NAME_OR_PATH"
      --adapter-path "$TRAIN_OUTPUT_DIR/adapter"
      --output-dir "$PACKAGE_OUTPUT_DIR"
    )
    if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
      package_args+=(--trust-remote-code)
    fi
    "$UV_BIN" run debug-depo-package-model "${package_args[@]}"
  fi
fi

cat <<MSG
$OBJECTIVE_LABEL training complete.
Adapter: $TRAIN_OUTPUT_DIR/adapter
Packaged evaluation model: $PACKAGE_OUTPUT_DIR
MSG
