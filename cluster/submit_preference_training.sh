#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-swesmith-pilot}"
source cluster/resolve_run_paths.sh
source scripts/preference_defaults.sh
source scripts/preference_trial_paths.sh

case "$EXPERIMENT_ARM" in
  dmpo) NEED_DMPO=1; NEED_DEPO=0 ;;
  depo) NEED_DMPO=0; NEED_DEPO=1 ;;
  dmpo-depo) NEED_DMPO=1; NEED_DEPO=1 ;;
esac

BASE_MODEL="${BASE_MODEL:-$PREFERENCE_BASE_MODEL_DEFAULT}"
NUM_PROCESSES="${NUM_PROCESSES:-$PREFERENCE_NUM_PROCESSES_DEFAULT}"
TRAIN_NCPUS="${TRAIN_NCPUS:-8}"
TRAIN_MEMORY="${TRAIN_MEMORY:-64gb}"
if [[ ! "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_PROCESSES must be a positive integer." >&2
  exit 2
fi
MAX_LENGTH="${MAX_LENGTH:-$PREFERENCE_TRAIN_MAX_LENGTH_DEFAULT}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"
PREFERENCE_MAX_ROLLOUTS="${PREFERENCE_MAX_ROLLOUTS:-$PREFERENCE_MAX_ROLLOUTS_DEFAULT}"
PREFERENCE_SAMPLE_INDICES="${PREFERENCE_SAMPLE_INDICES:-}"
PREFERENCE_SAMPLE_INDICES="${PREFERENCE_SAMPLE_INDICES//,/:}"
if [[ -n "$PREFERENCE_SAMPLE_INDICES" && ! "$PREFERENCE_SAMPLE_INDICES" =~ ^[0-9]+(:[0-9]+)*$ ]]; then
  echo "PREFERENCE_SAMPLE_INDICES must contain non-negative indices separated by ':' or ','." >&2
  exit 2
fi

TOKEN_METRIC="${TOKEN_METRIC:-total_tokens}"
MIN_COST_RATIO="${MIN_COST_RATIO:-1.1}"
MAX_PAIRS_PER_TASK="${MAX_PAIRS_PER_TASK:-0}"
INCLUDE_FAILURE_EFFICIENCY_PAIRS="${INCLUDE_FAILURE_EFFICIENCY_PAIRS:-0}"
PREFERENCE_DATA_MODE="${PREFERENCE_DATA_MODE:-reuse}"
DMPO_MODE="${DMPO_MODE:-train}"
for setting in \
  "PREFERENCE_DATA_MODE:$PREFERENCE_DATA_MODE:build:reuse" \
  "DMPO_MODE:$DMPO_MODE:train:reuse"; do
  IFS=: read -r name value first second <<<"$setting"
  if [[ "$value" != "$first" && "$value" != "$second" ]]; then
    echo "$name must be $first or $second." >&2
    exit 2
  fi
done

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-$PREFERENCE_PER_DEVICE_BATCH_SIZE_DEFAULT}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$PREFERENCE_GRADIENT_ACCUMULATION_STEPS_DEFAULT}"
EPOCHS="${EPOCHS:-$PREFERENCE_EPOCHS_DEFAULT}"
SAVE_STEPS="${SAVE_STEPS:-$PREFERENCE_SAVE_STEPS_DEFAULT}"
DMPO_LEARNING_RATE="${DMPO_LEARNING_RATE:-$PREFERENCE_DMPO_LEARNING_RATE_DEFAULT}"
DMPO_BETA="${DMPO_BETA:-$PREFERENCE_DMPO_BETA_DEFAULT}"
DMPO_GAMMA="${DMPO_GAMMA:-$PREFERENCE_DMPO_GAMMA_DEFAULT}"
DEPO_LEARNING_RATE="${DEPO_LEARNING_RATE:-$PREFERENCE_DEPO_LEARNING_RATE_DEFAULT}"
DEPO_BETA="${DEPO_BETA:-$PREFERENCE_DEPO_BETA_DEFAULT}"
ALPHA_TOKENS="${ALPHA_TOKENS:-$PREFERENCE_ALPHA_TOKENS_DEFAULT}"
ALPHA_STEPS="${ALPHA_STEPS:-$PREFERENCE_ALPHA_STEPS_DEFAULT}"
DEPO_TOKEN_METRIC="${DEPO_TOKEN_METRIC:-$PREFERENCE_TOKEN_METRIC_DEFAULT}"

SUBMIT_MODEL_EVALUATIONS="${SUBMIT_MODEL_EVALUATIONS:-1}"
if [[ "$EXPERIMENT_ARM" == "dmpo" || "$DMPO_MODE" == "train" ]]; then
  SUBMIT_DMPO_EVALUATION="${SUBMIT_DMPO_EVALUATION:-1}"
else
  SUBMIT_DMPO_EVALUATION="${SUBMIT_DMPO_EVALUATION:-0}"
fi
for setting in \
  "SUBMIT_MODEL_EVALUATIONS:$SUBMIT_MODEL_EVALUATIONS" \
  "SUBMIT_DMPO_EVALUATION:$SUBMIT_DMPO_EVALUATION"; do
  name="${setting%%:*}"
  value="${setting#*:}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "$name must be 0 or 1." >&2
    exit 2
  fi
done

EVAL_TASK_IDS_FILE="${EVAL_TASK_IDS_FILE:-}"
EVAL_EXPECTED_TASKS="${EVAL_EXPECTED_TASKS:-500}"
EVAL_NUM_SHARDS="${EVAL_NUM_SHARDS:-10}"
EVAL_ROLLOUT_WORKERS="${EVAL_ROLLOUT_WORKERS:-6}"
EVAL_CONTEXT_LENGTH="${EVAL_CONTEXT_LENGTH:-$PREFERENCE_EVAL_CONTEXT_LENGTH_DEFAULT}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
DMPO_EVAL_RUN_NAME="${DMPO_EVAL_RUN_NAME:-$RUN_NAME-dmpo-$DMPO_TRIAL_NAME-evaluation-500}"
if [[ "$EXPERIMENT_ARM" == "depo" ]]; then
  DEPO_BASE_MODEL="$BASE_MODEL"
  DEPO_EVAL_RUN_NAME="${DEPO_EVAL_RUN_NAME:-$RUN_NAME-depo-$DEPO_TRIAL_NAME-evaluation-500}"
else
  DEPO_BASE_MODEL="$DMPO_MODEL_DIR"
  DEPO_EVAL_RUN_NAME="${DEPO_EVAL_RUN_NAME:-$RUN_NAME-dmpo-$DMPO_TRIAL_NAME-depo-$DEPO_TRIAL_NAME-evaluation-500}"
fi

data_variables="RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,TOKEN_METRIC=$TOKEN_METRIC,MIN_COST_RATIO=$MIN_COST_RATIO,MAX_PAIRS_PER_TASK=$MAX_PAIRS_PER_TASK,INCLUDE_FAILURE_EFFICIENCY_PAIRS=$INCLUDE_FAILURE_EFFICIENCY_PAIRS,PREFERENCE_MAX_ROLLOUTS=$PREFERENCE_MAX_ROLLOUTS"
if [[ -n "$PREFERENCE_SAMPLE_INDICES" ]]; then
  data_variables+=",PREFERENCE_SAMPLE_INDICES=$PREFERENCE_SAMPLE_INDICES"
fi
trial_variables="EXPERIMENT_ARM=$EXPERIMENT_ARM,DMPO_TRIAL_NAME=$DMPO_TRIAL_NAME,DEPO_TRIAL_NAME=$DEPO_TRIAL_NAME,DMPO_TRAIN_OUTPUT_DIR=$DMPO_TRAIN_OUTPUT_DIR,DMPO_MODEL_DIR=$DMPO_MODEL_DIR,DEPO_TRAIN_OUTPUT_DIR=$DEPO_TRAIN_OUTPUT_DIR,DEPO_MODEL_DIR=$DEPO_MODEL_DIR"
dmpo_variables="PREFERENCE_OBJECTIVE=dmpo,$trial_variables,RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,MODEL_NAME_OR_PATH=$BASE_MODEL,NUM_PROCESSES=$NUM_PROCESSES,MAX_LENGTH=$MAX_LENGTH,MAX_TRAIN_ROWS=$MAX_TRAIN_ROWS,PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE,GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS,EPOCHS=$EPOCHS,LEARNING_RATE=$DMPO_LEARNING_RATE,BETA=$DMPO_BETA,GAMMA=$DMPO_GAMMA,SAVE_STEPS=$SAVE_STEPS,PACKAGE_MODEL=0"
depo_variables="PREFERENCE_OBJECTIVE=depo,$trial_variables,RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,MODEL_NAME_OR_PATH=$DEPO_BASE_MODEL,NUM_PROCESSES=$NUM_PROCESSES,MAX_LENGTH=$MAX_LENGTH,MAX_TRAIN_ROWS=$MAX_TRAIN_ROWS,PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE,GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS,EPOCHS=$EPOCHS,LEARNING_RATE=$DEPO_LEARNING_RATE,BETA=$DEPO_BETA,ALPHA_TOKENS=$ALPHA_TOKENS,ALPHA_STEPS=$ALPHA_STEPS,DEPO_TOKEN_METRIC=$DEPO_TOKEN_METRIC,SAVE_STEPS=$SAVE_STEPS,PACKAGE_MODEL=0"
dmpo_package_variables="PREFERENCE_OBJECTIVE=dmpo,$trial_variables,RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,BASE_MODEL=$BASE_MODEL"
depo_package_variables="PREFERENCE_OBJECTIVE=depo,$trial_variables,RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,BASE_MODEL=$DEPO_BASE_MODEL"

initial_dependency=()
initial_description=none
if [[ -n "${AFTEROK_JOB_ID:-}" ]]; then
  initial_dependency=(-W "depend=afterok:$AFTEROK_JOB_ID")
  initial_description="$AFTEROK_JOB_ID"
fi

if [[ "$PREFERENCE_DATA_MODE" == "reuse" && "${DRY_RUN:-0}" != "1" ]]; then
  if ((NEED_DMPO)); then
    scripts/validate_preference_data.sh dmpo
  fi
  if ((NEED_DEPO)); then
    scripts/validate_preference_data.sh depo
  fi
fi
if ((NEED_DMPO)) && [[ "$DMPO_MODE" == "reuse" && "${DRY_RUN:-0}" != "1" ]] && \
  [[ ! -f "$DMPO_MODEL_DIR/config.json" || ! -f "$DMPO_MODEL_DIR/package_manifest.json" ]]; then
  echo "Cannot reuse missing packaged DMPO model: $DMPO_MODEL_DIR" >&2
  exit 2
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<MSG
Dry run: no jobs submitted.
Experiment arm: $EXPERIMENT_ARM
Model lineage: $BASE_MODEL$(if [[ "$EXPERIMENT_ARM" == "dmpo" ]]; then printf ' -> DMPO'; elif [[ "$EXPERIMENT_ARM" == "depo" ]]; then printf ' -> DEPO'; else printf ' -> DMPO -> DEPO'; fi)
Run root: $RUN_ROOT
DMPO trial: $DMPO_TRIAL_NAME
DEPO trial: $DEPO_TRIAL_NAME
Preference data mode: $PREFERENCE_DATA_MODE
DMPO mode: $DMPO_MODE
Rollouts per task: $PREFERENCE_MAX_ROLLOUTS (temperature-balanced)
Maximum training rows: ${MAX_TRAIN_ROWS:-all}
Explicit sample indices: ${PREFERENCE_SAMPLE_INDICES:-automatic}
Initial dependency: $initial_description
Training resources: 1 node, $TRAIN_NCPUS CPUs, $NUM_PROCESSES GPUs, $TRAIN_MEMORY memory
MSG
  step=1
  if [[ "$PREFERENCE_DATA_MODE" == "build" ]]; then
    if ((NEED_DMPO)); then
      echo "$step. DMPO data: qsub -v $data_variables cluster/pbs/build_dmpo_pairs.pbs"
      ((step += 1))
    fi
    if ((NEED_DEPO)); then
      echo "$step. DEPO data: qsub -v $data_variables cluster/pbs/build_depo_data.pbs"
      ((step += 1))
    fi
  else
    echo "$step. Preference data: reuse existing $(if ((NEED_DMPO && NEED_DEPO)); then printf 'DMPO and DEPO'; elif ((NEED_DMPO)); then printf 'DMPO'; else printf 'DEPO'; fi) files"
    ((step += 1))
  fi
  if ((NEED_DMPO)); then
    if [[ "$DMPO_MODE" == "train" ]]; then
      echo "$step. DMPO train: qsub -W depend=afterok:<data-jobs> -v $dmpo_variables cluster/pbs/train_preference.pbs"
      ((step += 1))
      echo "$step. DMPO package: qsub -W depend=afterok:<dmpo-train> -v $dmpo_package_variables cluster/pbs/package_preference.pbs"
      ((step += 1))
    else
      echo "$step. DMPO train/package: reuse $DMPO_MODEL_DIR"
      ((step += 1))
    fi
  fi
  if ((NEED_DEPO)); then
    echo "$step. DEPO train: qsub -W depend=afterok:<parent-and-data> -v $depo_variables cluster/pbs/train_preference.pbs"
    ((step += 1))
    echo "$step. DEPO package: qsub -W depend=afterok:<depo-train> -v $depo_package_variables cluster/pbs/package_preference.pbs"
  fi
  cat <<MSG

Evaluations (${EVAL_EXPECTED_TASKS} expected tasks, ${EVAL_NUM_SHARDS} shards, temperature ${EVAL_TEMPERATURE}):
  DMPO: $(if ((NEED_DMPO)); then printf '%s (enabled: %s)' "$DMPO_EVAL_RUN_NAME" "$SUBMIT_DMPO_EVALUATION"; else printf 'not part of this arm'; fi)
  DEPO: $(if ((NEED_DEPO)); then printf '%s' "$DEPO_EVAL_RUN_NAME"; else printf 'not part of this arm'; fi)
  Task IDs: ${EVAL_TASK_IDS_FILE:-existing SWE-bench Verified 500-task evaluation split}
  Enabled: $SUBMIT_MODEL_EVALUATIONS

Packaged models:
  DMPO: $(if ((NEED_DMPO)); then printf '%s' "$DMPO_MODEL_DIR"; else printf 'not produced'; fi)
  DEPO: $(if ((NEED_DEPO)); then printf '%s' "$DEPO_MODEL_DIR"; else printf 'not produced'; fi)
MSG
  exit 0
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub is not available. Run this script on the cluster login node." >&2
  exit 127
fi
mkdir -p "$CLUSTER_LOG_DIR"
log_args=(-o "$CLUSTER_LOG_DIR/" -e "$CLUSTER_LOG_DIR/")
train_resource_args=(-l "select=1:ncpus=$TRAIN_NCPUS:ngpus=$NUM_PROCESSES:mem=$TRAIN_MEMORY")

data_job_ids=()
if [[ "$PREFERENCE_DATA_MODE" == "build" ]]; then
  if ((NEED_DMPO)); then
    dmpo_data_job="$(qsub -N "dmpo-data-$RUN_NAME" "${log_args[@]}" "${initial_dependency[@]}" -v "$data_variables" cluster/pbs/build_dmpo_pairs.pbs)"
    data_job_ids+=("$dmpo_data_job")
    echo "Submitted DMPO data: $dmpo_data_job"
  fi
  if ((NEED_DEPO)); then
    depo_data_job="$(qsub -N "depo-data-$RUN_NAME" "${log_args[@]}" "${initial_dependency[@]}" -v "$data_variables" cluster/pbs/build_depo_data.pbs)"
    data_job_ids+=("$depo_data_job")
    echo "Submitted DEPO data: $depo_data_job"
  fi
else
  echo "Reusing existing preference data under $RUN_ROOT/preference-data"
fi

stage_dependency=("${initial_dependency[@]}")
if ((${#data_job_ids[@]})); then
  data_dependency="$(IFS=:; echo "${data_job_ids[*]}")"
  stage_dependency=(-W "depend=afterok:$data_dependency")
fi

dmpo_package_job=""
if ((NEED_DMPO)); then
  if [[ "$DMPO_MODE" == "train" ]]; then
    dmpo_train_job="$(qsub -N "dmpo-$DMPO_TRIAL_NAME" "${log_args[@]}" "${train_resource_args[@]}" "${stage_dependency[@]}" -v "$dmpo_variables" cluster/pbs/train_preference.pbs)"
    echo "Submitted DMPO training: $dmpo_train_job"
    dmpo_package_job="$(qsub -N "dmpo-pkg-$DMPO_TRIAL_NAME" "${log_args[@]}" -W "depend=afterok:$dmpo_train_job" -v "$dmpo_package_variables" cluster/pbs/package_preference.pbs)"
    echo "Submitted DMPO CPU packaging: $dmpo_package_job"
  else
    echo "Reusing packaged DMPO model: $DMPO_MODEL_DIR"
  fi
fi

depo_package_job=""
if ((NEED_DEPO)); then
  depo_dependency=("${stage_dependency[@]}")
  if [[ "$EXPERIMENT_ARM" == "dmpo-depo" && -n "$dmpo_package_job" ]]; then
    depo_dependency=(-W "depend=afterok:$dmpo_package_job")
  fi
  depo_train_job="$(qsub -N "depo-$DEPO_TRIAL_NAME" "${log_args[@]}" "${train_resource_args[@]}" "${depo_dependency[@]}" -v "$depo_variables" cluster/pbs/train_preference.pbs)"
  echo "Submitted DEPO training: $depo_train_job"
  depo_package_job="$(qsub -N "depo-pkg-$DEPO_TRIAL_NAME" "${log_args[@]}" -W "depend=afterok:$depo_train_job" -v "$depo_package_variables" cluster/pbs/package_preference.pbs)"
  echo "Submitted DEPO CPU packaging: $depo_package_job"
fi

submit_evaluation() {
  local objective="$1"
  local model_path="$2"
  local eval_run_name="$3"
  local dependency="$4"
  local eval_env=(
    "PREFERENCE_OBJECTIVE=$objective"
    "EXPERIMENT_ARM=$EXPERIMENT_ARM"
    "TRAIN_RUN_NAME=$RUN_NAME"
    "TRAIN_RUN_ROOT=$RUN_ROOT"
    "DMPO_TRIAL_NAME=$DMPO_TRIAL_NAME"
    "DEPO_TRIAL_NAME=$DEPO_TRIAL_NAME"
    "MODEL_PATH=$model_path"
    "EVAL_RUN_NAME=$eval_run_name"
    "TASK_IDS_FILE=$EVAL_TASK_IDS_FILE"
    "EXPECTED_COUNT=$EVAL_EXPECTED_TASKS"
    "EVAL_NUM_SHARDS=$EVAL_NUM_SHARDS"
    "EVAL_ROLLOUT_WORKERS=$EVAL_ROLLOUT_WORKERS"
    "EVAL_CONTEXT_LENGTH=$EVAL_CONTEXT_LENGTH"
    "EVAL_TEMPERATURE=$EVAL_TEMPERATURE"
  )
  if [[ -n "$dependency" ]]; then
    eval_env+=("AFTEROK_JOB_ID=$dependency")
  fi
  env "${eval_env[@]}" cluster/submit_preference_evaluation.sh
}

if [[ "$SUBMIT_MODEL_EVALUATIONS" == "1" ]]; then
  if ((NEED_DMPO)) && [[ "$SUBMIT_DMPO_EVALUATION" == "1" ]]; then
    submit_evaluation dmpo "$DMPO_MODEL_DIR" "$DMPO_EVAL_RUN_NAME" "$dmpo_package_job"
  fi
  if ((NEED_DEPO)); then
    submit_evaluation depo "$DEPO_MODEL_DIR" "$DEPO_EVAL_RUN_NAME" "$depo_package_job"
  fi
fi
