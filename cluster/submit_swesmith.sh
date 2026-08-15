#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${SWESMITH_MODE:-pilot}"
case "$MODE" in
  smoke)
    RUN_NAME="${RUN_NAME:-swesmith-smoke}"
    TASK_LIMIT="${TASK_LIMIT:-2}"
    EXPECTED_TASKS="${EXPECTED_TASKS:-$TASK_LIMIT}"
    NUM_SHARDS=1
    DEFAULT_ROLLOUT_WORKERS=2
    DEFAULT_EVAL_MAX_WORKERS=2
    COLLECTION_PBS="cluster/pbs/collect_swesmith_smoke.pbs"
    EVALUATION_PBS="cluster/pbs/evaluate_swesmith_smoke.pbs"
    ANALYSIS_PBS="cluster/pbs/analyze_swesmith.pbs"
    array_args=()
    ;;
  pilot)
    RUN_NAME="${RUN_NAME:-swesmith-pilot}"
    TASK_LIMIT="${TASK_LIMIT:-30}"
    EXPECTED_TASKS="${EXPECTED_TASKS:-$TASK_LIMIT}"
    NUM_SHARDS="${NUM_SHARDS:-3}"
    DEFAULT_ROLLOUT_WORKERS=5
    DEFAULT_EVAL_MAX_WORKERS=12
    COLLECTION_PBS="cluster/pbs/collect_swesmith_pilot.pbs"
    EVALUATION_PBS="cluster/pbs/evaluate_swesmith_pilot.pbs"
    ANALYSIS_PBS="cluster/pbs/analyze_swesmith.pbs"
    array_args=(-J "0-$((NUM_SHARDS - 1))")
    ;;
  full)
    RUN_NAME="${RUN_NAME:-swesmith-full}"
    TASK_LIMIT="${TASK_LIMIT:-}"
    if [[ -n "$TASK_LIMIT" ]]; then
      EXPECTED_TASKS="${EXPECTED_TASKS:-$TASK_LIMIT}"
    else
      EXPECTED_TASKS="${EXPECTED_TASKS:-50908}"
    fi
    NUM_SHARDS="${NUM_SHARDS:-50}"
    DEFAULT_ROLLOUT_WORKERS=6
    DEFAULT_EVAL_MAX_WORKERS=25
    COLLECTION_PBS="cluster/pbs/collect_swesmith_array.pbs"
    EVALUATION_PBS="cluster/pbs/evaluate_swesmith.pbs"
    ANALYSIS_PBS="cluster/pbs/analyze_swesmith_full.pbs"
    array_args=(-J "0-$((NUM_SHARDS - 1))")
    ;;
  *)
    echo "SWESMITH_MODE must be smoke, pilot, or full, got: $MODE" >&2
    exit 2
    ;;
esac

DATASET="${DATASET:-SWE-bench/SWE-smith-py}"
SWESMITH_DATASET_REVISION="${SWESMITH_DATASET_REVISION:-77cab9055d42ab4a5c25c89a8f937096db13558e}"
SPLIT="${SPLIT:-train}"
TASK_IDS_FILE="${TASK_IDS_FILE:-}"
RUNS_PER_TEMPERATURE="${RUNS_PER_TEMPERATURE:-4}"
if [[ ! "$RUNS_PER_TEMPERATURE" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUNS_PER_TEMPERATURE must be a positive integer." >&2
  exit 2
fi
TEMPERATURES="${TEMPERATURES:-0.6:0.7}"
IFS=":" read -r -a temperature_values <<<"$TEMPERATURES"
if ((${#temperature_values[@]} == 0)); then
  echo "TEMPERATURES must contain at least one value." >&2
  exit 2
fi
for temperature in "${temperature_values[@]}"; do
  if [[ ! "$temperature" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "TEMPERATURES must be a colon-separated list of non-negative numbers." >&2
    exit 2
  fi
done
TOTAL_SAMPLES=$((${#temperature_values[@]} * RUNS_PER_TEMPERATURE))
BASE_SEED="${BASE_SEED:-42}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-$DEFAULT_ROLLOUT_WORKERS}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-$DEFAULT_EVAL_MAX_WORKERS}"
AGENTFORGE_MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
MINI_SWE_MODEL="${MINI_SWE_MODEL:-hosted_vllm/$AGENTFORGE_MODEL}"
MINI_SWE_CONFIG="${MINI_SWE_CONFIG:-}"
MINI_SWE_RUNNER="${MINI_SWE_RUNNER:-singularity}"
MINI_SWE_ENVIRONMENT_CLASS="${MINI_SWE_ENVIRONMENT_CLASS:-singularity}"
MAX_STEPS="${MAX_STEPS:-200}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
TOP_P="${TOP_P:-1.0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-}"
SWESMITH_EVAL_RUNTIME="${SWESMITH_EVAL_RUNTIME:-apptainer}"
SUBMIT_EVAL="${SUBMIT_EVAL:-1}"
SUBMIT_ANALYSIS="${SUBMIT_ANALYSIS:-1}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_NAME may contain only letters, numbers, dots, underscores, and dashes." >&2
  exit 2
fi
source cluster/resolve_run_paths.sh
for value_name in NUM_SHARDS EXPECTED_TASKS RUNS_PER_TEMPERATURE TOTAL_SAMPLES; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer." >&2
    exit 2
  fi
done
if [[ -n "$TASK_LIMIT" && ! "$TASK_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "TASK_LIMIT must be empty or a positive integer." >&2
  exit 2
fi
if ((NUM_SHARDS > EXPECTED_TASKS)); then
  echo "NUM_SHARDS ($NUM_SHARDS) cannot exceed EXPECTED_TASKS ($EXPECTED_TASKS)." >&2
  exit 2
fi
if [[ "$MINI_SWE_RUNNER" == "pool_way" ]]; then
  echo "MINI_SWE_RUNNER=pool_way is unsupported for SWE-smith collection." >&2
  echo "Use swebench with Docker, or use the singularity runner." >&2
  exit 2
fi

collection_variables="SWESMITH_MODE=$MODE,RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,DATASET=$DATASET,SWESMITH_DATASET_REVISION=$SWESMITH_DATASET_REVISION,SPLIT=$SPLIT,EXPECTED_TASKS=$EXPECTED_TASKS,NUM_SHARDS=$NUM_SHARDS,RUNS_PER_TEMPERATURE=$RUNS_PER_TEMPERATURE,TEMPERATURES=$TEMPERATURES,TOTAL_SAMPLES=$TOTAL_SAMPLES,BASE_SEED=$BASE_SEED,ROLLOUT_WORKERS=$ROLLOUT_WORKERS,AGENTFORGE_MODEL=$AGENTFORGE_MODEL,MINI_SWE_MODEL=$MINI_SWE_MODEL,MINI_SWE_RUNNER=$MINI_SWE_RUNNER,MINI_SWE_ENVIRONMENT_CLASS=$MINI_SWE_ENVIRONMENT_CLASS,MAX_STEPS=$MAX_STEPS,CONTEXT_LENGTH=$CONTEXT_LENGTH,TOP_P=$TOP_P,TIMEOUT_SECONDS=$TIMEOUT_SECONDS,OVERWRITE=$OVERWRITE"
evaluation_variables="SWESMITH_MODE=$MODE,RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,DATASET=$DATASET,SWESMITH_DATASET_REVISION=$SWESMITH_DATASET_REVISION,SPLIT=$SPLIT,EXPECTED_TASKS=$EXPECTED_TASKS,NUM_SHARDS=$NUM_SHARDS,RUNS_PER_TEMPERATURE=$RUNS_PER_TEMPERATURE,TEMPERATURES=$TEMPERATURES,TOTAL_SAMPLES=$TOTAL_SAMPLES,EVAL_MAX_WORKERS=$EVAL_MAX_WORKERS,SWESMITH_EVAL_RUNTIME=$SWESMITH_EVAL_RUNTIME,OVERWRITE=$OVERWRITE"
analysis_variables="SWESMITH_MODE=$MODE,RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,EXPECTED_TASKS=$EXPECTED_TASKS,RUNS_PER_TEMPERATURE=$RUNS_PER_TEMPERATURE,TEMPERATURES=$TEMPERATURES,TOTAL_SAMPLES=$TOTAL_SAMPLES"
if [[ -n "$TASK_LIMIT" ]]; then
  collection_variables+=",LIMIT=$TASK_LIMIT"
fi
if [[ -n "$TASK_IDS_FILE" ]]; then
  collection_variables+=",TASK_IDS_FILE=$TASK_IDS_FILE"
fi
if [[ -n "$MINI_SWE_CONFIG" ]]; then
  collection_variables+=",MINI_SWE_CONFIG=$MINI_SWE_CONFIG"
fi
if [[ -n "$EVAL_TIMEOUT" ]]; then
  evaluation_variables+=",EVAL_TIMEOUT=$EVAL_TIMEOUT"
fi

print_section() {
  local title="$1"
  local pbs="$2"
  local detail="$3"
  local variables="${4//,/$'\n'}"
  printf '%s\n  PBS script: %s\n' "$title" "$pbs"
  if [[ -n "$detail" ]]; then
    printf '  %s\n' "$detail"
  fi
  printf '  Variables:\n'
  while IFS= read -r variable; do
    printf '    %s\n' "$variable"
  done <<<"$variables"
  printf '\n'
}

collection_dependency_args=()
collection_dependency_detail=""
if [[ -n "${AFTEROK_JOB_ID:-}" ]]; then
  collection_dependency_args=(-W "depend=afterok:$AFTEROK_JOB_ID")
  collection_dependency_detail="Dependency: successful job $AFTEROK_JOB_ID"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Dry run: no jobs submitted.\n\n'
  printf 'Cluster logs: %s\n\n' "$CLUSTER_LOG_DIR"
  detail=""
  if [[ "$MODE" != "smoke" ]]; then
    detail="Array indices: 0-$((NUM_SHARDS - 1))"
  fi
  if [[ -n "$collection_dependency_detail" ]]; then
    if [[ -n "$detail" ]]; then
      detail+="; $collection_dependency_detail"
    else
      detail="$collection_dependency_detail"
    fi
  fi
  print_section "SWE-smith collection job" "$COLLECTION_PBS" "$detail" "$collection_variables"
  if [[ "$SUBMIT_EVAL" == "1" ]]; then
    print_section "SWE-smith evaluation job" "$EVALUATION_PBS" "Dependency: successful collection" "$evaluation_variables"
    if [[ "$SUBMIT_ANALYSIS" == "1" ]]; then
      print_section "SWE-smith analysis job" "$ANALYSIS_PBS" "Dependency: successful evaluation" "$analysis_variables"
    fi
  fi
  exit 0
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub is not available. Run this script on the cluster login node." >&2
  exit 127
fi

mkdir -p "$CLUSTER_LOG_DIR"
qsub_log_args=(-o "$CLUSTER_LOG_DIR/" -e "$CLUSTER_LOG_DIR/")

collection_job="$(qsub -N "swesmith-$MODE-collect" "${qsub_log_args[@]}" "${array_args[@]}" "${collection_dependency_args[@]}" -v "$collection_variables" "$COLLECTION_PBS")"
echo "Submitted SWE-smith collection: $collection_job"
if [[ "$SUBMIT_EVAL" == "1" ]]; then
  evaluation_job="$(qsub -N "swesmith-$MODE-eval" "${qsub_log_args[@]}" -W "depend=afterok:$collection_job" -v "$evaluation_variables" "$EVALUATION_PBS")"
  echo "Submitted dependent SWE-smith evaluation: $evaluation_job"
  if [[ "$SUBMIT_ANALYSIS" == "1" ]]; then
    analysis_job="$(qsub -N "swesmith-$MODE-analysis" "${qsub_log_args[@]}" -W "depend=afterok:$evaluation_job" -v "$analysis_variables" "$ANALYSIS_PBS")"
    echo "Submitted dependent SWE-smith analysis: $analysis_job"
  fi
fi
