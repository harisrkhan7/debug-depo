#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

print_dry_run_job() {
  local title="$1"
  local pbs_script="$2"
  local detail="$3"
  local variables="${4//,/$'\n'}"
  local variable

  printf '%s\n' "$title"
  printf '  PBS script: %s\n' "$pbs_script"
  if [[ -n "$detail" ]]; then
    printf '  %s\n' "$detail"
  fi
  printf '  Variables:\n'
  while IFS= read -r variable; do
    printf '    %s\n' "$variable"
  done <<<"$variables"
  printf '\n'
}

RUN_NAME="${RUN_NAME:-agentforge-verified-smoke}"
RUN_ID="${RUN_ID:-${RUN_NAME//-/_}}"
DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
SWEBENCH_DATASET_REVISION="${SWEBENCH_DATASET_REVISION:-}"
if [[ -z "$SWEBENCH_DATASET_REVISION" && "$DATASET" == "princeton-nlp/SWE-bench_Verified" ]]; then
  SWEBENCH_DATASET_REVISION="c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
fi
SPLIT="${SPLIT:-test}"
TASK_IDS_FILE="${TASK_IDS_FILE:-}"
SMOKE_LIMIT="${SMOKE_LIMIT:-5}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-2}"
AGENTFORGE_MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
HARNESS="${HARNESS:-mini-swe-agent-plus}"
MINI_SWE_RUNNER="${MINI_SWE_RUNNER:-singularity}"
MINI_SWE_ENVIRONMENT_CLASS="${MINI_SWE_ENVIRONMENT_CLASS:-singularity}"
MAX_STEPS="${MAX_STEPS:-200}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"
OVERWRITE="${OVERWRITE:-0}"
SUBMIT_EVAL="${SUBMIT_EVAL:-1}"
SUBMIT_ANALYSIS="${SUBMIT_ANALYSIS:-1}"
EXPECTED_COUNT="${EXPECTED_COUNT:-$SMOKE_LIMIT}"

if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_NAME may contain only letters, numbers, dots, underscores, and dashes." >&2
  exit 2
fi
source cluster/resolve_run_paths.sh

collection_variables="RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,DATASET=$DATASET,SPLIT=$SPLIT,SMOKE_LIMIT=$SMOKE_LIMIT,ROLLOUT_WORKERS=$ROLLOUT_WORKERS,AGENTFORGE_MODEL=$AGENTFORGE_MODEL,HARNESS=$HARNESS,MINI_SWE_RUNNER=$MINI_SWE_RUNNER,MINI_SWE_ENVIRONMENT_CLASS=$MINI_SWE_ENVIRONMENT_CLASS,MAX_STEPS=$MAX_STEPS,CONTEXT_LENGTH=$CONTEXT_LENGTH,TEMPERATURE=$TEMPERATURE,TOP_P=$TOP_P,TIMEOUT_SECONDS=$TIMEOUT_SECONDS,OVERWRITE=$OVERWRITE"
evaluation_variables="RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,RUN_ID=$RUN_ID,DATASET=$DATASET,SPLIT=$SPLIT,SMOKE_LIMIT=$SMOKE_LIMIT,EXPECTED_COUNT=$EXPECTED_COUNT,EVAL_MAX_WORKERS=$EVAL_MAX_WORKERS,EVAL_TIMEOUT=$EVAL_TIMEOUT,AGENTFORGE_MODEL=$AGENTFORGE_MODEL"
if [[ -n "$SWEBENCH_DATASET_REVISION" ]]; then
  collection_variables+=",SWEBENCH_DATASET_REVISION=$SWEBENCH_DATASET_REVISION"
  evaluation_variables+=",SWEBENCH_DATASET_REVISION=$SWEBENCH_DATASET_REVISION"
fi
if [[ -n "$TASK_IDS_FILE" ]]; then
  collection_variables+=",TASK_IDS_FILE=$TASK_IDS_FILE"
  evaluation_variables+=",TASK_IDS_FILE=$TASK_IDS_FILE"
fi
analysis_variables="RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,EXPECTED_COUNT=$EXPECTED_COUNT,ANALYSIS_SAMPLE_PER_SHARD=2,ANALYSIS_OUTPUT_SUBDIR=analysis-smoke"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Dry run: no jobs submitted.\n\n'
  printf 'Cluster logs: %s\n\n' "$CLUSTER_LOG_DIR"
  print_dry_run_job \
    "Collection job" \
    "cluster/pbs/collect_verified_smoke.pbs" \
    "" \
    "$collection_variables"
  if [[ "$SUBMIT_EVAL" == "1" ]]; then
    print_dry_run_job \
      "Evaluation job" \
      "cluster/pbs/evaluate_verified_smoke.pbs" \
      "Dependency: successful collection job" \
      "$evaluation_variables"
    if [[ "$SUBMIT_ANALYSIS" == "1" ]]; then
      print_dry_run_job \
        "Analysis job" \
        "cluster/pbs/analyze_verified.pbs" \
        "Dependency: successful evaluation job" \
        "$analysis_variables"
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

collection_job="$(qsub \
  "${qsub_log_args[@]}" \
  -v "$collection_variables" \
  cluster/pbs/collect_verified_smoke.pbs)"
echo "Submitted smoke collection: $collection_job"

if [[ "$SUBMIT_EVAL" == "1" ]]; then
  evaluation_job="$(qsub \
    "${qsub_log_args[@]}" \
    -W "depend=afterok:$collection_job" \
    -v "$evaluation_variables" \
    cluster/pbs/evaluate_verified_smoke.pbs)"
  echo "Submitted dependent smoke evaluation: $evaluation_job"

  if [[ "$SUBMIT_ANALYSIS" == "1" ]]; then
    RUN_NAME="$RUN_NAME" \
      EXPECTED_COUNT="$EXPECTED_COUNT" \
      AFTEROK_JOB_ID="$evaluation_job" \
      cluster/submit_verified_analysis_smoke.sh
  fi
fi
