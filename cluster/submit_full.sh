#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p cluster/logs

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

RUN_NAME="${RUN_NAME:-agentforge-verified-full}"
RUN_ID="${RUN_ID:-${RUN_NAME//-/_}}"
DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
TASK_IDS_FILE="${TASK_IDS_FILE:-}"
NUM_SHARDS="${NUM_SHARDS:-10}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-20}"
AGENTFORGE_MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
HARNESS="${HARNESS:-mini-swe-agent-plus}"
MINI_SWE_RUNNER="${MINI_SWE_RUNNER:-singularity}"
MINI_SWE_ENVIRONMENT_CLASS="${MINI_SWE_ENVIRONMENT_CLASS:-singularity}"
MAX_STEPS="${MAX_STEPS:-200}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-3600}"
OVERWRITE="${OVERWRITE:-0}"
SUBMIT_EVAL="${SUBMIT_EVAL:-1}"
SUBMIT_ANALYSIS="${SUBMIT_ANALYSIS:-1}"
EXPECTED_COUNT="${EXPECTED_COUNT:-500}"

if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_NAME may contain only letters, numbers, dots, underscores, and dashes." >&2
  exit 2
fi
if [[ ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SHARDS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$EXPECTED_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_COUNT must be a positive integer." >&2
  exit 2
fi
if ((NUM_SHARDS > EXPECTED_COUNT)); then
  echo "NUM_SHARDS ($NUM_SHARDS) cannot exceed EXPECTED_COUNT ($EXPECTED_COUNT)." >&2
  echo "Reduce NUM_SHARDS so every collection shard receives at least one task." >&2
  exit 2
fi
last_shard=$((NUM_SHARDS - 1))

collection_variables="RUN_NAME=$RUN_NAME,DATASET=$DATASET,SPLIT=$SPLIT,NUM_SHARDS=$NUM_SHARDS,ROLLOUT_WORKERS=$ROLLOUT_WORKERS,AGENTFORGE_MODEL=$AGENTFORGE_MODEL,HARNESS=$HARNESS,MINI_SWE_RUNNER=$MINI_SWE_RUNNER,MINI_SWE_ENVIRONMENT_CLASS=$MINI_SWE_ENVIRONMENT_CLASS,MAX_STEPS=$MAX_STEPS,CONTEXT_LENGTH=$CONTEXT_LENGTH,TEMPERATURE=$TEMPERATURE,TOP_P=$TOP_P,TIMEOUT_SECONDS=$TIMEOUT_SECONDS,OVERWRITE=$OVERWRITE"
evaluation_variables="RUN_NAME=$RUN_NAME,RUN_ID=$RUN_ID,DATASET=$DATASET,SPLIT=$SPLIT,NUM_SHARDS=$NUM_SHARDS,EXPECTED_COUNT=$EXPECTED_COUNT,EVAL_MAX_WORKERS=$EVAL_MAX_WORKERS,EVAL_TIMEOUT=$EVAL_TIMEOUT,AGENTFORGE_MODEL=$AGENTFORGE_MODEL"
if [[ -n "$TASK_IDS_FILE" ]]; then
  collection_variables+=",TASK_IDS_FILE=$TASK_IDS_FILE"
  evaluation_variables+=",TASK_IDS_FILE=$TASK_IDS_FILE"
fi
analysis_variables="RUN_NAME=$RUN_NAME,EXPECTED_COUNT=$EXPECTED_COUNT,ANALYSIS_SAMPLE_PER_SHARD=0,ANALYSIS_OUTPUT_SUBDIR=analysis"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Dry run: no jobs submitted.\n\n'
  print_dry_run_job \
    "Collection job" \
    "cluster/pbs/collect_rollouts.pbs" \
    "Array indices: 0-$last_shard" \
    "$collection_variables"
  if [[ "$SUBMIT_EVAL" == "1" ]]; then
    print_dry_run_job \
      "Evaluation job" \
      "cluster/pbs/evaluate_all.pbs" \
      "Dependency: successful collection array" \
      "$evaluation_variables"
    if [[ "$SUBMIT_ANALYSIS" == "1" ]]; then
      print_dry_run_job \
        "Analysis job" \
        "cluster/pbs/analyze_run.pbs" \
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

collection_job="$(qsub \
  -J "0-$last_shard" \
  -v "$collection_variables" \
  cluster/pbs/collect_rollouts.pbs)"
echo "Submitted $NUM_SHARDS-shard collection array: $collection_job"

if [[ "$SUBMIT_EVAL" == "1" ]]; then
  evaluation_job="$(qsub \
    -W "depend=afterok:$collection_job" \
    -v "$evaluation_variables" \
    cluster/pbs/evaluate_all.pbs)"
  echo "Submitted dependent full evaluation: $evaluation_job"

  if [[ "$SUBMIT_ANALYSIS" == "1" ]]; then
    RUN_NAME="$RUN_NAME" \
      EXPECTED_COUNT="$EXPECTED_COUNT" \
      AFTEROK_JOB_ID="$evaluation_job" \
      cluster/submit_analysis_full.sh
  fi
fi
