#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p cluster/logs

RUN_NAME="${RUN_NAME:-agentforge-verified-full}"
NUM_SHARDS="${NUM_SHARDS:-10}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-20}"
MAX_STEPS="${MAX_STEPS:-200}"
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
if ((NUM_SHARDS < 1)); then
  echo "NUM_SHARDS must be at least 1." >&2
  exit 2
fi
last_shard=$((NUM_SHARDS - 1))

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "qsub -J 0-$last_shard -v RUN_NAME=$RUN_NAME,NUM_SHARDS=$NUM_SHARDS,ROLLOUT_WORKERS=$ROLLOUT_WORKERS,MAX_STEPS=$MAX_STEPS,TIMEOUT_SECONDS=$TIMEOUT_SECONDS,OVERWRITE=$OVERWRITE cluster/pbs/collect_rollouts.pbs"
  if [[ "$SUBMIT_EVAL" == "1" ]]; then
    echo "qsub -W depend=afterok:<collection-array-id> -v RUN_NAME=$RUN_NAME,NUM_SHARDS=$NUM_SHARDS,EVAL_MAX_WORKERS=$EVAL_MAX_WORKERS,EVAL_TIMEOUT=$EVAL_TIMEOUT cluster/pbs/evaluate_all.pbs"
    if [[ "$SUBMIT_ANALYSIS" == "1" ]]; then
      echo "qsub -N debug-depo-analysis-full -W depend=afterok:<evaluation-job-id> -v RUN_NAME=$RUN_NAME,EXPECTED_COUNT=$EXPECTED_COUNT,ANALYSIS_SAMPLE_PER_SHARD=0,ANALYSIS_OUTPUT_SUBDIR=analysis cluster/pbs/analyze_run.pbs"
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
  -v "RUN_NAME=$RUN_NAME,NUM_SHARDS=$NUM_SHARDS,ROLLOUT_WORKERS=$ROLLOUT_WORKERS,MAX_STEPS=$MAX_STEPS,TIMEOUT_SECONDS=$TIMEOUT_SECONDS,OVERWRITE=$OVERWRITE" \
  cluster/pbs/collect_rollouts.pbs)"
echo "Submitted $NUM_SHARDS-shard collection array: $collection_job"

if [[ "$SUBMIT_EVAL" == "1" ]]; then
  evaluation_job="$(qsub \
    -W "depend=afterok:$collection_job" \
    -v "RUN_NAME=$RUN_NAME,NUM_SHARDS=$NUM_SHARDS,EVAL_MAX_WORKERS=$EVAL_MAX_WORKERS,EVAL_TIMEOUT=$EVAL_TIMEOUT" \
    cluster/pbs/evaluate_all.pbs)"
  echo "Submitted dependent full evaluation: $evaluation_job"

  if [[ "$SUBMIT_ANALYSIS" == "1" ]]; then
    RUN_NAME="$RUN_NAME" \
      EXPECTED_COUNT="$EXPECTED_COUNT" \
      AFTEROK_JOB_ID="$evaluation_job" \
      cluster/submit_analysis_full.sh
  fi
fi
