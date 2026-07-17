#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p cluster/logs

RUN_NAME="${RUN_NAME:-agentforge-verified-smoke}"
SMOKE_LIMIT="${SMOKE_LIMIT:-5}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-2}"
MAX_STEPS="${MAX_STEPS:-200}"
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

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "qsub -v RUN_NAME=$RUN_NAME,SMOKE_LIMIT=$SMOKE_LIMIT,ROLLOUT_WORKERS=$ROLLOUT_WORKERS,MAX_STEPS=$MAX_STEPS,TIMEOUT_SECONDS=$TIMEOUT_SECONDS,OVERWRITE=$OVERWRITE cluster/pbs/collect_smoke.pbs"
  if [[ "$SUBMIT_EVAL" == "1" ]]; then
    echo "qsub -W depend=afterok:<collection-job-id> -v RUN_NAME=$RUN_NAME,SMOKE_LIMIT=$SMOKE_LIMIT,EVAL_MAX_WORKERS=$EVAL_MAX_WORKERS,EVAL_TIMEOUT=$EVAL_TIMEOUT cluster/pbs/evaluate_smoke.pbs"
    if [[ "$SUBMIT_ANALYSIS" == "1" ]]; then
      echo "qsub -N debug-depo-analysis-smoke -W depend=afterok:<evaluation-job-id> -v RUN_NAME=$RUN_NAME,EXPECTED_COUNT=$EXPECTED_COUNT,ANALYSIS_SAMPLE_PER_SHARD=2,ANALYSIS_OUTPUT_SUBDIR=analysis-smoke cluster/pbs/analyze_run.pbs"
    fi
  fi
  exit 0
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub is not available. Run this script on the cluster login node." >&2
  exit 127
fi

collection_job="$(qsub \
  -v "RUN_NAME=$RUN_NAME,SMOKE_LIMIT=$SMOKE_LIMIT,ROLLOUT_WORKERS=$ROLLOUT_WORKERS,MAX_STEPS=$MAX_STEPS,TIMEOUT_SECONDS=$TIMEOUT_SECONDS,OVERWRITE=$OVERWRITE" \
  cluster/pbs/collect_smoke.pbs)"
echo "Submitted smoke collection: $collection_job"

if [[ "$SUBMIT_EVAL" == "1" ]]; then
  evaluation_job="$(qsub \
    -W "depend=afterok:$collection_job" \
    -v "RUN_NAME=$RUN_NAME,SMOKE_LIMIT=$SMOKE_LIMIT,EVAL_MAX_WORKERS=$EVAL_MAX_WORKERS,EVAL_TIMEOUT=$EVAL_TIMEOUT" \
    cluster/pbs/evaluate_smoke.pbs)"
  echo "Submitted dependent smoke evaluation: $evaluation_job"

  if [[ "$SUBMIT_ANALYSIS" == "1" ]]; then
    RUN_NAME="$RUN_NAME" \
      EXPECTED_COUNT="$EXPECTED_COUNT" \
      AFTEROK_JOB_ID="$evaluation_job" \
      cluster/submit_analysis_smoke.sh
  fi
fi
