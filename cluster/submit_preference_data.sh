#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-swesmith-pilot}"
source cluster/resolve_run_paths.sh
source scripts/preference_defaults.sh
TOKEN_METRIC="${TOKEN_METRIC:-total_tokens}"
MIN_COST_RATIO="${MIN_COST_RATIO:-1.1}"
MAX_PAIRS_PER_TASK="${MAX_PAIRS_PER_TASK:-0}"
INCLUDE_FAILURE_EFFICIENCY_PAIRS="${INCLUDE_FAILURE_EFFICIENCY_PAIRS:-0}"
PREFERENCE_MAX_ROLLOUTS="${PREFERENCE_MAX_ROLLOUTS:-$PREFERENCE_MAX_ROLLOUTS_DEFAULT}"
PREFERENCE_SAMPLE_INDICES="${PREFERENCE_SAMPLE_INDICES:-}"
PREFERENCE_SAMPLE_INDICES="${PREFERENCE_SAMPLE_INDICES//,/:}"
REBUILD_PREFERENCE_DATA="${REBUILD_PREFERENCE_DATA:-0}"
if [[ -n "$PREFERENCE_SAMPLE_INDICES" && ! "$PREFERENCE_SAMPLE_INDICES" =~ ^[0-9]+(:[0-9]+)*$ ]]; then
  echo "PREFERENCE_SAMPLE_INDICES must contain non-negative indices separated by ':' or ','." >&2
  exit 2
fi
if [[ "$REBUILD_PREFERENCE_DATA" != "0" && "$REBUILD_PREFERENCE_DATA" != "1" ]]; then
  echo "REBUILD_PREFERENCE_DATA must be 0 or 1." >&2
  exit 2
fi

variables="RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,TOKEN_METRIC=$TOKEN_METRIC,MIN_COST_RATIO=$MIN_COST_RATIO,MAX_PAIRS_PER_TASK=$MAX_PAIRS_PER_TASK,INCLUDE_FAILURE_EFFICIENCY_PAIRS=$INCLUDE_FAILURE_EFFICIENCY_PAIRS,PREFERENCE_MAX_ROLLOUTS=$PREFERENCE_MAX_ROLLOUTS,REBUILD_PREFERENCE_DATA=$REBUILD_PREFERENCE_DATA"
if [[ -n "$PREFERENCE_SAMPLE_INDICES" ]]; then
  variables+=",PREFERENCE_SAMPLE_INDICES=$PREFERENCE_SAMPLE_INDICES"
fi
dependency_args=()
dependency_description="none"
if [[ -n "${AFTEROK_JOB_ID:-}" ]]; then
  dependency_args=(-W "depend=afterok:$AFTEROK_JOB_ID")
  dependency_description="$AFTEROK_JOB_ID"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Dry run: no jobs submitted.\n'
  printf 'Run root: %s\n' "$RUN_ROOT"
  printf 'Initial dependency: %s\n' "$dependency_description"
  printf 'Completed artifacts: reused unless REBUILD_PREFERENCE_DATA=1\n'
  printf 'DMPO job: qsub -v %q cluster/pbs/build_dmpo_pairs.pbs\n' "$variables"
  printf 'DEPO job: qsub -v %q cluster/pbs/build_depo_data.pbs\n' "$variables"
  printf 'The two CPU jobs run independently and may overlap.\n'
  exit 0
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub is not available. Run this script on the cluster login node." >&2
  exit 127
fi
mkdir -p "$CLUSTER_LOG_DIR"
log_args=(-o "$CLUSTER_LOG_DIR/" -e "$CLUSTER_LOG_DIR/")
dmpo_job="$(
  qsub -N "dmpo-data-$RUN_NAME" "${log_args[@]}" "${dependency_args[@]}" \
    -v "$variables" cluster/pbs/build_dmpo_pairs.pbs
)"
echo "Submitted DMPO pair build: $dmpo_job"
depo_job="$(
  qsub -N "depo-data-$RUN_NAME" "${log_args[@]}" "${dependency_args[@]}" \
    -v "$variables" cluster/pbs/build_depo_data.pbs
)"
echo "Submitted DEPO trajectory build: $depo_job"
echo "Both datasets are one-time immutable inputs for repeated training trials."
