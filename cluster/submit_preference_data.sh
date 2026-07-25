#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-swesmith-pilot}"
source cluster/resolve_run_paths.sh
TOKEN_METRIC="${TOKEN_METRIC:-total_tokens}"
MIN_COST_RATIO="${MIN_COST_RATIO:-1.1}"
MAX_PAIRS_PER_TASK="${MAX_PAIRS_PER_TASK:-0}"
INCLUDE_FAILURE_EFFICIENCY_PAIRS="${INCLUDE_FAILURE_EFFICIENCY_PAIRS:-0}"

variables="RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,TOKEN_METRIC=$TOKEN_METRIC,MIN_COST_RATIO=$MIN_COST_RATIO,MAX_PAIRS_PER_TASK=$MAX_PAIRS_PER_TASK,INCLUDE_FAILURE_EFFICIENCY_PAIRS=$INCLUDE_FAILURE_EFFICIENCY_PAIRS"
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
  printf 'DMPO job: qsub -v %q cluster/pbs/build_dmpo_pairs.pbs\n' "$variables"
  printf 'DEPO job: qsub -W depend=afterok:<dmpo-job-id> -v %q cluster/pbs/build_depo_data.pbs\n' "$variables"
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
  qsub -N "depo-data-$RUN_NAME" "${log_args[@]}" -W "depend=afterok:$dmpo_job" \
    -v "$variables" cluster/pbs/build_depo_data.pbs
)"
echo "Submitted dependent DEPO data build: $depo_job"
