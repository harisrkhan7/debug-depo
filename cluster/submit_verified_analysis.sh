#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-agentforge-verified-full}"
source cluster/resolve_run_paths.sh
EXPECTED_COUNT="${EXPECTED_COUNT:-500}"
ANALYSIS_MODE="${ANALYSIS_MODE:-full}"

case "$ANALYSIS_MODE" in
  smoke)
    ANALYSIS_SAMPLE_PER_SHARD="${ANALYSIS_SAMPLE_PER_SHARD:-2}"
    ANALYSIS_OUTPUT_SUBDIR="${ANALYSIS_OUTPUT_SUBDIR:-analysis-smoke}"
    ;;
  full)
    ANALYSIS_SAMPLE_PER_SHARD="${ANALYSIS_SAMPLE_PER_SHARD:-0}"
    ANALYSIS_OUTPUT_SUBDIR="${ANALYSIS_OUTPUT_SUBDIR:-analysis}"
    ;;
  *)
    echo "ANALYSIS_MODE must be 'smoke' or 'full', got: $ANALYSIS_MODE" >&2
    exit 2
    ;;
esac

if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_NAME may contain only letters, numbers, dots, underscores, and dashes." >&2
  exit 2
fi

pbs_variables="RUN_NAME=$RUN_NAME,RUN_ROOT=$RUN_ROOT,EXPECTED_COUNT=$EXPECTED_COUNT,ANALYSIS_SAMPLE_PER_SHARD=$ANALYSIS_SAMPLE_PER_SHARD,ANALYSIS_OUTPUT_SUBDIR=$ANALYSIS_OUTPUT_SUBDIR"
dependency_args=()
if [[ -n "${AFTEROK_JOB_ID:-}" ]]; then
  dependency_args=(-W "depend=afterok:$AFTEROK_JOB_ID")
fi
command=(qsub -N "verified-analysis-$ANALYSIS_MODE" -o "$CLUSTER_LOG_DIR/" -e "$CLUSTER_LOG_DIR/" "${dependency_args[@]}" -v "$pbs_variables" cluster/pbs/analyze_verified.pbs)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Cluster logs: %s\n' "$CLUSTER_LOG_DIR"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub is not available. Run this script on the cluster login node." >&2
  exit 127
fi

mkdir -p "$CLUSTER_LOG_DIR"
job_id="$("${command[@]}")"
echo "Submitted $ANALYSIS_MODE analysis for $RUN_NAME: $job_id"
