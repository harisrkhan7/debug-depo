#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PBS_O_WORKDIR:-$ROOT_DIR}"

if [[ ! -f pyproject.toml ]]; then
  echo "Submit this job from the debug-depo repository root." >&2
  exit 2
fi

source cluster/load_job_environment.sh
source cluster/env/load.sh

EVALUATION_MODE="${EVALUATION_MODE:-full}"
case "$EVALUATION_MODE" in
  smoke)
    RUN_NAME="${RUN_NAME:-agentforge-verified-smoke}"
    DEFAULT_EVAL_MAX_WORKERS=2
    DEFAULT_EVAL_TIMEOUT=1800
    EXPECTED_SHARDS=1
    EVAL_LIMIT="${SMOKE_LIMIT:-5}"
    EXPECTED_PREDICTIONS="${EXPECTED_COUNT:-$EVAL_LIMIT}"
    RUN_ID="${RUN_ID:-agentforge_verified_smoke}"
    ;;
  full)
    RUN_NAME="${RUN_NAME:-agentforge-verified-full}"
    DEFAULT_EVAL_MAX_WORKERS=20
    DEFAULT_EVAL_TIMEOUT=3600
    EXPECTED_SHARDS="${NUM_SHARDS:-10}"
    EVAL_LIMIT=""
    EXPECTED_PREDICTIONS="${EXPECTED_COUNT:-500}"
    RUN_ID="${RUN_ID:-agentforge_verified_full}"
    ;;
  *)
    echo "EVALUATION_MODE must be 'smoke' or 'full', got: $EVALUATION_MODE" >&2
    exit 2
    ;;
esac

export DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
export SPLIT="${SPLIT:-test}"
RUN_ROOT="${RUN_ROOT:-$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME}"
SHARD_ROOT="${SHARD_ROOT:-$RUN_ROOT/rollouts}"
MERGED_DIR="${MERGED_DIR:-$RUN_ROOT/merged}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/evaluation}"
mkdir -p "$MERGED_DIR" "$EVAL_ROOT/reports" "$EVAL_ROOT/logs"

shopt -s nullglob
prediction_paths=("$SHARD_ROOT"/shard-*/predictions.jsonl)
shopt -u nullglob
if ((${#prediction_paths[@]} != EXPECTED_SHARDS)); then
  echo "Expected $EXPECTED_SHARDS prediction files under $SHARD_ROOT, found ${#prediction_paths[@]}." >&2
  printf '  %s\n' "${prediction_paths[@]:-<none>}" >&2
  exit 1
fi

MERGED="$MERGED_DIR/predictions.jsonl"
OUTPUT="$MERGED" \
SUMMARY_OUTPUT="$MERGED_DIR/predictions_summary.json" \
  scripts/merge_predictions.sh "${prediction_paths[@]}"

prediction_count="$(wc -l <"$MERGED")"
prediction_count="${prediction_count//[[:space:]]/}"
if ((prediction_count != EXPECTED_PREDICTIONS)); then
  echo "Expected $EXPECTED_PREDICTIONS merged predictions, found $prediction_count." >&2
  echo "Refusing to start an incomplete evaluation." >&2
  exit 1
fi

if [[ "${MERGE_ONLY:-0}" == "1" ]]; then
  echo "Merged and validated $prediction_count predictions at $MERGED"
  exit 0
fi

export PREDICTIONS_PATH="$MERGED"
export AGENTFORGE_MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
export RUN_ID
export REPORT_DIR="$EVAL_ROOT/reports"
export SUMMARY_OUTPUT="$EVAL_ROOT/${RUN_ID}_summary.json"
export LOG_DIR="$EVAL_ROOT/logs"
export MAX_WORKERS="${EVAL_MAX_WORKERS:-${MAX_WORKERS:-$DEFAULT_EVAL_MAX_WORKERS}}"
export TIMEOUT="${EVAL_TIMEOUT:-${TIMEOUT:-$DEFAULT_EVAL_TIMEOUT}}"

if [[ -n "$EVAL_LIMIT" ]]; then
  export LIMIT="$EVAL_LIMIT"
else
  unset LIMIT
fi

cat <<MSG
Starting $EVALUATION_MODE Apptainer evaluation
  run:        $RUN_NAME
  dataset:    $DATASET
  split:      $SPLIT
  predictions: $PREDICTIONS_PATH
  workers:    $MAX_WORKERS
  reports:    $REPORT_DIR
  logs:       $LOG_DIR
  limit:      ${LIMIT:-all submitted predictions}
MSG

scripts/evaluate_apptainer.sh
