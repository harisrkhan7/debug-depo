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

MODE="${SWESMITH_MODE:-pilot}"
case "$MODE" in
  smoke)
    DEFAULT_RUN_NAME=swesmith-smoke
    DEFAULT_EXPECTED_TASKS=2
    DEFAULT_EXPECTED_SHARDS=1
    DEFAULT_EVAL_MAX_WORKERS=2
    ;;
  pilot)
    DEFAULT_RUN_NAME=swesmith-pilot
    DEFAULT_EXPECTED_TASKS=30
    DEFAULT_EXPECTED_SHARDS=3
    DEFAULT_EVAL_MAX_WORKERS=12
    ;;
  full)
    DEFAULT_RUN_NAME=swesmith-full
    DEFAULT_EXPECTED_TASKS=50908
    DEFAULT_EXPECTED_SHARDS=50
    DEFAULT_EVAL_MAX_WORKERS=25
    ;;
  *)
    echo "SWESMITH_MODE must be smoke, pilot, or full, got: $MODE" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
RUN_ROOT="${RUN_ROOT:-$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-8}"
EXPECTED_TASKS="${EXPECTED_TASKS:-$DEFAULT_EXPECTED_TASKS}"
EXPECTED_SHARDS="${NUM_SHARDS:-$DEFAULT_EXPECTED_SHARDS}"
COLLECTION_ROOT="${COLLECTION_ROOT:-$RUN_ROOT/collection}"
MERGED_ROOT="${MERGED_ROOT:-$RUN_ROOT/merged}"
EVALUATION_ROOT="${EVALUATION_ROOT:-$RUN_ROOT/evaluation}"
export EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-$DEFAULT_EVAL_MAX_WORKERS}"
mkdir -p "$MERGED_ROOT" "$EVALUATION_ROOT"

for ((sample_index = 0; sample_index < TOTAL_SAMPLES; sample_index++)); do
  shopt -s nullglob
  prediction_paths=(
    "$COLLECTION_ROOT"/shard-*/samples/sample-"$sample_index"/predictions.jsonl
  )
  shopt -u nullglob
  if ((${#prediction_paths[@]} != EXPECTED_SHARDS)); then
    echo "Sample $sample_index: expected $EXPECTED_SHARDS shard files, found ${#prediction_paths[@]}." >&2
    exit 1
  fi

  sample_merged="$MERGED_ROOT/sample-$sample_index"
  sample_eval="$EVALUATION_ROOT/sample-$sample_index"
  mkdir -p "$sample_merged" "$sample_eval/logs"
  OUTPUT="$sample_merged/predictions.jsonl" \
  SUMMARY_OUTPUT="$sample_merged/predictions_summary.json" \
    scripts/merge_predictions.sh "${prediction_paths[@]}"

  prediction_count="$(wc -l <"$sample_merged/predictions.jsonl")"
  prediction_count="${prediction_count//[[:space:]]/}"
  if ((prediction_count != EXPECTED_TASKS)); then
    echo "Sample $sample_index: expected $EXPECTED_TASKS predictions, found $prediction_count." >&2
    exit 1
  fi

  PREDICTIONS_PATH="$sample_merged/predictions.jsonl" \
  SUMMARY_OUTPUT="$sample_eval/summary.json" \
  LOG_DIR="$sample_eval/logs" \
    scripts/evaluate_swesmith.sh
done
