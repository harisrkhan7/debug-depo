#!/usr/bin/env bash
set -euo pipefail

HYPERSTACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HYPERSTACK_DIR/common.sh"

FAMILY="${1:-verified}"
case "$FAMILY" in
  verified)
    RUN_NAME="${RUN_NAME:-agentforge-verified-hyperstack}"
    EXPECTED_TASKS="${EXPECTED_TASKS:-500}"
    DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
    SPLIT="${SPLIT:-test}"
    ;;
  swesmith)
    RUN_NAME="${RUN_NAME:-swesmith-train-5000}"
    EXPECTED_TASKS="${EXPECTED_TASKS:-5000}"
    DATASET="${DATASET:-SWE-bench/SWE-smith-py}"
    SPLIT="${SPLIT:-train}"
    TASK_IDS_FILE="${TASK_IDS_FILE:-data/splits/swesmith_train_5000_instance_ids.txt}"
    ;;
  *)
    echo "Usage: bash hyperstack/evaluate.sh verified|swesmith" >&2
    exit 2
    ;;
esac

require_run_name "$RUN_NAME"
require_positive_integer NUM_SHARDS "$NUM_SHARDS"
require_positive_integer EXPECTED_TASKS "$EXPECTED_TASKS"
require_positive_integer EVAL_MAX_WORKERS "$EVAL_MAX_WORKERS"
if [[ "${SWESMITH_EVAL_RUNTIME:-apptainer}" != "apptainer" ]]; then
  echo "HyperStack requires SWESMITH_EVAL_RUNTIME=apptainer." >&2
  exit 2
fi
export SWESMITH_EVAL_RUNTIME=apptainer
RUN_ROOT="${RUN_ROOT:-$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME}"
TASK_IDS_FILE="${TASK_IDS_FILE:-}"
AGENTFORGE_MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"

cat <<MSG
HyperStack $FAMILY evaluation
  run:             $RUN_NAME
  run root:        $RUN_ROOT
  expected tasks:  $EXPECTED_TASKS
  expected shards: $NUM_SHARDS
  workers:         $EVAL_MAX_WORKERS
  runtime:         Apptainer
MSG
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run: no predictions merged or evaluated."
  exit 0
fi

require_separate_storage
require_command apptainer
require_project_environment
cd "$DEBUG_DEPO_ROOT"

if [[ "$FAMILY" == "verified" ]]; then
  shard_root="$RUN_ROOT/rollouts"
  merged_dir="$RUN_ROOT/merged"
  eval_root="$RUN_ROOT/evaluation"
  mkdir -p "$merged_dir" "$eval_root/reports" "$eval_root/logs"

  shopt -s nullglob
  prediction_paths=("$shard_root"/shard-*/predictions.jsonl)
  shopt -u nullglob
  if ((${#prediction_paths[@]} != NUM_SHARDS)); then
    echo "Expected $NUM_SHARDS prediction files under $shard_root, found ${#prediction_paths[@]}." >&2
    exit 1
  fi

  merged="$merged_dir/predictions.jsonl"
  OUTPUT="$merged" \
  SUMMARY_OUTPUT="$merged_dir/predictions_summary.json" \
    scripts/merge_predictions.sh "${prediction_paths[@]}"
  prediction_count="$(wc -l <"$merged")"
  prediction_count="${prediction_count//[[:space:]]/}"
  if ((prediction_count != EXPECTED_TASKS)); then
    echo "Expected $EXPECTED_TASKS merged predictions, found $prediction_count." >&2
    exit 1
  fi

  export DATASET SPLIT TASK_IDS_FILE AGENTFORGE_MODEL
  export PREDICTIONS_PATH="$merged"
  export RUN_ID="${RUN_ID:-${RUN_NAME//-/_}}"
  export REPORT_DIR="$eval_root/reports"
  export SUMMARY_OUTPUT="$eval_root/${RUN_ID}_summary.json"
  export LOG_DIR="$eval_root/logs"
  export MAX_WORKERS="$EVAL_MAX_WORKERS"
  export TIMEOUT="${EVAL_TIMEOUT:-3600}"
  scripts/evaluate_apptainer.sh
else
  runs_per_temperature="${RUNS_PER_TEMPERATURE:-4}"
  temperatures="${TEMPERATURES:-0.6:0.7}"
  read -r -a temperature_values <<<"${temperatures//:/ }"
  total_samples=$((${#temperature_values[@]} * runs_per_temperature))
  collection_root="$RUN_ROOT/collection"
  merged_root="$RUN_ROOT/merged"
  evaluation_root="$RUN_ROOT/evaluation"
  mkdir -p "$merged_root" "$evaluation_root"

  for ((sample_index = 0; sample_index < total_samples; sample_index++)); do
    shopt -s nullglob
    prediction_paths=(
      "$collection_root"/shard-*/samples/sample-"$sample_index"/predictions.jsonl
    )
    shopt -u nullglob
    if ((${#prediction_paths[@]} != NUM_SHARDS)); then
      echo "Sample $sample_index: expected $NUM_SHARDS shard files, found ${#prediction_paths[@]}." >&2
      exit 1
    fi

    sample_merged="$merged_root/sample-$sample_index"
    sample_eval="$evaluation_root/sample-$sample_index"
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

    DATASET="$DATASET" \
    SPLIT="$SPLIT" \
    TASK_IDS_FILE="$TASK_IDS_FILE" \
    PREDICTIONS_PATH="$sample_merged/predictions.jsonl" \
    SUMMARY_OUTPUT="$sample_eval/summary.json" \
    LOG_DIR="$sample_eval/logs" \
    EVAL_MAX_WORKERS="$EVAL_MAX_WORKERS" \
      scripts/evaluate_swesmith.sh
  done
fi

echo "$FAMILY evaluation complete: $RUN_ROOT/evaluation"
