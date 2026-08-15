#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$CLOUD_DIR/common.sh"

TRAIN_RUN_ROOT="$DEBUG_DEPO_SCRATCH/runs/swesmith-train-1000-r2"
DMPO_MODEL="$TRAIN_RUN_ROOT/experiments/dmpo/g07-paper-informed/model"
DEPO_MODEL="$TRAIN_RUN_ROOT/experiments/dmpo-depo/g07-paper-informed/depo/total-balanced/model"

for model_path in "$DMPO_MODEL" "$DEPO_MODEL"; do
  if [[ ! -f "$model_path/config.json" ]]; then
    echo "Packaged model is missing: $model_path" >&2
    exit 2
  fi
done

run_verified() {
  local run_name="$1"
  local model="$2"
  local revision="${3:-}"

  echo "Starting SWE-bench Verified test run: $run_name"
  RUN_NAME="$run_name" \
  DATASET=princeton-nlp/SWE-bench_Verified \
  SPLIT=test \
  EXPECTED_TASKS=500 \
  TASK_IDS_FILE= \
  AGENTFORGE_MODEL="$model" \
  MINI_SWE_MODEL="hosted_vllm/$model" \
  VLLM_MODEL="$model" \
  VLLM_MODEL_REVISION="$revision" \
  CONTEXT_LENGTH=65536 \
  MAX_STEPS=200 \
  TEMPERATURE=0.0 \
  TOP_P=1.0 \
    bash "$CLOUD_DIR/run.sh" pipeline verified
}

run_verified \
  swesmith-train-1000-r2-sft-test-500 \
  "$BASELINE_SFT_MODEL" \
  "$BASELINE_SFT_MODEL_REVISION"

run_verified \
  swesmith-train-1000-r2-dmpo-g07-paper-informed-test-500 \
  "$DMPO_MODEL"

run_verified \
  swesmith-train-1000-r2-dmpo-g07-paper-informed-depo-total-balanced-test-500 \
  "$DEPO_MODEL"
