#!/usr/bin/env bash
set -euo pipefail

HYPERSTACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY="${1:-verified}"
# shellcheck disable=SC1091
source "$HYPERSTACK_DIR/common.sh"

# Exercise every detected GPU/shard path with one task and a bounded agent
# budget.
export EXPECTED_TASKS="${EXPECTED_TASKS:-$NUM_SHARDS}"
export LIMIT="${LIMIT:-$EXPECTED_TASKS}"
export MAX_STEPS="${MAX_STEPS:-20}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"
export EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-$NUM_SHARDS}"

case "$FAMILY" in
  verified)
    export RUN_NAME="${RUN_NAME:-agentforge-verified-hyperstack-smoke}"
    ;;
  swesmith)
    export RUN_NAME="${RUN_NAME:-swesmith-hyperstack-smoke}"
    export TASK_IDS_FILE="${TASK_IDS_FILE:-data/splits/swesmith_train_1000_instance_ids.txt}"
    # Collection limits this larger candidate list to EXPECTED_TASKS. Evaluation
    # must therefore use the merged prediction IDs, not require every candidate.
    export EVALUATION_TASK_IDS_FILE="${EVALUATION_TASK_IDS_FILE-}"
    # One run at each temperature exercises multi-sample collection, merging,
    # evaluation, and analysis with 16 trajectories instead of 64.
    export RUNS_PER_TEMPERATURE="${RUNS_PER_TEMPERATURE:-1}"
    export TEMPERATURES="${TEMPERATURES:-0.6:0.7}"
    ;;
  *)
    echo "Usage: bash hyperstack/smoke.sh verified|swesmith" >&2
    exit 2
    ;;
esac

exec bash "$HYPERSTACK_DIR/pipeline.sh" "$FAMILY"
