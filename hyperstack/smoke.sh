#!/usr/bin/env bash
set -euo pipefail

HYPERSTACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY="${1:-verified}"

# Keep all eight GPU/shard paths exercised, but assign only one task to each
# shard and use a bounded agent budget.
export EXPECTED_TASKS="${EXPECTED_TASKS:-8}"
export LIMIT="${LIMIT:-$EXPECTED_TASKS}"
export MAX_STEPS="${MAX_STEPS:-20}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"
export EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-8}"

case "$FAMILY" in
  verified)
    export RUN_NAME="${RUN_NAME:-agentforge-verified-hyperstack-smoke}"
    ;;
  swesmith)
    export RUN_NAME="${RUN_NAME:-swesmith-hyperstack-smoke}"
    export TASK_IDS_FILE="${TASK_IDS_FILE:-data/splits/swesmith_train_5000_instance_ids.txt}"
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
