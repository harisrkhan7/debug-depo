#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWESMITH_MODE=full
export TASK_IDS_FILE="${TASK_IDS_FILE-data/splits/swesmith_train_5000_instance_ids.txt}"
export NUM_SHARDS="${NUM_SHARDS:-50}"
if [[ -z "${EXPECTED_TASKS+x}" && -z "${TASK_LIMIT:-}" ]]; then
  export EXPECTED_TASKS=5000
fi
exec "$SCRIPT_DIR/submit_swesmith.sh" "$@"
