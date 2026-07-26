#!/usr/bin/env bash
set -euo pipefail

HYPERSTACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${1:-verified}"

case "$PIPELINE" in
  verified)
    FAMILY=verified
    ;;
  swesmith)
    FAMILY=swesmith
    ;;
  validation)
    FAMILY=swesmith
    export RUN_NAME="${RUN_NAME:-swesmith-validation-500}"
    export TASK_IDS_FILE="${TASK_IDS_FILE:-data/splits/swesmith_validation_500_instance_ids.txt}"
    export EXPECTED_TASKS="${EXPECTED_TASKS:-500}"
    ;;
  *)
    echo "Usage: bash hyperstack/pipeline.sh verified|swesmith|validation" >&2
    exit 2
    ;;
esac

bash "$HYPERSTACK_DIR/collect.sh" "$FAMILY"
bash "$HYPERSTACK_DIR/evaluate.sh" "$FAMILY"
bash "$HYPERSTACK_DIR/analyze.sh" "$FAMILY"
