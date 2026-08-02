#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${1:-verified}"

case "$PIPELINE" in
  verified)
    FAMILY=verified
    ;;
  swesmith)
    FAMILY=swesmith
    ;;
  validation)
    exec bash "$CLOUD_DIR/validate.sh"
    ;;
  *)
    echo "Usage: bash cloud/pipeline.sh verified|swesmith|validation" >&2
    exit 2
    ;;
esac

bash "$CLOUD_DIR/collect.sh" "$FAMILY"
bash "$CLOUD_DIR/evaluate.sh" "$FAMILY"
bash "$CLOUD_DIR/analyze.sh" "$FAMILY"
