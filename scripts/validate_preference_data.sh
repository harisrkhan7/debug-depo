#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

OBJECTIVE="${1:-}"
DATA_DIR="${2:-${RUN_ROOT:-}/preference-data/$OBJECTIVE}"
case "$OBJECTIVE" in
  dmpo|depo) ;;
  *)
    echo "Usage: scripts/validate_preference_data.sh dmpo|depo" >&2
    exit 2
    ;;
esac
if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "Set RUN_ROOT to the evaluated trajectory collection." >&2
  exit 2
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
"$UV_BIN" run python -m debug_depo.preference_data \
  --objective "$OBJECTIVE" \
  --data-dir "$DATA_DIR"
