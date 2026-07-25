#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "Set RUN_ROOT to the completed SWE-smith run directory." >&2
  exit 2
fi

OUTPUT_DIR="${DEPO_OUTPUT_DIR:-$RUN_ROOT/preference-data/depo}"
mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
"$UV_BIN" run python -m debug_depo.build_depo_data \
  --run-root "$RUN_ROOT" \
  --output "$OUTPUT_DIR/trajectories.jsonl" \
  --desirable-output "$OUTPUT_DIR/desirable.jsonl" \
  --undesirable-output "$OUTPUT_DIR/undesirable.jsonl" \
  --summary-output "$OUTPUT_DIR/summary.json" \
  "$@"
