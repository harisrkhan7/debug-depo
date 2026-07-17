#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "Set RUN_ROOT to the run directory to analyze." >&2
  exit 2
fi

OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-$RUN_ROOT/analysis}"
EXPECTED_COUNT="${EXPECTED_COUNT:-500}"
SAMPLE_PER_SHARD="${ANALYSIS_SAMPLE_PER_SHARD:-0}"
mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
"$UV_BIN" run python -m debug_depo.analyze_run \
  --run-root "$RUN_ROOT" \
  --output-csv "$OUTPUT_DIR/instances.csv" \
  --summary-output "$OUTPUT_DIR/summary.json" \
  --expected-count "$EXPECTED_COUNT" \
  --sample-per-shard "$SAMPLE_PER_SHARD" \
  "$@"
