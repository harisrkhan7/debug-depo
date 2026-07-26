#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/preference_defaults.sh"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "Set RUN_ROOT to the completed SWE-smith run directory." >&2
  exit 2
fi

OUTPUT_DIR="${DEPO_OUTPUT_DIR:-$RUN_ROOT/preference-data/depo}"
PREFERENCE_MAX_ROLLOUTS="${PREFERENCE_MAX_ROLLOUTS:-$PREFERENCE_MAX_ROLLOUTS_DEFAULT}"
mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${REBUILD_PREFERENCE_DATA:-0}" != "1" ]] && \
  "$ROOT_DIR/scripts/validate_preference_data.sh" depo "$OUTPUT_DIR" >/dev/null 2>&1; then
  echo "Reusing complete immutable DEPO data: $OUTPUT_DIR"
  exit 0
fi
args=(
  --run-root "$RUN_ROOT" \
  --output "$OUTPUT_DIR/trajectories.jsonl" \
  --desirable-output "$OUTPUT_DIR/desirable.jsonl" \
  --undesirable-output "$OUTPUT_DIR/undesirable.jsonl" \
  --summary-output "$OUTPUT_DIR/summary.json" \
  --max-rollouts "$PREFERENCE_MAX_ROLLOUTS"
)
if [[ -n "${PREFERENCE_SAMPLE_INDICES:-}" ]]; then
  args+=(--sample-indices "$PREFERENCE_SAMPLE_INDICES")
fi
"$UV_BIN" run python -m debug_depo.build_depo_data "${args[@]}" "$@"
"$ROOT_DIR/scripts/validate_preference_data.sh" depo "$OUTPUT_DIR" >/dev/null
