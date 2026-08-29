#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
DATASET_REVISION="${SWEBENCH_DATASET_REVISION:-}"
if [[ -z "$DATASET_REVISION" && "$DATASET" == "princeton-nlp/SWE-bench_Verified" ]]; then
  DATASET_REVISION="c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
fi
SPLIT="${SPLIT:-test}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$ROOT_DIR/data/processed/agentforge_swebench_verified/predictions.jsonl}"
MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
RUN_ID="${RUN_ID:-agentforge_8b_sft_verified}"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/results/swebench}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$REPORT_DIR/${RUN_ID}_summary.json}"
MAX_WORKERS="${MAX_WORKERS:-1}"
TIMEOUT="${TIMEOUT:-1800}"
CACHE_LEVEL="${CACHE_LEVEL:-env}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --dataset "$DATASET"
  --split "$SPLIT"
  --predictions-path "$PREDICTIONS_PATH"
  --model "$MODEL"
  --run-id "$RUN_ID"
  --report-dir "$REPORT_DIR"
  --summary-output "$SUMMARY_OUTPUT"
  --max-workers "$MAX_WORKERS"
  --timeout "$TIMEOUT"
  --cache-level "$CACHE_LEVEL"
)

if [[ -n "$DATASET_REVISION" ]]; then
  args+=(--dataset-revision "$DATASET_REVISION")
fi

if [[ -n "${TASK_IDS_FILE:-}" ]]; then
  args+=(--instance-ids-file "$TASK_IDS_FILE")
fi
if [[ -n "${NAMESPACE+x}" ]]; then
  args+=(--namespace "$NAMESPACE")
fi
if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
  args+=(--force-rebuild)
fi
if [[ "${CLEAN:-0}" == "1" ]]; then
  args+=(--clean)
fi
if [[ "${MODAL:-0}" == "1" ]]; then
  args+=(--modal)
fi

"$UV_BIN" run --extra swebench python -m debug_depo.evaluate "${args[@]}" "$@"
