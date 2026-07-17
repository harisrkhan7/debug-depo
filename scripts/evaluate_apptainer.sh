#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$ROOT_DIR/data/processed/agentforge_swebench_verified/predictions.jsonl}"
MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
RUN_ID="${RUN_ID:-agentforge_8b_sft_verified}"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/results/swebench}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$REPORT_DIR/${RUN_ID}_summary.json}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/run_evaluation}"
SIF_DIR="${SWEBENCH_APPTAINER_SIF_DIR:-${SIF_DIR:-$ROOT_DIR/data/apptainer/sifs}}"
APPTAINER_CACHE_DIR="${SWEBENCH_APPTAINER_CACHE_DIR:-${APPTAINER_CACHEDIR:-}}"
IMAGE_TEMPLATE="${SWEBENCH_APPTAINER_IMAGE_TEMPLATE:-}"
if [[ -z "$IMAGE_TEMPLATE" ]]; then
  IMAGE_TEMPLATE='docker://ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest'
fi
MAX_WORKERS="${MAX_WORKERS:-1}"
TIMEOUT="${TIMEOUT:-1800}"

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
  --log-dir "$LOG_DIR"
  --sif-dir "$SIF_DIR"
  --image-template "$IMAGE_TEMPLATE"
  --max-workers "$MAX_WORKERS"
  --timeout "$TIMEOUT"
)

if [[ -n "$APPTAINER_CACHE_DIR" ]]; then
  args+=(--apptainer-cache-dir "$APPTAINER_CACHE_DIR")
fi
if [[ -n "${TASK_IDS_FILE:-}" ]]; then
  args+=(--instance-ids-file "$TASK_IDS_FILE")
fi
if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi
if [[ -n "${APPTAINER_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=($APPTAINER_ARGS)
  for arg in "${extra_args[@]}"; do
    args+=(--apptainer-arg "$arg")
  done
fi

"$UV_BIN" run --extra swebench python -m debug_depo.evaluate_apptainer "${args[@]}" "$@"
