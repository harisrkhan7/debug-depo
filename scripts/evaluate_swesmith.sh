#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if [[ -z "${PREDICTIONS_PATH:-}" ]]; then
  echo "Set PREDICTIONS_PATH to one merged SWE-smith sample." >&2
  exit 2
fi
if [[ -z "${SUMMARY_OUTPUT:-}" ]]; then
  echo "Set SUMMARY_OUTPUT for the SWE-smith evaluation summary." >&2
  exit 2
fi
if [[ -z "${LOG_DIR:-}" ]]; then
  echo "Set LOG_DIR for per-instance SWE-smith evaluation artifacts." >&2
  exit 2
fi

DATASET="${DATASET:-SWE-bench/SWE-smith-py}"
DATASET_REVISION="${SWESMITH_DATASET_REVISION:-77cab9055d42ab4a5c25c89a8f937096db13558e}"
SPLIT="${SPLIT:-train}"
RUNTIME="${SWESMITH_EVAL_RUNTIME:-apptainer}"
SIF_DIR="${SWESMITH_APPTAINER_SIF_DIR:-$ROOT_DIR/data/apptainer/swesmith-sifs}"
APPTAINER_CACHE_DIR="${SWESMITH_APPTAINER_CACHE_DIR:-${APPTAINER_CACHEDIR:-}}"
MAX_WORKERS="${EVAL_MAX_WORKERS:-4}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/external/SWE-smith${PYTHONPATH:+:$PYTHONPATH}"
args=(
  --dataset "$DATASET"
  --dataset-revision "$DATASET_REVISION"
  --split "$SPLIT"
  --predictions-path "$PREDICTIONS_PATH"
  --summary-output "$SUMMARY_OUTPUT"
  --log-dir "$LOG_DIR"
  --runtime "$RUNTIME"
  --sif-dir "$SIF_DIR"
  --max-workers "$MAX_WORKERS"
)
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  args+=(--require-complete)
fi
if [[ -n "$APPTAINER_CACHE_DIR" ]]; then
  args+=(--apptainer-cache-dir "$APPTAINER_CACHE_DIR")
fi
if [[ -n "${TASK_IDS_FILE:-}" ]]; then
  args+=(--instance-ids-file "$TASK_IDS_FILE")
fi
if [[ -n "${EVAL_TIMEOUT:-}" ]]; then
  args+=(--timeout "$EVAL_TIMEOUT")
fi
if [[ "${F2P_ONLY:-0}" == "1" ]]; then
  args+=(--f2p-only)
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

"$UV_BIN" run python -m debug_depo.swesmith_evaluate "${args[@]}" "$@"
