#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

MODE="${CACHE_BUILD_MODE:-smoke}"
DATASETS="${CACHE_BUILD_DATASETS:-both}"
SWEBENCH_DATASET="${SWEBENCH_DATASET:-princeton-nlp/SWE-bench_Verified}"
SWEBENCH_DATASET_REVISION="${SWEBENCH_DATASET_REVISION:-c104f840cc67f8b6eec6f759ebc8b2693d585d4a}"
SWEBENCH_SPLIT="${SWEBENCH_SPLIT:-test}"
SWEBENCH_IMAGE_TEMPLATE="${SWEBENCH_APPTAINER_IMAGE_TEMPLATE:-}"
if [[ -z "$SWEBENCH_IMAGE_TEMPLATE" ]]; then
  SWEBENCH_IMAGE_TEMPLATE='docker://ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest'
fi
SWEBENCH_SIF_DIR="${SWEBENCH_APPTAINER_SIF_DIR:-$ROOT_DIR/data/apptainer/swebench-sifs}"
SWEBENCH_CACHE_DIR="${SWEBENCH_APPTAINER_CACHE_DIR:-${APPTAINER_CACHEDIR:-}}"
SWESMITH_DATASET="${SWESMITH_DATASET:-SWE-bench/SWE-smith-py}"
SWESMITH_DATASET_REVISION="${SWESMITH_DATASET_REVISION:-77cab9055d42ab4a5c25c89a8f937096db13558e}"
SWESMITH_SPLIT="${SWESMITH_SPLIT:-train}"
SWESMITH_TASK_IDS_FILE="${SWESMITH_TASK_IDS_FILE:-${TASK_IDS_FILE:-$ROOT_DIR/data/splits/train_instance_ids.txt}}"
SWESMITH_SIF_DIR="${SWESMITH_APPTAINER_SIF_DIR:-$ROOT_DIR/data/apptainer/swesmith-sifs}"
SWESMITH_CACHE_DIR="${SWESMITH_APPTAINER_CACHE_DIR:-${APPTAINER_CACHEDIR:-}}"
MAX_WORKERS="${CACHE_BUILD_MAX_WORKERS:-4}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/external/SWE-smith${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --mode "$MODE"
  --datasets "$DATASETS"
  --swebench-dataset "$SWEBENCH_DATASET"
  --swebench-dataset-revision "$SWEBENCH_DATASET_REVISION"
  --swebench-split "$SWEBENCH_SPLIT"
  --expected-swebench-tasks "${EXPECTED_SWEBENCH_TASKS:-500}"
  --swebench-image-template "$SWEBENCH_IMAGE_TEMPLATE"
  --swebench-sif-dir "$SWEBENCH_SIF_DIR"
  --swesmith-dataset "$SWESMITH_DATASET"
  --swesmith-dataset-revision "$SWESMITH_DATASET_REVISION"
  --swesmith-split "$SWESMITH_SPLIT"
  --swesmith-task-ids-file "$SWESMITH_TASK_IDS_FILE"
  --swesmith-sif-dir "$SWESMITH_SIF_DIR"
  --max-workers "$MAX_WORKERS"
)

if [[ -n "$SWEBENCH_CACHE_DIR" ]]; then
  args+=(--swebench-apptainer-cache-dir "$SWEBENCH_CACHE_DIR")
fi
if [[ -n "$SWESMITH_CACHE_DIR" ]]; then
  args+=(--swesmith-apptainer-cache-dir "$SWESMITH_CACHE_DIR")
fi
if [[ -n "${SWEBENCH_TASK_IDS_FILE:-}" ]]; then
  args+=(--swebench-task-ids-file "$SWEBENCH_TASK_IDS_FILE")
fi
if [[ -n "${SWEBENCH_CACHE_LIMIT:-}" ]]; then
  args+=(--swebench-limit "$SWEBENCH_CACHE_LIMIT")
fi
if [[ -n "${SWESMITH_CACHE_LIMIT:-}" ]]; then
  args+=(--swesmith-limit "$SWESMITH_CACHE_LIMIT")
fi
if [[ -n "${CACHE_BUILD_SUMMARY_OUTPUT:-}" ]]; then
  args+=(--summary-output "$CACHE_BUILD_SUMMARY_OUTPUT")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi
if [[ "${PROGRESS:-1}" != "1" ]]; then
  args+=(--no-progress)
fi

"$UV_BIN" run python -m debug_depo.apptainer_cache "${args[@]}" "$@"
