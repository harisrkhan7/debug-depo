#!/usr/bin/env bash

DEBUG_DEPO_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEBUG_DEPO_REPO_ROOT="$(cd "$DEBUG_DEPO_CONFIG_DIR/../.." && pwd)"
export DEBUG_DEPO_ROOT="${DEBUG_DEPO_ROOT:-$DEBUG_DEPO_REPO_ROOT}"

if [[ -f "$DEBUG_DEPO_CONFIG_DIR/local.sh" ]]; then
  source "$DEBUG_DEPO_CONFIG_DIR/local.sh"
fi

source "$DEBUG_DEPO_CONFIG_DIR/defaults.sh"

mkdir_dirs=(
  "$DEBUG_DEPO_SCRATCH"
  "$HF_HOME"
  "$UV_CACHE_DIR"
  "$TMPDIR"
  "$APPTAINER_CACHEDIR"
  "$SWEBENCH_APPTAINER_CACHE_DIR"
  "$SWEBENCH_APPTAINER_SIF_DIR"
)

if [[ "$VLLM_IMAGE" != docker://* && "$VLLM_IMAGE" != oras://* ]]; then
  mkdir_dirs+=("$(dirname "$VLLM_IMAGE")")
fi

mkdir -p "${mkdir_dirs[@]}"

unset DEBUG_DEPO_CONFIG_DIR DEBUG_DEPO_REPO_ROOT mkdir_dirs
