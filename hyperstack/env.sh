#!/usr/bin/env bash

# Shared HyperStack defaults. Source hyperstack/local.env first when present so
# machine-specific paths and image pins remain outside version control.
HYPERSTACK_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPERSTACK_REPO_ROOT="$(cd "$HYPERSTACK_CONFIG_DIR/.." && pwd)"

if [[ -f "$HYPERSTACK_CONFIG_DIR/local.env" ]]; then
  # shellcheck disable=SC1091
  source "$HYPERSTACK_CONFIG_DIR/local.env"
fi

export DEBUG_DEPO_ROOT="${DEBUG_DEPO_ROOT:-$HYPERSTACK_REPO_ROOT}"
export HYPERSTACK_PERSISTENT_ROOT="${HYPERSTACK_PERSISTENT_ROOT:-/root/debug-depo-persistent}"
if [[ -z "$HYPERSTACK_PERSISTENT_ROOT" || "$HYPERSTACK_PERSISTENT_ROOT" == "/" ]]; then
  echo "HYPERSTACK_PERSISTENT_ROOT must be a dedicated directory, not '/'." >&2
  return 2 2>/dev/null || exit 2
fi
export HYPERSTACK_EPHEMERAL_ROOT="${HYPERSTACK_EPHEMERAL_ROOT:-/ephemeral/debug-depo}"
if [[ -z "$HYPERSTACK_EPHEMERAL_ROOT" || "$HYPERSTACK_EPHEMERAL_ROOT" == "/" ]]; then
  echo "HYPERSTACK_EPHEMERAL_ROOT must be a dedicated directory, not '/'." >&2
  return 2 2>/dev/null || exit 2
fi
if [[ "$HYPERSTACK_EPHEMERAL_ROOT" == "$HYPERSTACK_PERSISTENT_ROOT" ]]; then
  echo "Persistent and ephemeral roots must be different directories." >&2
  return 2 2>/dev/null || exit 2
fi

# Runs and derived research artifacts survive under the persistent volume.
# Rebuildable caches, SIFs, and temporary files use HyperStack's local ephemeral
# disk and are expected to disappear when the VM is hibernated or deleted.
export DEBUG_DEPO_SCRATCH="${DEBUG_DEPO_SCRATCH:-$HYPERSTACK_PERSISTENT_ROOT/scratch}"
export DEBUG_DEPO_EPHEMERAL="${DEBUG_DEPO_EPHEMERAL:-$HYPERSTACK_EPHEMERAL_ROOT}"
export DEBUG_DEPO_CACHE_ROOT="${DEBUG_DEPO_CACHE_ROOT:-$DEBUG_DEPO_EPHEMERAL/cache}"
export HYPERSTACK_RUNTIME_DIR="${HYPERSTACK_RUNTIME_DIR:-$DEBUG_DEPO_EPHEMERAL/run-state}"

export HF_HOME="${HF_HOME:-$DEBUG_DEPO_CACHE_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DEBUG_DEPO_CACHE_ROOT/xdg}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DEBUG_DEPO_CACHE_ROOT/uv}"
export TORCH_HOME="${TORCH_HOME:-$DEBUG_DEPO_CACHE_ROOT/torch}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$DEBUG_DEPO_CACHE_ROOT/apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$DEBUG_DEPO_EPHEMERAL/tmp/apptainer}"
export APPTAINER_MKSQUASHFS_ARGS="${APPTAINER_MKSQUASHFS_ARGS:--processors 2}"
export TMPDIR="${TMPDIR:-$DEBUG_DEPO_EPHEMERAL/tmp}"
export SWEBENCH_APPTAINER_CACHE_DIR="${SWEBENCH_APPTAINER_CACHE_DIR:-$DEBUG_DEPO_CACHE_ROOT/swebench/apptainer-cache}"
export SWEBENCH_APPTAINER_SIF_DIR="${SWEBENCH_APPTAINER_SIF_DIR:-$DEBUG_DEPO_EPHEMERAL/sifs/swebench}"
export SWESMITH_APPTAINER_CACHE_DIR="${SWESMITH_APPTAINER_CACHE_DIR:-$DEBUG_DEPO_CACHE_ROOT/swesmith/apptainer-cache}"
export SWESMITH_APPTAINER_SIF_DIR="${SWESMITH_APPTAINER_SIF_DIR:-$DEBUG_DEPO_EPHEMERAL/sifs/swesmith}"

export HF_TOKEN_FILE="${HF_TOKEN_FILE:-/root/.config/debug-depo/hf_token}"
export UV="${UV:-$HYPERSTACK_PERSISTENT_ROOT/tools/uv-venv/bin/uv}"
export VLLM_IMAGE="${VLLM_IMAGE:-$DEBUG_DEPO_EPHEMERAL/sifs/vllm/vllm-openai.sif}"
export VLLM_APPTAINER_SOURCE="${VLLM_APPTAINER_SOURCE:-docker://vllm/vllm-openai:latest}"
export VLLM_PORT_BASE="${VLLM_PORT_BASE:-18000}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

# H200 x8 defaults requested for this instance.
export NUM_SHARDS="${NUM_SHARDS:-8}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
export CACHE_BUILD_MAX_WORKERS="${CACHE_BUILD_MAX_WORKERS:-50}"
export EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-80}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"

export SWEBENCH_DATASET_REVISION="${SWEBENCH_DATASET_REVISION:-c104f840cc67f8b6eec6f759ebc8b2693d585d4a}"
export SWESMITH_DATASET_REVISION="${SWESMITH_DATASET_REVISION:-77cab9055d42ab4a5c25c89a8f937096db13558e}"

hyperstack_dirs=(
  "$HYPERSTACK_PERSISTENT_ROOT"
  "$DEBUG_DEPO_SCRATCH"
  "$DEBUG_DEPO_EPHEMERAL"
  "$DEBUG_DEPO_CACHE_ROOT"
  "$HYPERSTACK_RUNTIME_DIR"
  "$HF_HOME"
  "$HF_HUB_CACHE"
  "$XDG_CACHE_HOME"
  "$UV_CACHE_DIR"
  "$TORCH_HOME"
  "$APPTAINER_CACHEDIR"
  "$APPTAINER_TMPDIR"
  "$TMPDIR"
  "$SWEBENCH_APPTAINER_CACHE_DIR"
  "$SWEBENCH_APPTAINER_SIF_DIR"
  "$SWESMITH_APPTAINER_CACHE_DIR"
  "$SWESMITH_APPTAINER_SIF_DIR"
  "$(dirname "$VLLM_IMAGE")"
)
mkdir -p "${hyperstack_dirs[@]}"

unset HYPERSTACK_CONFIG_DIR HYPERSTACK_REPO_ROOT hyperstack_dirs
