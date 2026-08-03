#!/usr/bin/env bash

# Shared cloud defaults. Source cloud/local.env first when present so
# machine-specific paths and image pins remain outside version control. Legacy
# HYPERSTACK_* overrides remain accepted for existing deployments.
CLOUD_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOUD_REPO_ROOT="$(cd "$CLOUD_CONFIG_DIR/.." && pwd)"

CLOUD_LOCAL_ENV="${CLOUD_LOCAL_ENV:-${HYPERSTACK_LOCAL_ENV:-$CLOUD_CONFIG_DIR/local.env}}"
if [[ -f "$CLOUD_LOCAL_ENV" ]]; then
  # shellcheck disable=SC1091
  source "$CLOUD_LOCAL_ENV"
fi

export DEBUG_DEPO_ROOT="${DEBUG_DEPO_ROOT:-$CLOUD_REPO_ROOT}"
export CLOUD_PERSISTENT_ROOT="${CLOUD_PERSISTENT_ROOT:-${HYPERSTACK_PERSISTENT_ROOT:-/lambda/nfs/Debug-Depo/debug-depo-persistent}}"
if [[ -z "$CLOUD_PERSISTENT_ROOT" || "$CLOUD_PERSISTENT_ROOT" == "/" ]]; then
  echo "CLOUD_PERSISTENT_ROOT must be a dedicated directory, not '/'." >&2
  return 2 2>/dev/null || exit 2
fi
export CLOUD_EPHEMERAL_ROOT="${CLOUD_EPHEMERAL_ROOT:-${HYPERSTACK_EPHEMERAL_ROOT:-${HOME:?HOME must be set}/debug-depo-ephemeral}}"
if [[ -z "$CLOUD_EPHEMERAL_ROOT" || "$CLOUD_EPHEMERAL_ROOT" == "/" ]]; then
  echo "CLOUD_EPHEMERAL_ROOT must be a dedicated directory, not '/'." >&2
  return 2 2>/dev/null || exit 2
fi
if [[ "$CLOUD_EPHEMERAL_ROOT" == "$CLOUD_PERSISTENT_ROOT" ]]; then
  echo "Persistent and ephemeral roots must be different directories." >&2
  return 2 2>/dev/null || exit 2
fi

# Runs and derived research artifacts survive under the persistent volume.
# Rebuildable caches, SIFs, and temporary files use the VM's local root volume
# and are expected to disappear when a Lambda Cloud instance is terminated.
export DEBUG_DEPO_SCRATCH="${DEBUG_DEPO_SCRATCH:-$CLOUD_PERSISTENT_ROOT/scratch}"
export DEBUG_DEPO_EPHEMERAL="${DEBUG_DEPO_EPHEMERAL:-$CLOUD_EPHEMERAL_ROOT}"
export DEBUG_DEPO_CACHE_ROOT="${DEBUG_DEPO_CACHE_ROOT:-$DEBUG_DEPO_EPHEMERAL/cache}"
export DEBUG_DEPO_SIF_ROOT="${DEBUG_DEPO_SIF_ROOT:-$DEBUG_DEPO_EPHEMERAL/sifs}"
export CLOUD_RUNTIME_DIR="${CLOUD_RUNTIME_DIR:-${HYPERSTACK_RUNTIME_DIR:-$DEBUG_DEPO_EPHEMERAL/run-state}}"

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
export SWEBENCH_APPTAINER_SIF_DIR="${SWEBENCH_APPTAINER_SIF_DIR:-$DEBUG_DEPO_SIF_ROOT/swebench}"
export SWESMITH_APPTAINER_CACHE_DIR="${SWESMITH_APPTAINER_CACHE_DIR:-$DEBUG_DEPO_CACHE_ROOT/swesmith/apptainer-cache}"
export SWESMITH_APPTAINER_SIF_DIR="${SWESMITH_APPTAINER_SIF_DIR:-$DEBUG_DEPO_SIF_ROOT/swesmith}"

export HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/debug-depo/hf_token}"
export UV="${UV:-$CLOUD_PERSISTENT_ROOT/tools/uv-venv/bin/uv}"
export VLLM_IMAGE="${VLLM_IMAGE:-$DEBUG_DEPO_SIF_ROOT/vllm/vllm-openai.sif}"
export VLLM_APPTAINER_SOURCE="${VLLM_APPTAINER_SOURCE:-docker://vllm/vllm-openai:v0.11.0}"
export VLLM_PORT_BASE="${VLLM_PORT_BASE:-18000}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_LOG_REQUESTS="${VLLM_LOG_REQUESTS:-1}"
export VLLM_MAX_LOG_LEN="${VLLM_MAX_LOG_LEN:-2048}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export STREAM_OUTPUT="${STREAM_OUTPUT:-1}"

# Use every GPU reported by nvidia-smi by default. The fallback keeps offline
# dry-runs and CPU-only evaluation hosts usable; GPU workflows validate the
# configured IDs before doing work. Set GPU_IDS explicitly to select a subset.
if [[ -n "${GPU_IDS:-}" ]]; then
  CLOUD_GPU_SOURCE="${CLOUD_GPU_SOURCE:-${HYPERSTACK_GPU_SOURCE:-configured}}"
else
  detected_gpu_ids=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    detected_gpu_ids="$({
      nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true
    } | awk '
      /^[[:space:]]*[0-9]+[[:space:]]*$/ {
        gsub(/[[:space:]]/, "")
        ids = ids (ids == "" ? "" : ",") $0
      }
      END { print ids }
    ')"
  fi
  if [[ -n "$detected_gpu_ids" ]]; then
    GPU_IDS="$detected_gpu_ids"
    CLOUD_GPU_SOURCE="nvidia-smi"
  else
    GPU_IDS="0,1,2,3,4,5,6,7"
    CLOUD_GPU_SOURCE="offline fallback"
  fi
fi
normalized_gpu_ids="${GPU_IDS//,/ }"
read -r -a detected_gpu_id_array <<<"$normalized_gpu_ids"
CLOUD_GPU_COUNT="${#detected_gpu_id_array[@]}"
export GPU_IDS CLOUD_GPU_COUNT CLOUD_GPU_SOURCE
export NUM_SHARDS="${NUM_SHARDS:-$CLOUD_GPU_COUNT}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
export CLOUD_SHARD_MAX_ATTEMPTS="${CLOUD_SHARD_MAX_ATTEMPTS:-3}"
export CLOUD_SHARD_STALL_TIMEOUT_SECONDS="${CLOUD_SHARD_STALL_TIMEOUT_SECONDS:-600}"
export CLOUD_WATCHDOG_INTERVAL_SECONDS="${CLOUD_WATCHDOG_INTERVAL_SECONDS:-30}"
export CLOUD_SHARD_RETRY_DELAY_SECONDS="${CLOUD_SHARD_RETRY_DELAY_SECONDS:-15}"
export CACHE_BUILD_MAX_WORKERS="${CACHE_BUILD_MAX_WORKERS:-50}"
export EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-80}"
export NUM_PROCESSES="${NUM_PROCESSES:-$CLOUD_GPU_COUNT}"

# Compatibility aliases for existing local.env files and external automation.
export HYPERSTACK_LOCAL_ENV="$CLOUD_LOCAL_ENV"
export HYPERSTACK_PERSISTENT_ROOT="$CLOUD_PERSISTENT_ROOT"
export HYPERSTACK_EPHEMERAL_ROOT="$CLOUD_EPHEMERAL_ROOT"
export HYPERSTACK_RUNTIME_DIR="$CLOUD_RUNTIME_DIR"
export HYPERSTACK_GPU_COUNT="$CLOUD_GPU_COUNT"
export HYPERSTACK_GPU_SOURCE="$CLOUD_GPU_SOURCE"

export SWEBENCH_DATASET_REVISION="${SWEBENCH_DATASET_REVISION:-c104f840cc67f8b6eec6f759ebc8b2693d585d4a}"
export SWESMITH_DATASET_REVISION="${SWESMITH_DATASET_REVISION:-77cab9055d42ab4a5c25c89a8f937096db13558e}"

cloud_dirs=(
  "$CLOUD_PERSISTENT_ROOT"
  "$DEBUG_DEPO_SCRATCH"
  "$DEBUG_DEPO_EPHEMERAL"
  "$DEBUG_DEPO_CACHE_ROOT"
  "$DEBUG_DEPO_SIF_ROOT"
  "$CLOUD_RUNTIME_DIR"
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
mkdir -p "${cloud_dirs[@]}"

unset CLOUD_CONFIG_DIR CLOUD_REPO_ROOT cloud_dirs
unset detected_gpu_ids detected_gpu_id_array normalized_gpu_ids
