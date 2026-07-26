#!/usr/bin/env bash

HYPERSTACK_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HYPERSTACK_COMMON_DIR/env.sh"

require_command() {
  local command_name="$1"
  local setup_hint="${2:-Run bash hyperstack/setup.sh first.}"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command_name" >&2
    echo "$setup_hint" >&2
    return 127
  fi
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer, got: $value" >&2
    return 2
  fi
}

require_run_name() {
  local value="$1"
  if [[ ! "$value" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, numbers, dots, underscores, and dashes." >&2
    return 2
  fi
}

gpu_id_array() {
  local normalized="${GPU_IDS//,/ }"
  read -r -a HYPERSTACK_GPU_ID_ARRAY <<<"$normalized"
  if ((${#HYPERSTACK_GPU_ID_ARRAY[@]} != NUM_SHARDS)); then
    echo "GPU_IDS contains ${#HYPERSTACK_GPU_ID_ARRAY[@]} IDs but NUM_SHARDS=$NUM_SHARDS." >&2
    return 2
  fi
  local gpu_id
  for gpu_id in "${HYPERSTACK_GPU_ID_ARRAY[@]}"; do
    if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
      echo "GPU_IDS must contain non-negative integers, got: $gpu_id" >&2
      return 2
    fi
  done
}

require_project_environment() {
  if [[ ! -x "$UV" ]]; then
    echo "uv is missing at $UV. Run bash hyperstack/setup.sh first." >&2
    return 127
  fi
  if [[ ! -x "$DEBUG_DEPO_ROOT/.venv/bin/python" ]]; then
    echo "Project environment is missing at $DEBUG_DEPO_ROOT/.venv." >&2
    echo "Run bash hyperstack/setup.sh first." >&2
    return 127
  fi
}

require_separate_storage() {
  local persistent_device
  local ephemeral_device
  local path
  local path_device
  local -a persistent_paths
  local -a ephemeral_paths

  persistent_device="$(findmnt -n -o MAJ:MIN -T "$HYPERSTACK_PERSISTENT_ROOT" 2>/dev/null || true)"
  ephemeral_device="$(findmnt -n -o MAJ:MIN -T "$HYPERSTACK_EPHEMERAL_ROOT" 2>/dev/null || true)"
  if [[ -z "$persistent_device" || -z "$ephemeral_device" ]]; then
    echo "Could not resolve the persistent and ephemeral filesystems." >&2
    return 2
  fi
  if [[ "$persistent_device" == "$ephemeral_device" ]]; then
    echo "Persistent and ephemeral roots resolve to the same filesystem:" >&2
    echo "  persistent: $HYPERSTACK_PERSISTENT_ROOT" >&2
    echo "  ephemeral:  $HYPERSTACK_EPHEMERAL_ROOT" >&2
    echo "Mount HyperStack ephemeral storage at /ephemeral or override HYPERSTACK_EPHEMERAL_ROOT." >&2
    return 2
  fi

  persistent_paths=("$DEBUG_DEPO_SCRATCH")
  for path in "${persistent_paths[@]}"; do
    path_device="$(findmnt -n -o MAJ:MIN -T "$path" 2>/dev/null || true)"
    if [[ "$path_device" != "$persistent_device" ]]; then
      echo "Persistent artifact path is not on $HYPERSTACK_PERSISTENT_ROOT: $path" >&2
      return 2
    fi
  done

  ephemeral_paths=(
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
  for path in "${ephemeral_paths[@]}"; do
    path_device="$(findmnt -n -o MAJ:MIN -T "$path" 2>/dev/null || true)"
    if [[ "$path_device" != "$ephemeral_device" ]]; then
      echo "Rebuildable cache/SIF path is not on $HYPERSTACK_EPHEMERAL_ROOT: $path" >&2
      return 2
    fi
  done
}

wait_for_vllm() {
  local url="$1"
  local process_id="$2"
  local log_path="$3"
  local timeout="${VLLM_STARTUP_TIMEOUT:-7200}"
  local deadline=$((SECONDS + timeout))
  until curl --fail --silent --max-time 2 "$url/models" >/dev/null; do
    if ! kill -0 "$process_id" 2>/dev/null; then
      wait "$process_id"
      return $?
    fi
    if ((SECONDS >= deadline)); then
      echo "vLLM did not become ready at $url; see $log_path" >&2
      return 1
    fi
    sleep 5
  done
}

available_gib() {
  df -Pk "$1" | awk 'NR == 2 {print int($4 / 1024 / 1024)}'
}

unset HYPERSTACK_COMMON_DIR
