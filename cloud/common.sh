#!/usr/bin/env bash

CLOUD_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$CLOUD_COMMON_DIR/env.sh"

require_command() {
  local command_name="$1"
  local setup_hint="${2:-Run bash cloud/setup.sh first.}"
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

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a non-negative integer, got: $value" >&2
    return 2
  fi
}

require_model_timeout_watchdog_compatibility() {
  local model_timeout="${MINI_SWE_MODEL_TIMEOUT_SECONDS:-}"
  local stall_timeout="${CLOUD_SHARD_STALL_TIMEOUT_SECONDS:-0}"
  if [[ -z "$model_timeout" ]]; then
    return 0
  fi
  require_positive_integer MINI_SWE_MODEL_TIMEOUT_SECONDS "$model_timeout" || return
  if ((stall_timeout > 0 && stall_timeout <= model_timeout)); then
    echo "CLOUD_SHARD_STALL_TIMEOUT_SECONDS must exceed MINI_SWE_MODEL_TIMEOUT_SECONDS, or be 0." >&2
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

is_lambda_persistent_fstype() {
  case "$1" in
    nfs|nfs4|virtiofs) return 0 ;;
    *) return 1 ;;
  esac
}

gpu_id_array() {
  local normalized="${GPU_IDS//,/ }"
  read -r -a CLOUD_GPU_ID_ARRAY <<<"$normalized"
  HYPERSTACK_GPU_ID_ARRAY=("${CLOUD_GPU_ID_ARRAY[@]}")
  if ((${#CLOUD_GPU_ID_ARRAY[@]} != NUM_SHARDS)); then
    echo "GPU_IDS contains ${#CLOUD_GPU_ID_ARRAY[@]} IDs but NUM_SHARDS=$NUM_SHARDS." >&2
    return 2
  fi
  local gpu_id
  local seen_gpu_ids=" "
  for gpu_id in "${CLOUD_GPU_ID_ARRAY[@]}"; do
    if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
      echo "GPU_IDS must contain non-negative integers, got: $gpu_id" >&2
      return 2
    fi
    if [[ "$seen_gpu_ids" == *" $gpu_id "* ]]; then
      echo "GPU_IDS must not contain duplicate IDs, got: $GPU_IDS" >&2
      return 2
    fi
    seen_gpu_ids+="$gpu_id "
  done
}

require_project_environment() {
  if [[ ! -x "$UV" ]]; then
    echo "uv is missing at $UV. Run bash cloud/setup.sh first." >&2
    return 127
  fi
  if [[ ! -x "$DEBUG_DEPO_ROOT/.venv/bin/python" ]]; then
    echo "Project environment is missing at $DEBUG_DEPO_ROOT/.venv." >&2
    echo "Run bash cloud/setup.sh first." >&2
    return 127
  fi
}

require_separate_storage() {
  local persistent_device
  local ephemeral_device
  local path
  local path_device
  local vllm_image_device
  local -a persistent_paths
  local -a ephemeral_paths

  persistent_device="$(findmnt -n -o MAJ:MIN -T "$CLOUD_PERSISTENT_ROOT" 2>/dev/null || true)"
  ephemeral_device="$(findmnt -n -o MAJ:MIN -T "$CLOUD_EPHEMERAL_ROOT" 2>/dev/null || true)"
  if [[ -z "$persistent_device" || -z "$ephemeral_device" ]]; then
    echo "Could not resolve the persistent and ephemeral filesystems." >&2
    return 2
  fi
  if [[ "$persistent_device" == "$ephemeral_device" ]]; then
    echo "Persistent and ephemeral roots resolve to the same filesystem:" >&2
    echo "  persistent: $CLOUD_PERSISTENT_ROOT" >&2
    echo "  ephemeral:  $CLOUD_EPHEMERAL_ROOT" >&2
    echo "Attach a Lambda filesystem for durable artifacts and keep CLOUD_EPHEMERAL_ROOT on the local root volume." >&2
    return 2
  fi

  if [[ "$CLOUD_PERSISTENT_ROOT" == /lambda/nfs/* ]]; then
    local persistent_type
    persistent_type="$(findmnt -n -o FSTYPE -T "$CLOUD_PERSISTENT_ROOT" 2>/dev/null || true)"
    if ! is_lambda_persistent_fstype "$persistent_type"; then
      echo "Lambda persistent root is not backed by an attached Lambda filesystem: $CLOUD_PERSISTENT_ROOT" >&2
      echo "Detected filesystem type: ${persistent_type:-unknown}" >&2
      echo "Attach the Lambda filesystem when launching the instance." >&2
      return 2
    fi
  fi

  persistent_paths=("$DEBUG_DEPO_SCRATCH")
  for path in "${persistent_paths[@]}"; do
    path_device="$(findmnt -n -o MAJ:MIN -T "$path" 2>/dev/null || true)"
    if [[ "$path_device" != "$persistent_device" ]]; then
      echo "Persistent artifact path is not on $CLOUD_PERSISTENT_ROOT: $path" >&2
      return 2
    fi
  done

  ephemeral_paths=(
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
  )
  for path in "${ephemeral_paths[@]}"; do
    path_device="$(findmnt -n -o MAJ:MIN -T "$path" 2>/dev/null || true)"
    if [[ "$path_device" != "$ephemeral_device" ]]; then
      echo "Rebuildable cache/SIF path is not on $CLOUD_EPHEMERAL_ROOT: $path" >&2
      return 2
    fi
  done

  vllm_image_device="$(findmnt -n -o MAJ:MIN -T "$(dirname "$VLLM_IMAGE")" 2>/dev/null || true)"
  if [[ "$vllm_image_device" != "$persistent_device" && \
    "$vllm_image_device" != "$ephemeral_device" ]]; then
    echo "VLLM_IMAGE must be on configured persistent or local VM storage: $VLLM_IMAGE" >&2
    return 2
  fi
}

wait_for_vllm() {
  local url="$1"
  local process_id="$2"
  local log_path="$3"
  local timeout="${VLLM_STARTUP_TIMEOUT:-7200}"
  local deadline=$((SECONDS + timeout))
  local process_status
  until curl --fail --silent --max-time 2 "$url/models" >/dev/null; do
    if ! kill -0 "$process_id" 2>/dev/null; then
      if wait "$process_id"; then
        process_status=1
      else
        process_status=$?
      fi
      echo "vLLM exited before becoming ready; see $log_path" >&2
      return "$process_status"
    fi
    if ((SECONDS >= deadline)); then
      echo "vLLM did not become ready at $url; see $log_path" >&2
      return 1
    fi
    sleep 5
  done
}

file_mtime_epoch() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    printf '0\n'
  elif stat -c '%Y' "$path" >/dev/null 2>&1; then
    stat -c '%Y' "$path"
  else
    stat -f '%m' "$path"
  fi
}

latest_activity_epoch() {
  local latest=0
  local path
  local modified
  for path in "$@"; do
    modified="$(file_mtime_epoch "$path")"
    if ((modified > latest)); then
      latest="$modified"
    fi
  done
  printf '%s\n' "$latest"
}

# Return distinct codes so callers can decide whether to restart a shard.
CLOUD_SUPERVISOR_VLLM_EXIT=70
CLOUD_SUPERVISOR_STALL=71

supervise_collector() {
  local vllm_pid="$1"
  local collector_pid="$2"
  local stall_timeout_seconds="$3"
  local interval_seconds="$4"
  shift 4
  local -a activity_paths=("$@")
  local last_activity
  local observed_activity
  local now

  last_activity="$(date +%s)"
  while kill -0 "$collector_pid" 2>/dev/null; do
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      echo "vLLM exited while its collector was still running." >&2
      return "$CLOUD_SUPERVISOR_VLLM_EXIT"
    fi

    observed_activity="$(latest_activity_epoch "${activity_paths[@]}")"
    if ((observed_activity > last_activity)); then
      last_activity="$observed_activity"
    fi
    now="$(date +%s)"
    if ((stall_timeout_seconds > 0 && now - last_activity >= stall_timeout_seconds)); then
      echo "Shard made no collector progress for ${stall_timeout_seconds}s." >&2
      return "$CLOUD_SUPERVISOR_STALL"
    fi
    sleep "$interval_seconds"
  done

  wait "$collector_pid"
}

terminate_process_group() {
  local process_id="$1"
  local grace_seconds="${2:-10}"
  local deadline=$((SECONDS + grace_seconds))

  kill -TERM -- "-$process_id" 2>/dev/null || kill -TERM "$process_id" 2>/dev/null || true
  while kill -0 "$process_id" 2>/dev/null && ((SECONDS < deadline)); do
    sleep 1
  done
  if kill -0 "$process_id" 2>/dev/null; then
    kill -KILL -- "-$process_id" 2>/dev/null || kill -KILL "$process_id" 2>/dev/null || true
  fi
  wait "$process_id" 2>/dev/null || true
}

cleanup_shard_tmp() {
  local tmp_root="$1"
  local shard_tmp="$2"
  if [[ -z "$tmp_root" || "$tmp_root" == "/" || "$shard_tmp" != "$tmp_root"/collection-shard-* ]]; then
    echo "Refusing unsafe shard tmp cleanup: $shard_tmp" >&2
    return 2
  fi
  mkdir -p "$shard_tmp"
  find "$shard_tmp" -mindepth 1 -maxdepth 1 -type d -name 'minisweagent-*' \
    -exec rm -rf -- {} +
  find "$shard_tmp" -mindepth 1 -maxdepth 1 -type s -delete
}

available_gib() {
  df -Pk "$1" | awk 'NR == 2 {print int($4 / 1024 / 1024)}'
}

unset CLOUD_COMMON_DIR
