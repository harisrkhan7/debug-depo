#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$CLOUD_DIR/common.sh"

require_command nvidia-smi "Use Lambda Stack or GPU Base with the NVIDIA driver installed."
require_command apptainer "Run bash cloud/setup.sh to install Apptainer."
require_command curl
require_project_environment
require_separate_storage
gpu_id_array

if ! apptainer inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
  echo "vLLM Apptainer image is missing or invalid: $VLLM_IMAGE" >&2
  echo "Run bash cloud/setup.sh first." >&2
  exit 2
fi

gpu_indices="$(nvidia-smi --query-gpu=index --format=csv,noheader)"
gpu_count="$(wc -l <<<"$gpu_indices")"
gpu_count="${gpu_count//[[:space:]]/}"
if ((gpu_count < NUM_SHARDS)); then
  echo "Expected at least $NUM_SHARDS GPUs, but nvidia-smi reports $gpu_count." >&2
  exit 2
fi
for gpu_id in "${CLOUD_GPU_ID_ARRAY[@]}"; do
  if ! grep -Eq "^[[:space:]]*$gpu_id[[:space:]]*$" <<<"$gpu_indices"; then
    echo "Configured GPU $gpu_id is not present in the nvidia-smi inventory." >&2
    exit 2
  fi
done
gpu_inventory="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader)"
cpu_count="$(nproc)"
memory_gib="$(awk '/^MemTotal:/ {print int($2 / 1024 / 1024)}' /proc/meminfo)"

root_source="$(findmnt -n -o SOURCE -T "$CLOUD_PERSISTENT_ROOT" 2>/dev/null || true)"
root_type="$(findmnt -n -o FSTYPE -T "$CLOUD_PERSISTENT_ROOT" 2>/dev/null || true)"
root_size="$(df -BG "$CLOUD_PERSISTENT_ROOT" | awk 'NR == 2 {print $2}')"
root_free="$(df -BG "$CLOUD_PERSISTENT_ROOT" | awk 'NR == 2 {print $4}')"
local_source="$(findmnt -n -o SOURCE -T "$CLOUD_EPHEMERAL_ROOT" 2>/dev/null || true)"
local_type="$(findmnt -n -o FSTYPE -T "$CLOUD_EPHEMERAL_ROOT" 2>/dev/null || true)"
local_size="$(df -BG "$CLOUD_EPHEMERAL_ROOT" | awk 'NR == 2 {print $2}')"
local_free="$(df -BG "$CLOUD_EPHEMERAL_ROOT" | awk 'NR == 2 {print $4}')"

cat <<MSG
Lambda Cloud preflight passed.
  repository:       $DEBUG_DEPO_ROOT
  persistent root:  $CLOUD_PERSISTENT_ROOT
  persistent fs:    ${root_source:-unknown} (${root_type:-unknown}), $root_size / $root_free free
  persistent runs:  $DEBUG_DEPO_SCRATCH
  local root:       $CLOUD_EPHEMERAL_ROOT
  local fs:         ${local_source:-unknown} (${local_type:-unknown}), $local_size / $local_free free
  local cache:      $DEBUG_DEPO_CACHE_ROOT
  vLLM SIF:         $VLLM_IMAGE
  CPU/RAM:          $cpu_count vCPUs / ${memory_gib} GiB
  GPUs/shards:      $gpu_count available / $NUM_SHARDS configured ($CLOUD_GPU_SOURCE)
  rollout workers:  $ROLLOUT_WORKERS per shard
  cache workers:    $CACHE_BUILD_MAX_WORKERS
  eval workers:     $EVAL_MAX_WORKERS

GPU inventory:
$gpu_inventory
MSG

root_total_gib="$(df -Pk "$CLOUD_PERSISTENT_ROOT" | awk 'NR == 2 {print int($2 / 1024 / 1024)}')"
if ((root_total_gib < 1000)); then
  cat >&2 <<MSG

WARNING: the persistent filesystem is only about ${root_total_gib} GiB.
Attach a sufficiently large Lambda filesystem when launching the instance and
set CLOUD_PERSISTENT_ROOT to its /lambda/nfs/<name> path before generating runs,
checkpoints, and models. Rebuildable caches and task SIFs remain on local disk.
MSG
fi
