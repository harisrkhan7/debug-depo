#!/usr/bin/env bash
set -euo pipefail

HYPERSTACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HYPERSTACK_DIR/common.sh"

require_command nvidia-smi "Use a HyperStack CUDA image with the NVIDIA driver installed."
require_command apptainer "Run bash hyperstack/setup.sh to install Apptainer."
require_command curl
require_project_environment
require_separate_storage
gpu_id_array

if ! apptainer inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
  echo "vLLM Apptainer image is missing or invalid: $VLLM_IMAGE" >&2
  echo "Run bash hyperstack/setup.sh first." >&2
  exit 2
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
gpu_count="${gpu_count//[[:space:]]/}"
if ((gpu_count < NUM_SHARDS)); then
  echo "Expected at least $NUM_SHARDS GPUs, but nvidia-smi reports $gpu_count." >&2
  exit 2
fi
gpu_inventory="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader)"
cpu_count="$(nproc)"
memory_gib="$(awk '/^MemTotal:/ {print int($2 / 1024 / 1024)}' /proc/meminfo)"

root_source="$(findmnt -n -o SOURCE -T "$HYPERSTACK_PERSISTENT_ROOT" 2>/dev/null || true)"
root_type="$(findmnt -n -o FSTYPE -T "$HYPERSTACK_PERSISTENT_ROOT" 2>/dev/null || true)"
root_size="$(df -BG "$HYPERSTACK_PERSISTENT_ROOT" | awk 'NR == 2 {print $2}')"
root_free="$(df -BG "$HYPERSTACK_PERSISTENT_ROOT" | awk 'NR == 2 {print $4}')"
ephemeral_source="$(findmnt -n -o SOURCE -T "$HYPERSTACK_EPHEMERAL_ROOT" 2>/dev/null || true)"
ephemeral_type="$(findmnt -n -o FSTYPE -T "$HYPERSTACK_EPHEMERAL_ROOT" 2>/dev/null || true)"
ephemeral_size="$(df -BG "$HYPERSTACK_EPHEMERAL_ROOT" | awk 'NR == 2 {print $2}')"
ephemeral_free="$(df -BG "$HYPERSTACK_EPHEMERAL_ROOT" | awk 'NR == 2 {print $4}')"

cat <<MSG
HyperStack preflight passed.
  repository:       $DEBUG_DEPO_ROOT
  persistent root:  $HYPERSTACK_PERSISTENT_ROOT
  persistent fs:    ${root_source:-unknown} (${root_type:-unknown}), $root_size / $root_free free
  persistent runs:  $DEBUG_DEPO_SCRATCH
  ephemeral root:   $HYPERSTACK_EPHEMERAL_ROOT
  ephemeral fs:     ${ephemeral_source:-unknown} (${ephemeral_type:-unknown}), $ephemeral_size / $ephemeral_free free
  ephemeral cache:  $DEBUG_DEPO_CACHE_ROOT
  vLLM SIF:         $VLLM_IMAGE
  CPU/RAM:          $cpu_count vCPUs / ${memory_gib} GiB
  GPUs/shards:      $gpu_count available / $NUM_SHARDS configured
  rollout workers:  $ROLLOUT_WORKERS per shard
  cache workers:    $CACHE_BUILD_MAX_WORKERS
  eval workers:     $EVAL_MAX_WORKERS

GPU inventory:
$gpu_inventory
MSG

if ((cpu_count < 192 || memory_gib < 1700)); then
  echo "WARNING: this VM is smaller than the expected 192-vCPU / ~1.9-TB host." >&2
fi

root_total_gib="$(df -Pk "$HYPERSTACK_PERSISTENT_ROOT" | awk 'NR == 2 {print int($2 / 1024 / 1024)}')"
if ((root_total_gib < 1000)); then
  cat >&2 <<MSG

WARNING: the persistent filesystem is only about ${root_total_gib} GiB.
Attach a sufficiently large Shared Storage Volume at
$HYPERSTACK_PERSISTENT_ROOT before generating runs, checkpoints, and models.
Rebuildable caches and task SIFs remain on the ephemeral disk.
MSG
fi
