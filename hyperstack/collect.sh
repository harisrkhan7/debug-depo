#!/usr/bin/env bash
set -euo pipefail

HYPERSTACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HYPERSTACK_DIR/common.sh"

FAMILY="${1:-verified}"
case "$FAMILY" in
  verified)
    RUN_NAME="${RUN_NAME:-agentforge-verified-hyperstack}"
    EXPECTED_TASKS="${EXPECTED_TASKS:-500}"
    DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
    SPLIT="${SPLIT:-test}"
    TASK_IDS_FILE="${TASK_IDS_FILE:-}"
    ;;
  swesmith)
    RUN_NAME="${RUN_NAME:-swesmith-train-1000}"
    EXPECTED_TASKS="${EXPECTED_TASKS:-1000}"
    DATASET="${DATASET:-SWE-bench/SWE-smith-py}"
    SPLIT="${SPLIT:-train}"
    TASK_IDS_FILE="${TASK_IDS_FILE:-data/splits/swesmith_train_1000_instance_ids.txt}"
    ;;
  *)
    echo "Usage: bash hyperstack/collect.sh verified|swesmith" >&2
    exit 2
    ;;
esac

require_run_name "$RUN_NAME"
require_positive_integer NUM_SHARDS "$NUM_SHARDS"
require_positive_integer ROLLOUT_WORKERS "$ROLLOUT_WORKERS"
require_positive_integer EXPECTED_TASKS "$EXPECTED_TASKS"
if ((NUM_SHARDS != 8)); then
  echo "HyperStack collection is configured for exactly 8 shards/GPUs; got NUM_SHARDS=$NUM_SHARDS." >&2
  exit 2
fi
if ((NUM_SHARDS > EXPECTED_TASKS)); then
  echo "NUM_SHARDS ($NUM_SHARDS) cannot exceed EXPECTED_TASKS ($EXPECTED_TASKS)." >&2
  exit 2
fi
gpu_id_array

RUN_ROOT="${RUN_ROOT:-$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/hyperstack-logs}"
AGENTFORGE_MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
MINI_SWE_MODEL="${MINI_SWE_MODEL:-hosted_vllm/$AGENTFORGE_MODEL}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
MAX_STEPS="${MAX_STEPS:-200}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"
VLLM_MODEL="${VLLM_MODEL:-$AGENTFORGE_MODEL}"

if [[ "${MINI_SWE_RUNNER:-singularity}" != "singularity" ]]; then
  echo "HyperStack requires MINI_SWE_RUNNER=singularity for Apptainer execution." >&2
  exit 2
fi
if [[ "${MINI_SWE_ENVIRONMENT_CLASS:-singularity}" != "singularity" ]]; then
  echo "HyperStack requires MINI_SWE_ENVIRONMENT_CLASS=singularity." >&2
  exit 2
fi
if [[ "${MSWEA_SINGULARITY_EXECUTABLE:-apptainer}" != "apptainer" ]]; then
  echo "HyperStack requires MSWEA_SINGULARITY_EXECUTABLE=apptainer." >&2
  exit 2
fi
if [[ "$FAMILY" == "verified" && "${HARNESS:-mini-swe-agent-plus}" != "mini-swe-agent-plus" ]]; then
  echo "HyperStack Verified collection requires HARNESS=mini-swe-agent-plus." >&2
  exit 2
fi

if [[ -n "$TASK_IDS_FILE" && ! -s "$DEBUG_DEPO_ROOT/$TASK_IDS_FILE" && ! -s "$TASK_IDS_FILE" ]]; then
  echo "Task-ID file is missing or empty: $TASK_IDS_FILE" >&2
  exit 2
fi

cat <<MSG
HyperStack $FAMILY trajectory collection
  run:                 $RUN_NAME
  run root:            $RUN_ROOT
  dataset/split:       $DATASET / $SPLIT
  expected tasks:      $EXPECTED_TASKS
  shards/GPUs:         $NUM_SHARDS / $GPU_IDS
  workers per shard:   $ROLLOUT_WORKERS
  total worker slots:  $((NUM_SHARDS * ROLLOUT_WORKERS))
  model:               $AGENTFORGE_MODEL
  task IDs:            ${TASK_IDS_FILE:-all selected dataset tasks}
MSG

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run: no servers or collectors started."
  exit 0
fi

require_separate_storage
if ((EXPECTED_TASKS >= 100)) && [[ "${SKIP_STORAGE_CHECK:-0}" != "1" ]]; then
  if [[ "$FAMILY" == "swesmith" && "$EXPECTED_TASKS" -ge 5000 ]]; then
    default_minimum_free_gib=1000
  else
    default_minimum_free_gib=500
  fi
  minimum_free_gib="${MIN_COLLECTION_FREE_GIB:-$default_minimum_free_gib}"
  free_gib="$(available_gib "$DEBUG_DEPO_SCRATCH")"
  if ((free_gib < minimum_free_gib)); then
    cat >&2 <<MSG
Only ${free_gib} GiB is free under $DEBUG_DEPO_SCRATCH.
This collection requires at least the conservative ${minimum_free_gib} GiB
guard. Mount a sufficiently large persistent volume for that path, or set
SKIP_STORAGE_CHECK=1 only after independently confirming capacity.
MSG
    exit 2
  fi
fi

require_command curl
require_command apptainer
require_project_environment

cd "$DEBUG_DEPO_ROOT"
mkdir -p "$LOG_DIR"

shard_pids=()
cleanup_shards() {
  if ((${#shard_pids[@]})); then
    kill "${shard_pids[@]}" 2>/dev/null || true
    wait "${shard_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup_shards HUP INT TERM

run_shard() {
  local shard_index="$1"
  local gpu_id="$2"
  local port=$((VLLM_PORT_BASE + shard_index))
  local output_dir
  if [[ "$FAMILY" == "verified" ]]; then
    output_dir="$RUN_ROOT/rollouts/shard-$shard_index"
  else
    output_dir="$RUN_ROOT/collection/shard-$shard_index"
  fi
  local vllm_log="$output_dir/vllm.log"
  mkdir -p "$output_dir"

  GPU_ID="$gpu_id" \
  PORT="$port" \
  RUN_NAME="$RUN_NAME" \
  SHARD_INDEX="$shard_index" \
  VLLM_MODEL="$VLLM_MODEL" \
  AGENTFORGE_MODEL="$AGENTFORGE_MODEL" \
  MINI_SWE_MODEL="$MINI_SWE_MODEL" \
  CONTEXT_LENGTH="$CONTEXT_LENGTH" \
    bash "$HYPERSTACK_DIR/serve_vllm.sh" >"$vllm_log" 2>&1 &
  local vllm_pid=$!

  stop_vllm() {
    local process_id="$1"
    kill "$process_id" 2>/dev/null || true
    wait "$process_id" 2>/dev/null || true
  }
  trap "stop_vllm $vllm_pid" EXIT HUP INT TERM

  local llm_base_url="http://127.0.0.1:$port/v1"
  wait_for_vllm "$llm_base_url" "$vllm_pid" "$vllm_log"

  export RUN_NAME RUN_ROOT OUTPUT_DIR="$output_dir"
  export NUM_SHARDS SHARD_INDEX="$shard_index"
  export DATASET SPLIT TASK_IDS_FILE EXPECTED_TASKS
  export AGENTFORGE_MODEL MINI_SWE_MODEL CONTEXT_LENGTH MAX_STEPS TIMEOUT_SECONDS
  export LLM_BASE_URL="$llm_base_url"
  export LLM_API_KEY="${LLM_API_KEY:-local}"
  export ROLLOUT_WORKERS
  export MINI_SWE_WORKERS="${MINI_SWE_WORKERS:-1}"
  export MINI_SWE_RUNNER=singularity
  export MINI_SWE_ENVIRONMENT_CLASS=singularity
  export MSWEA_SINGULARITY_EXECUTABLE=apptainer
  export STREAM_OUTPUT="${STREAM_OUTPUT:-0}"
  export CUDA_VISIBLE_DEVICES="$gpu_id"

  echo "Shard $shard_index is using GPU $gpu_id and $llm_base_url."
  if [[ "$FAMILY" == "verified" ]]; then
    export HARNESS=mini-swe-agent-plus
    export MINI_SWE_IMAGE_TEMPLATE="${MINI_SWE_IMAGE_TEMPLATE:-docker://ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest}"
    scripts/collect_rollouts.sh
  else
    export RUNS_PER_TEMPERATURE="${RUNS_PER_TEMPERATURE:-4}"
    export TEMPERATURES="${TEMPERATURES:-0.6:0.7}"
    export BASE_SEED="${BASE_SEED:-42}"
    scripts/collect_swesmith.sh
  fi
}

for ((shard_index = 0; shard_index < NUM_SHARDS; shard_index++)); do
  gpu_id="${HYPERSTACK_GPU_ID_ARRAY[$shard_index]}"
  collector_log="$LOG_DIR/collect-$FAMILY-shard-$shard_index.log"
  run_shard "$shard_index" "$gpu_id" >"$collector_log" 2>&1 &
  shard_pids+=("$!")
  echo "Started shard $shard_index on GPU $gpu_id (log: $collector_log)"
done

failed=0
for shard_index in "${!shard_pids[@]}"; do
  if ! wait "${shard_pids[$shard_index]}"; then
    echo "Shard $shard_index failed; see $LOG_DIR/collect-$FAMILY-shard-$shard_index.log" >&2
    failed=1
  fi
done
shard_pids=()
trap - HUP INT TERM

if ((failed)); then
  exit 1
fi
echo "All $NUM_SHARDS $FAMILY collection shards completed: $RUN_ROOT"
