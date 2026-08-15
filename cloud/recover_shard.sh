#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$CLOUD_DIR/common.sh"

FAMILY="${1:-swesmith}"
SHARD_INDEX="${2:-}"
if [[ "$FAMILY" != "swesmith" || -z "$SHARD_INDEX" ]]; then
  echo "Usage: bash cloud/recover_shard.sh swesmith SHARD_INDEX" >&2
  exit 2
fi
require_nonnegative_integer SHARD_INDEX "$SHARD_INDEX"
require_positive_integer NUM_SHARDS "$NUM_SHARDS"
if ((SHARD_INDEX >= NUM_SHARDS)); then
  echo "SHARD_INDEX must be less than NUM_SHARDS ($NUM_SHARDS)." >&2
  exit 2
fi

RUN_NAME="${RUN_NAME:-swesmith-train-1000}"
EXPECTED_TASKS="${EXPECTED_TASKS:-1000}"
DATASET="${DATASET:-SWE-bench/SWE-smith-py}"
SPLIT="${SPLIT:-train}"
TASK_IDS_FILE="${TASK_IDS_FILE:-data/splits/swesmith_train_1000_instance_ids.txt}"
RUN_ROOT="${RUN_ROOT:-$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME}"
OUTPUT_DIR="$RUN_ROOT/collection/shard-$SHARD_INDEX"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/cloud-logs}"
AGENTFORGE_MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
VLLM_MODEL="${VLLM_MODEL:-$AGENTFORGE_MODEL}"
MINI_SWE_MODEL="${MINI_SWE_MODEL:-hosted_vllm/$AGENTFORGE_MODEL}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_STEPS="${MAX_STEPS:-200}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"
RUNS_PER_TEMPERATURE="${RUNS_PER_TEMPERATURE:-4}"
TEMPERATURES="${TEMPERATURES:-0.6:0.7}"
BASE_SEED="${BASE_SEED:-42}"
RECOVERY_GPU_IDS="${RECOVERY_GPU_IDS:-$GPU_IDS}"
RECOVERY_WORKERS_PER_GPU="${RECOVERY_WORKERS_PER_GPU:-$ROLLOUT_WORKERS}"
RECOVERY_RUN_ID="${RECOVERY_RUN_ID:-$(date -u '+%Y%m%dT%H%M%SZ')-$$}"

require_run_name "$RUN_NAME"
require_run_name "$RECOVERY_RUN_ID"
require_positive_integer EXPECTED_TASKS "$EXPECTED_TASKS"
require_positive_integer RECOVERY_WORKERS_PER_GPU "$RECOVERY_WORKERS_PER_GPU"
require_model_timeout_watchdog_compatibility
if [[ -n "$TASK_IDS_FILE" && ! -s "$DEBUG_DEPO_ROOT/$TASK_IDS_FILE" && ! -s "$TASK_IDS_FILE" ]]; then
  echo "Task-ID file is missing or empty: $TASK_IDS_FILE" >&2
  exit 2
fi

normalized_recovery_gpu_ids="${RECOVERY_GPU_IDS//,/ }"
read -r -a recovery_gpu_ids <<<"$normalized_recovery_gpu_ids"
recovery_replicas="${#recovery_gpu_ids[@]}"
if ((recovery_replicas < 1)); then
  echo "RECOVERY_GPU_IDS must contain at least one GPU ID." >&2
  exit 2
fi
seen_recovery_gpu_ids=" "
for gpu_id in "${recovery_gpu_ids[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "RECOVERY_GPU_IDS must contain non-negative integers, got: $gpu_id" >&2
    exit 2
  fi
  if [[ "$seen_recovery_gpu_ids" == *" $gpu_id "* ]]; then
    echo "RECOVERY_GPU_IDS must not contain duplicate IDs: $RECOVERY_GPU_IDS" >&2
    exit 2
  fi
  seen_recovery_gpu_ids+="$gpu_id "
done
unset normalized_recovery_gpu_ids seen_recovery_gpu_ids

cat <<MSG
Cloud SWE-smith tail recovery
  run:                  $RUN_NAME
  logical shard:        $SHARD_INDEX of $NUM_SHARDS
  recovery run:         $RECOVERY_RUN_ID
  physical GPUs:        $RECOVERY_GPU_IDS
  GPU replicas:         $recovery_replicas
  workers per GPU:      $RECOVERY_WORKERS_PER_GPU
  total worker slots:   $((recovery_replicas * RECOVERY_WORKERS_PER_GPU))
  output:               $OUTPUT_DIR
  stall watchdog:       ${CLOUD_SHARD_STALL_TIMEOUT_SECONDS}s
  model timeout:        ${MINI_SWE_MODEL_TIMEOUT_SECONDS:-LiteLLM default}
MSG

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run: no recovery servers or collectors started."
  exit 0
fi

if [[ ! -s "$OUTPUT_DIR/collection_manifest.json" ]]; then
  echo "Recovery requires an existing shard manifest: $OUTPUT_DIR/collection_manifest.json" >&2
  exit 2
fi

require_separate_storage
require_command curl
require_command apptainer
require_command setsid
require_command nvidia-smi
require_project_environment

gpu_inventory="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')"
for gpu_id in "${recovery_gpu_ids[@]}"; do
  if ! grep -qx "$gpu_id" <<<"$gpu_inventory"; then
    echo "Configured recovery GPU $gpu_id is not present in nvidia-smi." >&2
    exit 2
  fi
done
unset gpu_inventory

for replica_index in "${!recovery_gpu_ids[@]}"; do
  recovery_port=$((VLLM_PORT_BASE + replica_index))
  if curl --fail --silent --max-time 1 \
    "http://127.0.0.1:$recovery_port/v1/models" >/dev/null 2>&1; then
    echo "Port $recovery_port already has a vLLM server. Stop the normal collection before recovery." >&2
    exit 2
  fi
done
unset recovery_port

cd "$DEBUG_DEPO_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export RUN_NAME RUN_ROOT OUTPUT_DIR NUM_SHARDS SHARD_INDEX
export DATASET SPLIT TASK_IDS_FILE EXPECTED_TASKS
export AGENTFORGE_MODEL VLLM_MODEL MINI_SWE_MODEL
export CONTEXT_LENGTH MAX_STEPS TIMEOUT_SECONDS
export RUNS_PER_TEMPERATURE TEMPERATURES BASE_SEED
export MINI_SWE_WORKERS="${MINI_SWE_WORKERS:-1}"
export MINI_SWE_RUNNER=singularity
export MINI_SWE_ENVIRONMENT_CLASS=singularity
export MSWEA_SINGULARITY_EXECUTABLE=apptainer
export STREAM_OUTPUT="${STREAM_OUTPUT:-1}"
export LLM_API_KEY="${LLM_API_KEY:-local}"

recovery_pids=()
cleanup_recovery() {
  if ((${#recovery_pids[@]})); then
    kill "${recovery_pids[@]}" 2>/dev/null || true
    wait "${recovery_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup_recovery HUP INT TERM

run_recovery_replica() {
  local replica_index="$1"
  local gpu_id="$2"
  local port=$((VLLM_PORT_BASE + replica_index))
  local llm_base_url="http://127.0.0.1:$port/v1"
  local vllm_log="$OUTPUT_DIR/vllm.recovery-${RECOVERY_RUN_ID}-replica-${replica_index}.log"
  local collector_log="$LOG_DIR/recover-swesmith-shard-${SHARD_INDEX}-replica-${replica_index}.log"
  local events_log="$OUTPUT_DIR/rollout_events.recovery-${RECOVERY_RUN_ID}-replica-${replica_index}.jsonl"
  local active_log="$OUTPUT_DIR/active_rollouts.recovery-${RECOVERY_RUN_ID}-replica-${replica_index}.json"
  local replica_tmp="$TMPDIR/collection-shard-${SHARD_INDEX}-recovery-${RECOVERY_RUN_ID}-${replica_index}"
  local vllm_pid=""
  local collector_pid=""
  local status=1

  cleanup_replica() {
    if [[ -n "$collector_pid" ]] && kill -0 "$collector_pid" 2>/dev/null; then
      terminate_process_group "$collector_pid"
    fi
    collector_pid=""
    if [[ -n "$vllm_pid" ]] && kill -0 "$vllm_pid" 2>/dev/null; then
      terminate_process_group "$vllm_pid"
    fi
    vllm_pid=""
  }
  trap cleanup_replica EXIT HUP INT TERM

  cleanup_shard_tmp "$TMPDIR" "$replica_tmp"
  : >"$collector_log"
  echo "Starting recovery replica $replica_index/$recovery_replicas on GPU $gpu_id." >>"$collector_log"
  GPU_ID="$gpu_id" \
  PORT="$port" \
  RUN_NAME="$RUN_NAME" \
  SHARD_INDEX="$SHARD_INDEX" \
  VLLM_MODEL="$VLLM_MODEL" \
  AGENTFORGE_MODEL="$AGENTFORGE_MODEL" \
  MINI_SWE_MODEL="$MINI_SWE_MODEL" \
  CONTEXT_LENGTH="$CONTEXT_LENGTH" \
    setsid bash "$CLOUD_DIR/serve_vllm.sh" >"$vllm_log" 2>&1 &
  vllm_pid=$!

  if wait_for_vllm "$llm_base_url" "$vllm_pid" "$vllm_log" 2>>"$collector_log"; then
    LLM_BASE_URL="$llm_base_url" \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    TMPDIR="$replica_tmp" \
    ROLLOUT_WORKERS="$RECOVERY_WORKERS_PER_GPU" \
      setsid bash scripts/collect_swesmith.sh \
        --recovery-run-id "$RECOVERY_RUN_ID" \
        --recovery-replicas "$recovery_replicas" \
        --recovery-replica-index "$replica_index" \
        >>"$collector_log" 2>&1 &
    collector_pid=$!
    if supervise_collector \
      "$vllm_pid" \
      "$collector_pid" \
      "$CLOUD_SHARD_STALL_TIMEOUT_SECONDS" \
      "$CLOUD_WATCHDOG_INTERVAL_SECONDS" \
      "$collector_log" "$events_log" >>"$collector_log" 2>&1; then
      status=0
    else
      status=$?
    fi
  else
    status=$?
  fi

  cleanup_replica
  if ((status != 0)) && [[ -f "$active_log" ]]; then
    cp "$active_log" "$OUTPUT_DIR/active_rollouts.failed-${RECOVERY_RUN_ID}-replica-${replica_index}.json"
  fi
  cleanup_shard_tmp "$TMPDIR" "$replica_tmp"
  trap - EXIT HUP INT TERM
  return "$status"
}

for replica_index in "${!recovery_gpu_ids[@]}"; do
  run_recovery_replica "$replica_index" "${recovery_gpu_ids[$replica_index]}" &
  recovery_pids+=("$!")
done

recovery_failed=0
for replica_index in "${!recovery_pids[@]}"; do
  if ! wait "${recovery_pids[$replica_index]}"; then
    echo "Recovery replica $replica_index failed; see $LOG_DIR/recover-swesmith-shard-${SHARD_INDEX}-replica-${replica_index}.log" >&2
    recovery_failed=1
  fi
done
recovery_pids=()
trap - HUP INT TERM
if ((recovery_failed)); then
  echo "Tail recovery was incomplete. Rerun the same command to resume remaining slots." >&2
  exit 1
fi

finalizer_gpu="${recovery_gpu_ids[0]}"
finalizer_port="$VLLM_PORT_BASE"
finalizer_url="http://127.0.0.1:$finalizer_port/v1"
finalizer_vllm_log="$OUTPUT_DIR/vllm.recovery-${RECOVERY_RUN_ID}-finalize.log"
finalizer_collector_log="$LOG_DIR/recover-swesmith-shard-${SHARD_INDEX}-finalize.log"
GPU_ID="$finalizer_gpu" \
PORT="$finalizer_port" \
RUN_NAME="$RUN_NAME" \
SHARD_INDEX="$SHARD_INDEX" \
VLLM_MODEL="$VLLM_MODEL" \
AGENTFORGE_MODEL="$AGENTFORGE_MODEL" \
MINI_SWE_MODEL="$MINI_SWE_MODEL" \
CONTEXT_LENGTH="$CONTEXT_LENGTH" \
  setsid bash "$CLOUD_DIR/serve_vllm.sh" >"$finalizer_vllm_log" 2>&1 &
finalizer_vllm_pid=$!
cleanup_finalizer() {
  if kill -0 "$finalizer_vllm_pid" 2>/dev/null; then
    terminate_process_group "$finalizer_vllm_pid"
  fi
}
trap cleanup_finalizer EXIT HUP INT TERM
wait_for_vllm "$finalizer_url" "$finalizer_vllm_pid" "$finalizer_vllm_log"
: >"$finalizer_collector_log"
LLM_BASE_URL="$finalizer_url" \
CUDA_VISIBLE_DEVICES="$finalizer_gpu" \
  bash scripts/collect_swesmith.sh >"$finalizer_collector_log" 2>&1
cleanup_finalizer
trap - EXIT HUP INT TERM

echo "Shard $SHARD_INDEX recovery and canonical summary finalization complete."
cat <<MSG
Continue evaluation with the same collection geometry:
  RUN_NAME=$RUN_NAME EXPECTED_TASKS=$EXPECTED_TASKS \\
  TASK_IDS_FILE=$TASK_IDS_FILE RUNS_PER_TEMPERATURE=$RUNS_PER_TEMPERATURE \\
  TEMPERATURES=$TEMPERATURES bash cloud/run.sh evaluate swesmith

Then write the analysis:
  RUN_NAME=$RUN_NAME EXPECTED_TASKS=$EXPECTED_TASKS \\
  RUNS_PER_TEMPERATURE=$RUNS_PER_TEMPERATURE TEMPERATURES=$TEMPERATURES \\
  bash cloud/run.sh analyze swesmith
MSG
