#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$CLOUD_DIR/common.sh"

require_command apptainer
require_command setsid
require_separate_storage

GPU_ID="${GPU_ID:?Set GPU_ID to one physical GPU index.}"
PORT="${PORT:?Set PORT to the vLLM port for this shard.}"
VLLM_MODEL="${VLLM_MODEL:-${AGENTFORGE_MODEL:-$BASELINE_SFT_MODEL}}"
VLLM_MODEL_REVISION="${VLLM_MODEL_REVISION:-}"
if [[ -z "$VLLM_MODEL_REVISION" && "$VLLM_MODEL" == "$BASELINE_SFT_MODEL" ]]; then
  VLLM_MODEL_REVISION="$BASELINE_SFT_MODEL_REVISION"
fi
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$VLLM_MODEL}"
if [[ "$SERVED_MODEL_NAME" == hosted_vllm/* ]]; then
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME#hosted_vllm/}"
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${CONTEXT_LENGTH:-65536}}"
RUN_NAME="${RUN_NAME:-manual}"
SHARD_INDEX="${SHARD_INDEX:-$GPU_ID}"

if [[ -z "${HF_TOKEN:-}" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(<"$HF_TOKEN_FILE")"
  export HF_TOKEN
fi

if ! apptainer inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
  echo "vLLM Apptainer image is missing or invalid: $VLLM_IMAGE" >&2
  echo "Run bash cloud/setup.sh first." >&2
  exit 2
fi

bind_args=(
  --bind "$DEBUG_DEPO_ROOT:$DEBUG_DEPO_ROOT"
  --bind "$CLOUD_PERSISTENT_ROOT:$CLOUD_PERSISTENT_ROOT"
  --bind "$DEBUG_DEPO_EPHEMERAL:$DEBUG_DEPO_EPHEMERAL"
)

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$GPU_ID"
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export APPTAINERENV_HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export APPTAINERENV_RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
  export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

read -r -a vllm_extra_args <<<"${VLLM_EXTRA_ARGS:-}"
vllm_revision_args=()
if [[ -n "$VLLM_MODEL_REVISION" ]]; then
  vllm_revision_args+=(--revision "$VLLM_MODEL_REVISION")
fi
vllm_diagnostic_args=()
if [[ "${VLLM_LOG_REQUESTS:-1}" == "1" ]]; then
  vllm_diagnostic_args+=(--enable-log-requests --max-log-len "${VLLM_MAX_LOG_LEN:-2048}")
fi

setsid apptainer exec --nv \
  "${bind_args[@]}" \
  --pwd "$DEBUG_DEPO_ROOT" \
  "$VLLM_IMAGE" \
  vllm serve "$VLLM_MODEL" \
    "${vllm_revision_args[@]}" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    "${vllm_diagnostic_args[@]}" \
    "${vllm_extra_args[@]}" &
vllm_pid=$!

cleanup() {
  trap - EXIT HUP INT TERM
  terminate_process_group "$vllm_pid"
}
trap cleanup EXIT HUP INT TERM
wait "$vllm_pid"
