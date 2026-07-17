#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DEBUG_DEPO_ROOT="${DEBUG_DEPO_ROOT:-$ROOT_DIR}"
source "$ROOT_DIR/cluster/env/load.sh"

VLLM_IMAGE="${VLLM_IMAGE:-$ROOT_DIR/cluster/apptainer/vllm-openai.sif}"
VLLM_MODEL="${VLLM_MODEL:-${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-7200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${CONTEXT_LENGTH:-65536}}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
HF_HOME="${HF_HOME:-$DEBUG_DEPO_SCRATCH/huggingface}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/debug-depo/hf_token}"
read -r -a VLLM_EXTRA_ARGS_ARRAY <<< "${VLLM_EXTRA_ARGS:-}"

if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(<"$HF_TOKEN_FILE")"
  export HF_TOKEN
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [[ -n "${MINI_SWE_MODEL:-}" && "$MINI_SWE_MODEL" == hosted_vllm/* ]]; then
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MINI_SWE_MODEL#hosted_vllm/}}"
else
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$VLLM_MODEL}"
fi

if [[ "$VLLM_IMAGE" != docker://* && "$VLLM_IMAGE" != oras://* && ! -f "$VLLM_IMAGE" ]]; then
  cat >&2 <<MSG
Missing VLLM_IMAGE: $VLLM_IMAGE
Build it first, for example:
  apptainer pull "$VLLM_IMAGE" docker://vllm/vllm-openai:latest
MSG
  exit 1
fi

mkdir -p "$HF_HOME"

bind_args=(--bind "$ROOT_DIR:$ROOT_DIR" --bind "$HF_HOME:$HF_HOME")
if [[ -n "${DEBUG_DEPO_SCRATCH:-}" ]]; then
  mkdir -p "$DEBUG_DEPO_SCRATCH"
  bind_args+=(--bind "$DEBUG_DEPO_SCRATCH:$DEBUG_DEPO_SCRATCH")
fi
if [[ -n "${TMPDIR:-}" && -d "$TMPDIR" ]]; then
  bind_args+=(--bind "$TMPDIR:$TMPDIR")
fi

export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
fi
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

apptainer exec --nv \
  "${bind_args[@]}" \
  --pwd "$ROOT_DIR" \
  "$VLLM_IMAGE" \
  vllm serve "$VLLM_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    "${VLLM_EXTRA_ARGS_ARRAY[@]}" \
    "$@" &
server_pid=$!

cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT HUP INT TERM

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --fail --silent --max-time 2 "http://$HOST:$PORT/v1/models" >/dev/null; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    exit $?
  fi
  if ((SECONDS >= deadline)); then
    echo "vLLM did not become ready within ${STARTUP_TIMEOUT}s." >&2
    exit 1
  fi
  sleep 5
done

echo "vLLM ready at http://$HOST:$PORT/v1 for $SERVED_MODEL_NAME."
wait "$server_pid"
