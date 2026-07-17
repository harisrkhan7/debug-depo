#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  cat >&2 <<MSG
Could not find uv executable: $UV_BIN

Install uv in the Python environment used for this run, for example:
  python -m pip install -U uv

On the cluster JupyterHub kernel, use the kernel Python explicitly:
  "$HOME/miniforge3/envs/debug-depo/bin/python" -m pip install -U uv

Or set UV to the full path:
  UV=/path/to/env/bin/uv scripts/collect_rollouts.sh
MSG
  exit 127
fi

DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data/processed/agentforge_swebench_verified}"
MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
HARNESS="${HARNESS:-command}"
MINI_SWE_RUNNER="${MINI_SWE_RUNNER:-pool_way}"
MINI_SWE_ENVIRONMENT_CLASS="${MINI_SWE_ENVIRONMENT_CLASS:-}"
MAX_STEPS="${MAX_STEPS:-200}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}"
LIMIT="${LIMIT:-}"
START_INDEX="${START_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-1}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/debug-depo/hf_token}"

if [[ -z "${HF_TOKEN:-}" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(<"$HF_TOKEN_FILE")"
  export HF_TOKEN
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --dataset "$DATASET"
  --split "$SPLIT"
  --output-dir "$OUTPUT_DIR"
  --model "$MODEL"
  --harness "$HARNESS"
  --max-steps "$MAX_STEPS"
  --context-length "$CONTEXT_LENGTH"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --timeout-seconds "$TIMEOUT_SECONDS"
  --start-index "$START_INDEX"
  --num-shards "$NUM_SHARDS"
  --shard-index "$SHARD_INDEX"
  --rollout-workers "$ROLLOUT_WORKERS"
)

if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ -n "${TASK_IDS_FILE:-}" ]]; then
  args+=(--instance-ids-file "$TASK_IDS_FILE")
fi
if [[ -n "${AGENTFORGE_COMMAND:-}" ]]; then
  args+=(--agentforge-command "$AGENTFORGE_COMMAND")
fi
if [[ -n "${AGENTFORGE_REPO:-}" ]]; then
  args+=(--agentforge-cwd "$AGENTFORGE_REPO")
fi
if [[ -n "${MINI_SWE_MODEL:-}" ]]; then
  args+=(--mini-model "$MINI_SWE_MODEL")
fi
if [[ -n "${MINI_SWE_CONFIG:-}" ]]; then
  args+=(--mini-config "$MINI_SWE_CONFIG")
fi
args+=(--mini-runner "$MINI_SWE_RUNNER")
if [[ -n "$MINI_SWE_ENVIRONMENT_CLASS" ]]; then
  args+=(--mini-environment-class "$MINI_SWE_ENVIRONMENT_CLASS")
fi
if [[ -n "${MINI_SWE_WORKERS:-}" ]]; then
  args+=(--mini-workers "$MINI_SWE_WORKERS")
fi
if [[ -n "${MINI_SWE_DOCKER_START_CONCURRENCY:-}" ]]; then
  args+=(--mini-docker-start-concurrency "$MINI_SWE_DOCKER_START_CONCURRENCY")
fi
if [[ -n "${LLM_BASE_URL:-}" ]]; then
  args+=(--llm-base-url "$LLM_BASE_URL")
fi
if [[ -n "${LLM_API_KEY:-}" ]]; then
  args+=(--llm-api-key "$LLM_API_KEY")
fi
if [[ "${MOCK:-0}" == "1" ]]; then
  args+=(--mock --mock-patch "${MOCK_PATCH:-empty}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "${STREAM_OUTPUT:-0}" == "1" ]]; then
  args+=(--stream-output)
fi

uses_miniswe_singularity=0
if [[ "$MINI_SWE_RUNNER" == "singularity" || "$MINI_SWE_ENVIRONMENT_CLASS" == "singularity" ]]; then
  uses_miniswe_singularity=1
fi

if [[ "$HARNESS" == "mini-swe-agent-plus" && "$uses_miniswe_singularity" == "1" ]]; then
  singularity_executable="${MSWEA_SINGULARITY_EXECUTABLE:-apptainer}"
  if ! command -v "$singularity_executable" >/dev/null 2>&1; then
    cat >&2 <<MSG
mini-swe-agent-plus is configured for Singularity/Apptainer task containers,
but '$singularity_executable' was not found on PATH.

Set the executable explicitly if your cluster uses a different command:
  MSWEA_SINGULARITY_EXECUTABLE=/path/to/apptainer scripts/collect_rollouts.sh
MSG
    exit 127
  fi
  export MSWEA_SINGULARITY_EXECUTABLE="$singularity_executable"
  if [[ -n "${DEBUG_DEPO_SCRATCH:-}" ]]; then
    export TMPDIR="${TMPDIR:-$DEBUG_DEPO_SCRATCH/tmp}"
    mkdir -p "$TMPDIR"
  fi
elif [[ "$HARNESS" == "mini-swe-agent-plus" && "${SKIP_DOCKER_PREFLIGHT:-0}" != "1" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<MSG
mini-swe-agent-plus SWE-bench rollouts are configured for Docker task
containers, but 'docker' was not found on PATH.

This failure happens before any LLM call: mini-swe-agent-plus starts each
SWE-bench task environment through DockerEnvironment. The Apptainer vLLM server
only serves the model; it does not replace Docker for the agent task container.

On an Apptainer-only cluster, use the mini-swe-agent-plus Singularity backend:
  MINI_SWE_RUNNER=singularity scripts/collect_rollouts.sh

To bypass this preflight anyway:
  SKIP_DOCKER_PREFLIGHT=1 scripts/collect_rollouts.sh
MSG
    exit 127
  fi
  if ! docker info >/dev/null 2>&1; then
    cat >&2 <<MSG
mini-swe-agent-plus found 'docker', but the Docker daemon is not reachable.

This cluster session likely does not provide Docker access. The current
mini-swe-agent-plus rollout path needs Docker for SWE-bench task containers.
MSG
    exit 2
  fi
fi

"$UV_BIN" run python -m debug_depo.rollout "${args[@]}" "$@"
