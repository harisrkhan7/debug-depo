#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DATASET="${DATASET:-SWE-bench/SWE-smith-py}"
DATASET_REVISION="${SWESMITH_DATASET_REVISION:-77cab9055d42ab4a5c25c89a8f937096db13558e}"
SPLIT="${SPLIT:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data/processed/swesmith_collection}"
RUNS_PER_TEMPERATURE="${RUNS_PER_TEMPERATURE:-4}"
TEMPERATURES="${TEMPERATURES:-0.6:0.7}"
BASE_SEED="${BASE_SEED:-42}"
MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
MINI_SWE_RUNNER="${MINI_SWE_RUNNER:-singularity}"
MINI_SWE_ENVIRONMENT_CLASS="${MINI_SWE_ENVIRONMENT_CLASS:-singularity}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
LIMIT="${LIMIT:-}"
SIF_DIR="${SWESMITH_APPTAINER_SIF_DIR:-$ROOT_DIR/data/apptainer/swesmith-sifs}"
APPTAINER_CACHE_DIR="${SWESMITH_APPTAINER_CACHE_DIR:-${APPTAINER_CACHEDIR:-}}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/external/mini-swe-agent-plus/src:$ROOT_DIR/external/SWE-smith${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --dataset "$DATASET"
  --dataset-revision "$DATASET_REVISION"
  --split "$SPLIT"
  --output-dir "$OUTPUT_DIR"
  --runs-per-temperature "$RUNS_PER_TEMPERATURE"
  --temperatures "$TEMPERATURES"
  --base-seed "$BASE_SEED"
  --model "$MODEL"
  --mini-runner "$MINI_SWE_RUNNER"
  --rollout-workers "$ROLLOUT_WORKERS"
  --num-shards "$NUM_SHARDS"
  --shard-index "$SHARD_INDEX"
  --require-complete
)

if [[ -n "$MINI_SWE_ENVIRONMENT_CLASS" ]]; then
  args+=(--mini-environment-class "$MINI_SWE_ENVIRONMENT_CLASS")
fi
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ -n "${EXPECTED_TASKS:-}" ]]; then
  args+=(--expected-tasks "$EXPECTED_TASKS")
fi
if [[ -n "${TASK_IDS_FILE:-}" ]]; then
  args+=(--instance-ids-file "$TASK_IDS_FILE")
fi
if [[ -n "${MINI_SWE_MODEL:-}" ]]; then
  args+=(--mini-model "$MINI_SWE_MODEL")
fi
if [[ -n "${MINI_SWE_CONFIG:-}" ]]; then
  args+=(--mini-config "$MINI_SWE_CONFIG")
fi
if [[ -n "${MINI_SWE_WORKERS:-}" ]]; then
  args+=(--mini-workers "$MINI_SWE_WORKERS")
fi
if [[ -n "${LLM_BASE_URL:-}" ]]; then
  args+=(--llm-base-url "$LLM_BASE_URL")
fi
if [[ -n "${LLM_API_KEY:-}" ]]; then
  args+=(--llm-api-key "$LLM_API_KEY")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  args+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "${CONTEXT_LENGTH:-}" ]]; then
  args+=(--context-length "$CONTEXT_LENGTH")
fi
if [[ -n "${TOP_P:-}" ]]; then
  args+=(--top-p "$TOP_P")
fi
if [[ -n "${TIMEOUT_SECONDS:-}" ]]; then
  args+=(--timeout-seconds "$TIMEOUT_SECONDS")
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

if [[ "${MOCK:-0}" != "1" ]]; then
  if [[ "$MINI_SWE_RUNNER" == "singularity" || "$MINI_SWE_ENVIRONMENT_CLASS" == "singularity" ]]; then
    executable="${MSWEA_SINGULARITY_EXECUTABLE:-apptainer}"
    if ! command -v "$executable" >/dev/null 2>&1; then
      echo "SWE-smith collection requires '$executable' for task environments." >&2
      exit 127
    fi
    export MSWEA_SINGULARITY_EXECUTABLE="$executable"
    export MSWEA_SINGULARITY_SIF_DIR="${MSWEA_SINGULARITY_SIF_DIR:-$SIF_DIR}"
    if [[ -n "$APPTAINER_CACHE_DIR" ]]; then
      export MSWEA_SINGULARITY_CACHE_DIR="${MSWEA_SINGULARITY_CACHE_DIR:-$APPTAINER_CACHE_DIR}"
    fi
  elif ! command -v docker >/dev/null 2>&1; then
    echo "SWE-smith collection requires Docker or MINI_SWE_RUNNER=singularity." >&2
    exit 127
  fi
fi

"$UV_BIN" run python -m debug_depo.swesmith_collect "${args[@]}" "$@"
