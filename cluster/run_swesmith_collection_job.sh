#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PBS_O_WORKDIR:-$ROOT_DIR}"
if [[ ! -f pyproject.toml ]]; then
  echo "Submit this job from the debug-depo repository root." >&2
  exit 2
fi

source cluster/load_job_environment.sh
source cluster/env/load.sh

MODE="${SWESMITH_MODE:-pilot}"
case "$MODE" in
  smoke)
    DEFAULT_RUN_NAME=swesmith-smoke
    DEFAULT_NUM_SHARDS=1
    DEFAULT_LIMIT=2
    DEFAULT_ROLLOUT_WORKERS=2
    ;;
  pilot)
    DEFAULT_RUN_NAME=swesmith-pilot
    DEFAULT_NUM_SHARDS=3
    DEFAULT_LIMIT=30
    DEFAULT_ROLLOUT_WORKERS=5
    ;;
  full)
    DEFAULT_RUN_NAME=swesmith-full
    DEFAULT_NUM_SHARDS=50
    DEFAULT_LIMIT=""
    DEFAULT_ROLLOUT_WORKERS=6
    ;;
  *)
    echo "SWESMITH_MODE must be smoke, pilot, or full, got: $MODE" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
NUM_SHARDS="${NUM_SHARDS:-$DEFAULT_NUM_SHARDS}"
SHARD_INDEX="${PBS_ARRAY_INDEX:-${SHARD_INDEX:-0}}"
if ((NUM_SHARDS < 1 || SHARD_INDEX < 0 || SHARD_INDEX >= NUM_SHARDS)); then
  echo "Invalid shard selection: SHARD_INDEX=$SHARD_INDEX NUM_SHARDS=$NUM_SHARDS" >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/collection/shard-$SHARD_INDEX}"
mkdir -p "$OUTPUT_DIR"

export RUN_NAME RUN_ROOT OUTPUT_DIR NUM_SHARDS SHARD_INDEX
export DATASET="${DATASET:-SWE-bench/SWE-smith-py}"
export SWESMITH_DATASET_REVISION="${SWESMITH_DATASET_REVISION:-77cab9055d42ab4a5c25c89a8f937096db13558e}"
export SPLIT="${SPLIT:-train}"
export LIMIT="${LIMIT:-$DEFAULT_LIMIT}"
export RUNS_PER_TEMPERATURE="${RUNS_PER_TEMPERATURE:-4}"
export TEMPERATURES="${TEMPERATURES:-0.6:0.7}"
export BASE_SEED="${BASE_SEED:-42}"
export AGENTFORGE_MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
export MINI_SWE_MODEL="${MINI_SWE_MODEL:-hosted_vllm/$AGENTFORGE_MODEL}"
export MINI_SWE_RUNNER="${MINI_SWE_RUNNER:-singularity}"
export MINI_SWE_ENVIRONMENT_CLASS="${MINI_SWE_ENVIRONMENT_CLASS:-singularity}"
export MSWEA_SINGULARITY_EXECUTABLE="${MSWEA_SINGULARITY_EXECUTABLE:-apptainer}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-$DEFAULT_ROLLOUT_WORKERS}"
export MINI_SWE_WORKERS="${MINI_SWE_WORKERS:-1}"
export MAX_STEPS="${MAX_STEPS:-200}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
export TOP_P="${TOP_P:-1.0}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"
export STREAM_OUTPUT="${STREAM_OUTPUT:-0}"
export LLM_API_KEY="${LLM_API_KEY:-local}"
export VLLM_IMAGE="${VLLM_IMAGE:-$ROOT_DIR/cluster/apptainer/vllm-openai.sif}"
export USE_APPTAINER_VLLM="${USE_APPTAINER_VLLM:-1}"

vllm_pid=""
cleanup() {
  if [[ -n "$vllm_pid" ]]; then
    kill "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT HUP INT TERM

if [[ "$USE_APPTAINER_VLLM" == "1" ]]; then
  if [[ -z "${VLLM_PORT:-}" ]]; then
    read -r job_checksum _ <<< "$(printf '%s' "${PBS_JOBID:-$$}-$SHARD_INDEX-smith" | cksum)"
    VLLM_PORT=$((20000 + job_checksum % 20000))
  fi
  export HOST="${VLLM_HOST:-127.0.0.1}"
  export PORT="$VLLM_PORT"
  export LLM_BASE_URL="http://$HOST:$PORT/v1"
  cluster/apptainer/serve_vllm.sh >"$OUTPUT_DIR/vllm.log" 2>&1 &
  vllm_pid=$!
  printf '%s\n' "$vllm_pid" >"$OUTPUT_DIR/vllm.pid"

  deadline=$((SECONDS + ${VLLM_STARTUP_TIMEOUT:-7200}))
  until curl --fail --silent --max-time 2 "$LLM_BASE_URL/models" >/dev/null; do
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      wait "$vllm_pid"
      exit $?
    fi
    if ((SECONDS >= deadline)); then
      echo "vLLM did not become ready; see $OUTPUT_DIR/vllm.log" >&2
      exit 1
    fi
    sleep 5
  done
else
  export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
fi

cat <<MSG
Starting SWE-smith collection
  run:          $RUN_NAME
  dataset:      $DATASET ($SPLIT)
  revision:     $SWESMITH_DATASET_REVISION
  shard:        $SHARD_INDEX / $NUM_SHARDS
  tasks limit:  $LIMIT
  temperatures: $TEMPERATURES
  runs/temp:    $RUNS_PER_TEMPERATURE
  output:       $OUTPUT_DIR
MSG

scripts/collect_swesmith.sh
