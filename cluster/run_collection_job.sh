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

COLLECTION_MODE="${COLLECTION_MODE:-full}"
case "$COLLECTION_MODE" in
  smoke)
    RUN_NAME="${RUN_NAME:-agentforge-verified-smoke}"
    DEFAULT_ROLLOUT_WORKERS=4
    NUM_SHARDS=1
    SHARD_INDEX=0
    LIMIT="${SMOKE_LIMIT:-5}"
    ;;
  full)
    RUN_NAME="${RUN_NAME:-agentforge-verified-full}"
    DEFAULT_ROLLOUT_WORKERS=8
    NUM_SHARDS="${NUM_SHARDS:-10}"
    SHARD_INDEX="${PBS_ARRAY_INDEX:-${SHARD_INDEX:-0}}"
    unset LIMIT
    ;;
  *)
    echo "COLLECTION_MODE must be 'smoke' or 'full', got: $COLLECTION_MODE" >&2
    exit 2
    ;;
esac

if ((NUM_SHARDS < 1 || SHARD_INDEX < 0 || SHARD_INDEX >= NUM_SHARDS)); then
  echo "Invalid shard selection: SHARD_INDEX=$SHARD_INDEX NUM_SHARDS=$NUM_SHARDS" >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/rollouts/shard-$SHARD_INDEX}"
mkdir -p "$OUTPUT_DIR"

export RUN_NAME RUN_ROOT OUTPUT_DIR NUM_SHARDS SHARD_INDEX
export DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
export SPLIT="${SPLIT:-test}"
export AGENTFORGE_MODEL="${AGENTFORGE_MODEL:-Kwai-Klear/Klear-AgentForge-8B-SFT}"
export HARNESS="${HARNESS:-mini-swe-agent-plus}"
export MINI_SWE_MODEL="${MINI_SWE_MODEL:-hosted_vllm/$AGENTFORGE_MODEL}"
export MINI_SWE_RUNNER="${MINI_SWE_RUNNER:-singularity}"
export MINI_SWE_ENVIRONMENT_CLASS="${MINI_SWE_ENVIRONMENT_CLASS:-singularity}"
export MSWEA_SINGULARITY_EXECUTABLE="${MSWEA_SINGULARITY_EXECUTABLE:-apptainer}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-$DEFAULT_ROLLOUT_WORKERS}"
export MINI_SWE_WORKERS="${MINI_SWE_WORKERS:-1}"
export MINI_SWE_DOCKER_START_CONCURRENCY="${MINI_SWE_DOCKER_START_CONCURRENCY:-1}"
export MAX_STEPS="${MAX_STEPS:-200}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"
export STREAM_OUTPUT="${STREAM_OUTPUT:-0}"
export LLM_API_KEY="${LLM_API_KEY:-local}"
export VLLM_IMAGE="${VLLM_IMAGE:-$ROOT_DIR/cluster/apptainer/vllm-openai.sif}"
export USE_APPTAINER_VLLM="${USE_APPTAINER_VLLM:-1}"

if [[ "$COLLECTION_MODE" == "smoke" ]]; then
  export LIMIT
fi

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
    read -r job_checksum _ <<< "$(printf '%s' "${PBS_JOBID:-$$}-$SHARD_INDEX" | cksum)"
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
Starting $COLLECTION_MODE trajectory collection
  run:             $RUN_NAME
  dataset:         $DATASET
  split:           $SPLIT
  shard:           $SHARD_INDEX / $NUM_SHARDS
  output:          $OUTPUT_DIR
  rollout workers: $ROLLOUT_WORKERS
  mini workers:    $MINI_SWE_WORKERS
  model server:    $LLM_BASE_URL
  limit:           ${LIMIT:-all tasks assigned to shard}
MSG

scripts/collect_rollouts.sh
