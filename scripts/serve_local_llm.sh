#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
# Default to a small public MLX coder model for local smoke tests. Override
# MODEL=Kwai-Klear/Klear-AgentForge-8B-SFT when reproducing the pinned target.
# Avoid hf-xet downloader crashes seen on the gated Kwai-Klear model. Set
# HF_HUB_DISABLE_XET=0 before launching if you specifically want Xet enabled.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

MODEL="${MODEL:-mlx-community/Qwen2.5-Coder-7B-Instruct-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_TOKENS="${MAX_TOKENS:-65536}"
MLX_LM_VERSION="${MLX_LM_VERSION:-0.31.3}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
GENERATION_TIMEOUT="${GENERATION_TIMEOUT:-7200}"
PROMPT_CACHE_SIZE="${PROMPT_CACHE_SIZE:-1}"
PID_FILE="${PID_FILE:-/tmp/debug-depo-mlx-${PORT}.pid}"

SERVER_PID=""
SERVER_PGID=""
HEALTH_RESPONSE=""
CLEANED_UP=0

manager_is_running() {
  local manager_pid="$1"
  local manager_command

  if [[ ! "$manager_pid" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  if ! kill -0 "$manager_pid" 2>/dev/null; then
    return 1
  fi
  manager_command="$(ps -p "$manager_pid" -o command= 2>/dev/null || true)"
  [[ "$manager_command" == *"serve_local_llm.sh"* ]]
}

find_manager_for_port() {
  local current_pid
  local listener_pids
  local parent_pid
  local process_command
  local attempt

  if ! command -v lsof >/dev/null 2>&1; then
    return 1
  fi
  listener_pids="$(lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  current_pid="${listener_pids%%$'\n'*}"

  for attempt in {1..12}; do
    if [[ ! "$current_pid" =~ ^[0-9]+$ ]] || ((current_pid <= 1)); then
      return 1
    fi
    process_command="$(ps -p "$current_pid" -o command= 2>/dev/null || true)"
    if [[ "$process_command" == *"serve_local_llm.sh"* ]]; then
      printf '%s\n' "$current_pid"
      return 0
    fi
    parent_pid="$(ps -p "$current_pid" -o ppid= 2>/dev/null | tr -d ' ' || true)"
    current_pid="$parent_pid"
  done
  return 1
}

port_listener_pids() {
  if ! command -v lsof >/dev/null 2>&1; then
    return 1
  fi
  lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
}

describe_port_listeners() {
  local pids="$1"
  local pid

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if command -v ps >/dev/null 2>&1; then
      ps -p "$pid" -o pid=,ppid=,etime=,command= 2>/dev/null || echo "  pid $pid"
    else
      echo "  pid $pid"
    fi
  done <<<"$pids"
}

remove_own_pid_file() {
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE")" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
}

server_processes_running() {
  if [[ -n "$SERVER_PGID" ]]; then
    kill -0 -- "-$SERVER_PGID" 2>/dev/null
  elif [[ -n "$SERVER_PID" ]]; then
    kill -0 "$SERVER_PID" 2>/dev/null
  else
    return 1
  fi
}

cleanup() {
  local attempt

  if ((CLEANED_UP)); then
    return
  fi
  CLEANED_UP=1
  trap - EXIT HUP INT TERM

  if server_processes_running; then
    echo "Stopping local LLM (pid $SERVER_PID)..."
    if [[ -n "$SERVER_PGID" ]]; then
      kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
    else
      kill -TERM "$SERVER_PID" 2>/dev/null || true
    fi

    for attempt in {1..50}; do
      if ! server_processes_running; then
        break
      fi
      sleep 0.1
    done

    if server_processes_running; then
      if [[ -n "$SERVER_PGID" ]]; then
        kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
      else
        kill -KILL "$SERVER_PID" 2>/dev/null || true
      fi
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "$HEALTH_RESPONSE" ]]; then
    rm -f "$HEALTH_RESPONSE"
  fi
  remove_own_pid_file
}

handle_signal() {
  local exit_code="$1"
  cleanup
  exit "$exit_code"
}

stop_managed_server() {
  local manager_pid
  local attempt
  local listener_pids

  if [[ ! -f "$PID_FILE" ]]; then
    manager_pid="$(find_manager_for_port || true)"
    if [[ -z "$manager_pid" ]]; then
      listener_pids="$(port_listener_pids || true)"
      if [[ -n "$listener_pids" ]]; then
        echo "Port $PORT has a listener that was not started by this script; refusing to stop it." >&2
        describe_port_listeners "$listener_pids" >&2
        return 1
      fi
      if curl --fail --silent --max-time 2 "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
        echo "Port $PORT has a server that was not started by this script; refusing to stop it." >&2
        return 1
      fi
      echo "No server managed by this script is running on port $PORT."
      return 0
    fi
  else
    manager_pid="$(cat "$PID_FILE")"
  fi

  if ! manager_is_running "$manager_pid"; then
    rm -f "$PID_FILE"
    echo "Removed stale server state; no managed server is running on port $PORT."
    return 0
  fi

  echo "Stopping server manager $manager_pid..."
  kill -TERM "$manager_pid"
  for attempt in {1..100}; do
    if ! manager_is_running "$manager_pid"; then
      rm -f "$PID_FILE"
      echo "Local LLM stopped."
      return 0
    fi
    sleep 0.1
  done

  echo "The server manager did not stop within 10 seconds." >&2
  return 1
}

show_status() {
  local manager_pid
  local listener_pids

  if [[ -f "$PID_FILE" ]]; then
    manager_pid="$(cat "$PID_FILE")"
    if manager_is_running "$manager_pid"; then
      echo "Local LLM is managed by script pid $manager_pid at http://$HOST:$PORT/v1."
      return 0
    fi
    rm -f "$PID_FILE"
  fi

  manager_pid="$(find_manager_for_port || true)"
  if [[ -n "$manager_pid" ]]; then
    echo "Local LLM is managed by script pid $manager_pid at http://$HOST:$PORT/v1."
    return 0
  fi

  if curl --fail --silent --max-time 2 "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    echo "Port $PORT has an LLM server that was not started by this script." >&2
    return 1
  fi
  listener_pids="$(port_listener_pids || true)"
  if [[ -n "$listener_pids" ]]; then
    echo "Port $PORT has a listener, but it is not answering /v1/models yet." >&2
    describe_port_listeners "$listener_pids" >&2
    return 1
  fi
  echo "No local LLM is running on port $PORT."
}

generation_is_healthy() {
  HEALTH_RESPONSE="$(mktemp -t debug-depo-mlx-health.XXXXXX)"
  if ! curl --fail --silent --show-error --max-time "$GENERATION_TIMEOUT" \
    "http://$HOST:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"temperature\":0,\"max_tokens\":2,\"stream\":false}" \
    --output "$HEALTH_RESPONSE"; then
    rm -f "$HEALTH_RESPONSE"
    HEALTH_RESPONSE=""
    return 1
  fi

  if ! grep -q '"choices"' "$HEALTH_RESPONSE"; then
    cat "$HEALTH_RESPONSE" >&2
    rm -f "$HEALTH_RESPONSE"
    HEALTH_RESPONSE=""
    return 1
  fi

  rm -f "$HEALTH_RESPONSE"
  HEALTH_RESPONSE=""
  return 0
}

has_hf_token() {
  [[ -n "${HF_TOKEN:-}" ]] && return 0
  [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]] && return 0
  [[ -f "${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}" ]] && return 0
  [[ -f "$HOME/.huggingface/token" ]]
}

ACTION="start"
if (($# > 0)); then
  case "$1" in
    start | stop | status)
      ACTION="$1"
      shift
      ;;
  esac
fi

case "$ACTION" in
  stop)
    stop_managed_server
    exit $?
    ;;
  status)
    show_status
    exit $?
    ;;
esac

trap cleanup EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

cd "$ROOT_DIR"

echo "MLX-LM model: $MODEL"
if [[ "$MODEL" == Kwai-Klear/* ]] && ! has_hf_token; then
  cat >&2 <<'MSG'
This Hugging Face model requires accepted access conditions and an auth token.
If startup fails with 401 Unauthorized, log in to Hugging Face, accept the
model terms, then export HF_TOKEN or run `huggingface-cli login`.
MSG
fi

if [[ -f "$PID_FILE" ]]; then
  existing_manager_pid="$(cat "$PID_FILE")"
  if manager_is_running "$existing_manager_pid"; then
    echo "A server is already managed by script pid $existing_manager_pid." >&2
    echo "Run '$0 stop' before starting another one." >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

existing_models="$(
  curl --fail --silent --max-time 2 "http://$HOST:$PORT/v1/models" 2>/dev/null || true
)"
if [[ -n "$existing_models" ]]; then
  echo "Port $PORT is already occupied by an LLM server not managed by this script." >&2
  echo "Stop that process or choose another PORT; this launcher will not adopt it." >&2
  exit 1
fi
listener_pids="$(port_listener_pids || true)"
if [[ -n "$listener_pids" ]]; then
  echo "Port $PORT is already occupied by a process that is not answering /v1/models." >&2
  describe_port_listeners "$listener_pids" >&2
  echo "Stop that process or choose another PORT; this launcher will not start on top of it." >&2
  exit 1
fi

printf '%s\n' "$$" > "$PID_FILE"
# Job control gives the background server its own process group, allowing cleanup
# to terminate uv and every child it launches together.
set -m
"$UV_BIN" tool run --from "mlx-lm==$MLX_LM_VERSION" mlx_lm.server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-tokens "$MAX_TOKENS" \
  --decode-concurrency 1 \
  --prompt-concurrency 1 \
  --prompt-cache-size "$PROMPT_CACHE_SIZE" \
  "$@" &
SERVER_PID=$!
SERVER_PGID="$SERVER_PID"

deadline=$((SECONDS + STARTUP_TIMEOUT))
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    server_status=0
    wait "$SERVER_PID" || server_status=$?
    exit "$server_status"
  fi
  if curl --fail --silent --max-time 2 "http://$HOST:$PORT/v1/models" >/dev/null; then
    break
  fi
  if ((SECONDS >= deadline)); then
    echo "Local LLM server did not start within ${STARTUP_TIMEOUT}s." >&2
    exit 1
  fi
  sleep 1
done

echo "Server is listening; checking generation with timeout ${GENERATION_TIMEOUT}s..."
if ! generation_is_healthy; then
  echo "Local LLM accepted HTTP requests but failed the generation smoke test." >&2
  exit 1
fi

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "Local LLM process exited during its generation smoke test." >&2
  exit 1
fi

echo "Local LLM ready at http://$HOST:$PORT/v1 (pid $SERVER_PID)."
echo "Press Ctrl+C here, or run '$0 stop' from another terminal, to stop it."
server_status=0
wait "$SERVER_PID" || server_status=$?
exit "$server_status"
