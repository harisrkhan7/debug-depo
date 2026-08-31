#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$CLOUD_DIR/common.sh"

require_command findmnt
require_separate_storage

if [[ "$(id -u)" == "0" ]]; then
  sudo_command=()
elif command -v sudo >/dev/null 2>&1; then
  sudo_command=(sudo)
else
  echo "System package installation requires root or sudo." >&2
  exit 2
fi

if [[ "${INSTALL_SYSTEM_PACKAGES:-1}" == "1" ]]; then
  "${sudo_command[@]}" apt-get update
  "${sudo_command[@]}" apt-get install -y \
    build-essential ca-certificates curl git jq python3 python3-pip \
    python3-venv rclone rsync software-properties-common tmux

  if ! command -v apptainer >/dev/null 2>&1; then
    "${sudo_command[@]}" add-apt-repository -y ppa:apptainer/ppa
    "${sudo_command[@]}" apt-get update
    "${sudo_command[@]}" apt-get install -y apptainer
  fi
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is missing. Use a Lambda Stack or GPU Base image." >&2
  exit 127
fi
if ! command -v apptainer >/dev/null 2>&1; then
  echo "Apptainer installation failed." >&2
  exit 127
fi

tool_venv="$CLOUD_PERSISTENT_ROOT/tools/uv-venv"
python3 -m venv "$tool_venv"
"$tool_venv/bin/python" -m pip install --upgrade pip uv

cd "$DEBUG_DEPO_ROOT"
"$UV" sync --extra dev --extra swebench --extra training
scripts/install_mini_swe_agent_plus.sh
scripts/install_swesmith.sh

temporary_vllm_image=""
cleanup_setup() {
  if [[ -n "$temporary_vllm_image" ]]; then
    rm -f -- "$temporary_vllm_image"
  fi
}
trap cleanup_setup EXIT

if ! apptainer inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
  temporary_vllm_image="$VLLM_IMAGE.incomplete.$$"
  rm -f -- "$temporary_vllm_image"
  if ! apptainer pull "$temporary_vllm_image" "$VLLM_APPTAINER_SOURCE"; then
    cat >&2 <<MSG
Could not create the vLLM SIF at $VLLM_IMAGE from:
  $VLLM_APPTAINER_SOURCE
If the OCI registry requires authentication, use 'apptainer registry login'
for that registry and rerun setup.
MSG
    exit 1
  fi
  mv -f -- "$temporary_vllm_image" "$VLLM_IMAGE"
  temporary_vllm_image=""
fi

# Populate the shared Hugging Face cache before any all-GPU workflow starts.
# This avoids every vLLM shard racing to download the same model.
bash "$CLOUD_DIR/prefetch_model.sh"

PYTHONPATH="$DEBUG_DEPO_ROOT/src:$DEBUG_DEPO_ROOT/external/mini-swe-agent-plus/src:$DEBUG_DEPO_ROOT/external/SWE-smith" \
  "$UV" run python - <<'PY'
import importlib.util

required = ("accelerate", "debug_depo", "minisweagent", "peft", "swebench", "swesmith", "torch")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing after setup: {', '.join(missing)}")
print("Lambda Cloud Python environment is ready.")
PY

cat <<MSG

Setup complete.
  Persistent runs:    $DEBUG_DEPO_SCRATCH
  Ephemeral cache:    $DEBUG_DEPO_CACHE_ROOT
  Ephemeral SIFs:     $(dirname "$VLLM_IMAGE")

Next:
  1. Verify storage: bash cloud/run.sh storage
  2. Run: bash cloud/run.sh preflight
  3. Preview: DRY_RUN=1 bash cloud/run.sh pipeline verified

Apptainer runs both vLLM and the task/evaluation environments. No Docker daemon
is required.
MSG
