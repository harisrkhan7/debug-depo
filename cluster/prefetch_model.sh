#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source cluster/env/load.sh

VLLM_MODEL="${VLLM_MODEL:-${AGENTFORGE_MODEL:-$BASELINE_SFT_MODEL}}"
VLLM_MODEL_REVISION="${VLLM_MODEL_REVISION:-}"
if [[ -z "$VLLM_MODEL_REVISION" && "$VLLM_MODEL" == "$BASELINE_SFT_MODEL" ]]; then
  VLLM_MODEL_REVISION="$BASELINE_SFT_MODEL_REVISION"
fi

if [[ -e "$VLLM_MODEL" ]]; then
  echo "Model is already a local path; no Hugging Face prefetch needed: $VLLM_MODEL"
  exit 0
fi

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/debug-depo/hf_token}"
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(<"$HF_TOKEN_FILE")"
  export HF_TOKEN
fi

PROJECT_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PROJECT_PYTHON" ]]; then
  echo "Project environment is missing at $ROOT_DIR/.venv." >&2
  echo "Run bash cluster/setup_rollout_env.sh first." >&2
  exit 2
fi

echo "Prefetching $VLLM_MODEL at ${VLLM_MODEL_REVISION:-the default branch} into $HF_HOME ..."
"$PROJECT_PYTHON" - "$VLLM_MODEL" "$VLLM_MODEL_REVISION" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

model, revision = sys.argv[1:]
snapshot_download(
    repo_id=model,
    revision=revision or None,
    cache_dir=os.environ["HF_HOME"] + "/hub",
    token=os.environ.get("HF_TOKEN"),
)
PY

echo "Model prefetch complete."
