#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$CLOUD_DIR/common.sh"

VLLM_MODEL="${VLLM_MODEL:-${AGENTFORGE_MODEL:-$BASELINE_SFT_MODEL}}"
VLLM_MODEL_REVISION="${VLLM_MODEL_REVISION:-}"
if [[ -z "$VLLM_MODEL_REVISION" && "$VLLM_MODEL" == "$BASELINE_SFT_MODEL" ]]; then
  VLLM_MODEL_REVISION="$BASELINE_SFT_MODEL_REVISION"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<MSG
Would prefetch the vLLM model serially:
  model:       $VLLM_MODEL
  revision:    ${VLLM_MODEL_REVISION:-<default branch>}
  cache:       $HF_HUB_CACHE
  hf_transfer: disabled
MSG
  exit 0
fi

require_command apptainer
require_separate_storage

if ! apptainer inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
  echo "vLLM Apptainer image is missing or invalid: $VLLM_IMAGE" >&2
  echo "Run bash cloud/run.sh setup first." >&2
  exit 2
fi

if [[ -z "${HF_TOKEN:-}" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
  export HF_TOKEN
fi
if [[ -z "${HF_TOKEN:-}" && -s "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(<"$HF_TOKEN_FILE")"
  export HF_TOKEN
fi
if [[ "$VLLM_MODEL" == Kwai-Klear/* && -z "${HF_TOKEN:-}" ]]; then
  cat >&2 <<MSG
The gated model requires a Hugging Face token: $VLLM_MODEL
Store it first with:
  bash cluster/save_hf_token.sh
Then rerun:
  bash cloud/run.sh prefetch-model
MSG
  exit 2
fi

bind_args=(
  --bind "$DEBUG_DEPO_ROOT:$DEBUG_DEPO_ROOT"
  --bind "$CLOUD_PERSISTENT_ROOT:$CLOUD_PERSISTENT_ROOT"
  --bind "$DEBUG_DEPO_EPHEMERAL:$DEBUG_DEPO_EPHEMERAL"
)

export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export APPTAINERENV_HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0
if [[ -n "${HF_TOKEN:-}" ]]; then
  export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
  export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "Prefetching $VLLM_MODEL at ${VLLM_MODEL_REVISION:-the default branch} into $HF_HUB_CACHE ..."
apptainer exec \
  "${bind_args[@]}" \
  --pwd "$DEBUG_DEPO_ROOT" \
  "$VLLM_IMAGE" \
  python3 - "$VLLM_MODEL" "$VLLM_MODEL_REVISION" <<'PY'
import sys

from huggingface_hub import snapshot_download

model = sys.argv[1]
revision = sys.argv[2] or None
path = snapshot_download(repo_id=model, revision=revision)
print(f"Model cache ready: {path}")
PY
