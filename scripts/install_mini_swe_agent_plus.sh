#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DEST="${MINI_SWE_AGENT_PLUS_REPO:-$ROOT_DIR/external/mini-swe-agent-plus}"
REPO_URL="${MINI_SWE_AGENT_PLUS_URL:-https://github.com/Kwai-Klear/mini-swe-agent-plus.git}"
REVISION="${MINI_SWE_AGENT_PLUS_REVISION:-3dfa5e26831306978ff3cfa2da15b49113ded0e6}"

cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  "$UV_BIN" sync --extra dev
fi

PROJECT_PYTHON="$("$UV_BIN" run python -c 'import sys; print(sys.executable)')"

if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone "$REPO_URL" "$DEST"
fi

if ! git -C "$DEST" cat-file -e "$REVISION^{commit}" 2>/dev/null; then
  git -C "$DEST" fetch origin "$REVISION"
  REVISION=FETCH_HEAD
fi
TARGET_REVISION="$(git -C "$DEST" rev-parse "$REVISION^{commit}")"
CURRENT_REVISION="$(git -C "$DEST" rev-parse HEAD)"
if [[ "$CURRENT_REVISION" != "$TARGET_REVISION" ]]; then
  if ! git -C "$DEST" diff --quiet || ! git -C "$DEST" diff --cached --quiet; then
    echo "Cannot switch mini-swe-agent-plus revisions with local changes in $DEST." >&2
    echo "Clean or recreate that ignored external checkout, then rerun setup." >&2
    exit 1
  fi
  git -C "$DEST" checkout --detach "$TARGET_REVISION"
fi

"$UV_BIN" run python scripts/patch_mini_swe_agent_plus.py patch "$DEST"

"$UV_BIN" pip install --python "$PROJECT_PYTHON" -e "$DEST"

"$UV_BIN" run python scripts/patch_mini_swe_agent_plus.py verify

cat <<MSG
Installed official mini-swe-agent-plus from:
  $DEST
Pinned revision:
  $TARGET_REVISION
Into Python:
  $PROJECT_PYTHON

Try a local no-model smoke first:
  MOCK=1 LIMIT=1 scripts/collect_rollouts.sh

For a real mini-swe-agent-plus rollout, set your OpenAI-compatible model server
and run for one task:
  HARNESS=mini-swe-agent-plus \\
  MINI_SWE_MODEL=hosted_vllm/Kwai-Klear/Klear-AgentForge-8B-SFT \\
  LLM_BASE_URL=http://127.0.0.1:8000/v1 \\
  LLM_API_KEY=local \\
  LIMIT=1 scripts/collect_rollouts.sh

On an Apptainer-only cluster, add:
  MINI_SWE_RUNNER=singularity \\
  MINI_SWE_ENVIRONMENT_CLASS=singularity \\
  MSWEA_SINGULARITY_EXECUTABLE=apptainer
MSG
