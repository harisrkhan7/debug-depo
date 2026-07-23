#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DEST="${SWESMITH_REPO:-$ROOT_DIR/external/SWE-smith}"
REPO_URL="${SWESMITH_URL:-https://github.com/SWE-bench/SWE-smith.git}"
REVISION="${SWESMITH_REVISION:-9b74ac08118a85c39c356802f7961893af73e07f}"

cd "$ROOT_DIR"
if [[ ! -d ".venv" ]]; then
  "$UV_BIN" sync --extra dev --extra swebench
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
    echo "Cannot switch SWE-smith revisions with local changes in $DEST." >&2
    echo "Clean or recreate that ignored external checkout, then rerun setup." >&2
    exit 1
  fi
  git -C "$DEST" checkout --detach "$TARGET_REVISION"
fi

"$UV_BIN" pip install --python "$PROJECT_PYTHON" -e "$DEST[validate]"

PYTHONPATH="$DEST${PYTHONPATH:+:$PYTHONPATH}" \
  "$UV_BIN" run python - <<'PY'
from swesmith.profiles import registry

print(f"SWE-smith profile registry loaded: {len(list(registry.keys()))} keys")
PY

cat <<MSG
Installed official SWE-smith from:
  $DEST
Pinned revision:
  $TARGET_REVISION
Into Python:
  $PROJECT_PYTHON
MSG
