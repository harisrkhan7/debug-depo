#!/usr/bin/env bash
set -euo pipefail

# Copy this debug-depo checkout to Imperial CX3.
#
# Defaults can be overridden, for example:
#   REMOTE=another-cluster REMOTE_DIR='~/work/debug-depo' bash cluster/sync_to_cx3.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/cluster/env/load.sh"

if [[ -z "${REMOTE:-}" && -n "${REMOTE_USER:-}" && -n "${REMOTE_HOST:-}" ]]; then
  REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
fi
REMOTE="${REMOTE:-debug-depo-cluster}"
REMOTE_DIR="${REMOTE_DIR:-~/debug-depo}"
DRY_RUN="${DRY_RUN:-0}"
DELETE="${DELETE:-0}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"

rsync_args=(
  -az
  --human-readable
  --exclude ".DS_Store"
  --exclude ".git"
  --exclude ".ipynb_checkpoints"
  --exclude ".mypy_cache"
  --exclude ".pytest_cache"
  --exclude ".ruff_cache"
  --exclude ".uv-cache"
  --exclude ".venv"
  --exclude "__pycache__"
  --exclude "cluster/apptainer/*.sif"
  --exclude "cluster/logs/*"
  --exclude "cluster/env/local.sh"
  --exclude "data/processed/*"
  --exclude "external/*"
  --exclude "results/*"
  --exclude "scratch/*"
)

if "$RSYNC_BIN" --help 2>&1 | grep -q -- "--info"; then
  rsync_args+=(--info=progress2)
else
  rsync_args+=(--progress)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  rsync_args+=(--dry-run)
fi

if [[ "$DELETE" == "1" ]]; then
  rsync_args+=(--delete)
fi

remote="${REMOTE}:${REMOTE_DIR}/"

cat <<MSG
Syncing debug-depo to:
  $remote

Source:
  $ROOT_DIR/

Set DRY_RUN=1 to preview. Set DELETE=1 to remove remote files not present locally.
MSG

"$RSYNC_BIN" "${rsync_args[@]}" "$ROOT_DIR/" "$remote"
