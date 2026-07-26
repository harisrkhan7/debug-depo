#!/usr/bin/env bash
set -euo pipefail

# Copy this checkout to the HyperStack VM without copying local environments,
# caches, results, or persistent scratch artifacts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT_DIR/hyperstack/local.env" ]]; then
  # Load transfer settings only. Do not source env.sh on the local machine,
  # because its runtime defaults intentionally point at /root.
  # shellcheck disable=SC1091
  source "$ROOT_DIR/hyperstack/local.env"
fi

if [[ -z "${HYPERSTACK_REMOTE:-}" && -n "${HYPERSTACK_REMOTE_USER:-}" && -n "${HYPERSTACK_REMOTE_HOST:-}" ]]; then
  HYPERSTACK_REMOTE="${HYPERSTACK_REMOTE_USER}@${HYPERSTACK_REMOTE_HOST}"
fi
HYPERSTACK_REMOTE="${HYPERSTACK_REMOTE:-debug-depo-hyperstack}"
HYPERSTACK_REMOTE_REPO_DIR="${HYPERSTACK_REMOTE_REPO_DIR:-/root/debug-depo}"
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
  --exclude "cluster/env/local.sh"
  --exclude "data/processed/*"
  --exclude "external/*"
  --exclude "hyperstack/local.env"
  --exclude "results/*"
  --exclude "scratch/"
)

if "$RSYNC_BIN" --help 2>&1 | grep -q -- "--protect-args"; then
  rsync_args+=(--protect-args)
fi
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

remote="$HYPERSTACK_REMOTE:$HYPERSTACK_REMOTE_REPO_DIR/"
cat <<MSG
Pushing debug-depo to HyperStack:
  source:      $ROOT_DIR/
  destination: $remote
  dry run:     $DRY_RUN
  delete:      $DELETE

Local/remote virtual environments, caches, scratch, results, external
checkouts, and hyperstack/local.env are not transferred.
MSG

"$RSYNC_BIN" "${rsync_args[@]}" "$ROOT_DIR/" "$remote"
