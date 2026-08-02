#!/usr/bin/env bash
set -euo pipefail

# Copy this checkout to the cloud VM without copying local environments,
# caches, results, or persistent scratch artifacts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT_DIR/cloud/local.env" ]]; then
  # Load transfer settings only. Do not source env.sh on the local machine,
    # because its runtime defaults intentionally point at Lambda storage.
  # shellcheck disable=SC1091
  source "$ROOT_DIR/cloud/local.env"
fi

if [[ -z "${CLOUD_REMOTE:-}" && -n "${HYPERSTACK_REMOTE:-}" ]]; then
  CLOUD_REMOTE="$HYPERSTACK_REMOTE"
fi
if [[ -z "${CLOUD_REMOTE_USER:-}" && -n "${HYPERSTACK_REMOTE_USER:-}" ]]; then
  CLOUD_REMOTE_USER="$HYPERSTACK_REMOTE_USER"
fi
if [[ -z "${CLOUD_REMOTE_HOST:-}" && -n "${HYPERSTACK_REMOTE_HOST:-}" ]]; then
  CLOUD_REMOTE_HOST="$HYPERSTACK_REMOTE_HOST"
fi
if [[ -z "${CLOUD_REMOTE:-}" && -n "${CLOUD_REMOTE_USER:-}" && -n "${CLOUD_REMOTE_HOST:-}" ]]; then
  CLOUD_REMOTE="${CLOUD_REMOTE_USER}@${CLOUD_REMOTE_HOST}"
fi
CLOUD_REMOTE="${CLOUD_REMOTE:-debug-depo-cloud}"
CLOUD_REMOTE_REPO_DIR="${CLOUD_REMOTE_REPO_DIR:-${HYPERSTACK_REMOTE_REPO_DIR:-/home/ubuntu/debug-depo}}"
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
  --exclude "cloud/local.env"
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

remote="$CLOUD_REMOTE:$CLOUD_REMOTE_REPO_DIR/"
cat <<MSG
Pushing debug-depo to cloud VM:
  source:      $ROOT_DIR/
  destination: $remote
  dry run:     $DRY_RUN
  delete:      $DELETE

Local/remote virtual environments, caches, scratch, results, external
checkouts, and cloud/local.env are not transferred.
MSG

"$RSYNC_BIN" "${rsync_args[@]}" "$ROOT_DIR/" "$remote"
