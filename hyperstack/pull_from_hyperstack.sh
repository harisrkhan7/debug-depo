#!/usr/bin/env bash
set -euo pipefail

# Pull the complete persistent HyperStack scratch tree for local inspection.
# Rebuildable caches, SIFs, and temporary files are intentionally outside it.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT_DIR/hyperstack/local.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/hyperstack/local.env"
fi

if [[ -z "${HYPERSTACK_REMOTE:-}" && -n "${HYPERSTACK_REMOTE_USER:-}" && -n "${HYPERSTACK_REMOTE_HOST:-}" ]]; then
  HYPERSTACK_REMOTE="${HYPERSTACK_REMOTE_USER}@${HYPERSTACK_REMOTE_HOST}"
fi
HYPERSTACK_REMOTE="${HYPERSTACK_REMOTE:-debug-depo-hyperstack}"
HYPERSTACK_REMOTE_PERSISTENT_ROOT="${HYPERSTACK_REMOTE_PERSISTENT_ROOT:-/root/debug-depo-persistent}"
HYPERSTACK_REMOTE_SCRATCH_DIR="${HYPERSTACK_REMOTE_SCRATCH_DIR:-$HYPERSTACK_REMOTE_PERSISTENT_ROOT/scratch}"
LOCAL_HYPERSTACK_SCRATCH_DIR="${LOCAL_HYPERSTACK_SCRATCH_DIR:-$ROOT_DIR/scratch/hyperstack}"
DRY_RUN="${DRY_RUN:-0}"
DELETE="${DELETE:-0}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"

mkdir -p "$LOCAL_HYPERSTACK_SCRATCH_DIR"

rsync_args=(
  -az
  --human-readable
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

cat <<MSG
Pulling the complete HyperStack scratch tree:
  source:      $HYPERSTACK_REMOTE:$HYPERSTACK_REMOTE_SCRATCH_DIR/
  destination: $LOCAL_HYPERSTACK_SCRATCH_DIR/
  dry run:     $DRY_RUN
  delete:      $DELETE

This includes runs, trajectories, merged predictions, evaluations, analyses,
training checkpoints/models, cache-build summaries, and logs. It does not copy
the ephemeral caches, SIFs, runtime state, or temporary files.
MSG

"$RSYNC_BIN" "${rsync_args[@]}" \
  "$HYPERSTACK_REMOTE:$HYPERSTACK_REMOTE_SCRATCH_DIR/" \
  "$LOCAL_HYPERSTACK_SCRATCH_DIR/"

if [[ "$DRY_RUN" != "1" ]]; then
  {
    printf 'remote=%s\n' "$HYPERSTACK_REMOTE"
    printf 'remote_scratch_dir=%s\n' "$HYPERSTACK_REMOTE_SCRATCH_DIR"
    printf 'local_scratch_dir=%s\n' "$LOCAL_HYPERSTACK_SCRATCH_DIR"
    printf 'delete=%s\n' "$DELETE"
    printf 'pulled_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } >"$LOCAL_HYPERSTACK_SCRATCH_DIR/_pull_manifest.txt"
fi

echo "HyperStack scratch bundle ready: $LOCAL_HYPERSTACK_SCRATCH_DIR"
