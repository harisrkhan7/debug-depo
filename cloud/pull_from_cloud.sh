#!/usr/bin/env bash
set -euo pipefail

# Pull the persistent Cloud scratch tree for local inspection. SIF backups,
# rebuildable caches, and temporary files are outside scratch. Large experiment
# model/checkpoint payloads are excluded by default.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT_DIR/cloud/local.env" ]]; then
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
CLOUD_REMOTE_PERSISTENT_ROOT="${CLOUD_REMOTE_PERSISTENT_ROOT:-${HYPERSTACK_REMOTE_PERSISTENT_ROOT:-/lambda/nfs/Debug-Depo/debug-depo-persistent}}"
CLOUD_REMOTE_SCRATCH_DIR="${CLOUD_REMOTE_SCRATCH_DIR:-$CLOUD_REMOTE_PERSISTENT_ROOT/scratch}"
LOCAL_CLOUD_SCRATCH_DIR="${LOCAL_CLOUD_SCRATCH_DIR:-${LOCAL_HYPERSTACK_SCRATCH_DIR:-$ROOT_DIR/scratch/cloud}}"
DRY_RUN="${DRY_RUN:-0}"
DELETE="${DELETE:-0}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
PULL_MODEL_ADAPTERS="${PULL_MODEL_ADAPTERS:-false}"
case "$PULL_MODEL_ADAPTERS" in
  1|true|TRUE) pull_model_adapters=1 ;;
  0|false|FALSE) pull_model_adapters=0 ;;
  *)
    echo "PULL_MODEL_ADAPTERS must be true or false, got: $PULL_MODEL_ADAPTERS" >&2
    exit 2
    ;;
esac

mkdir -p "$LOCAL_CLOUD_SCRATCH_DIR"

rsync_args=(
  -az
  --human-readable
)
if [[ "${PULL_EXPERIMENT_MODELS:-0}" != "1" ]]; then
  if [[ "$pull_model_adapters" == "1" ]]; then
    # Rsync uses the first matching filter rule, so these adapter payloads are
    # admitted before the broad experiment-model exclusions below.
    rsync_args+=(
      --include "**/experiments/**/adapter/***"
    )
  fi
  rsync_args+=(
    --exclude "**/experiments/**/*.safetensors"
    --exclude "**/experiments/**/*.bin"
    --exclude "**/experiments/**/*.pt"
    --exclude "**/experiments/**/*.pth"
    --exclude "**/experiments/**/*.ckpt"
    --exclude "**/experiments/**/*.gguf"
  )
fi
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
Pulling the cloud scratch tree:
  source:      $CLOUD_REMOTE:$CLOUD_REMOTE_SCRATCH_DIR/
  destination: $LOCAL_CLOUD_SCRATCH_DIR/
  dry run:     $DRY_RUN
  delete:      $DELETE
  experiment models: ${PULL_EXPERIMENT_MODELS:-0}
  model adapters:    $PULL_MODEL_ADAPTERS

This includes runs, trajectories, merged predictions, evaluations, analyses,
experiment metadata, cache-build summaries, and logs. Model/checkpoint payloads
under experiments/ are skipped unless PULL_EXPERIMENT_MODELS=1. Set
PULL_MODEL_ADAPTERS=true to copy only adapter directories while continuing to
skip merged models and checkpoints. The sibling
persistent sifs/ directory, ephemeral caches, runtime state, and temporary
files are not copied.
MSG

"$RSYNC_BIN" "${rsync_args[@]}" \
  "$CLOUD_REMOTE:$CLOUD_REMOTE_SCRATCH_DIR/" \
  "$LOCAL_CLOUD_SCRATCH_DIR/"

if [[ "$DRY_RUN" != "1" ]]; then
  {
    printf 'remote=%s\n' "$CLOUD_REMOTE"
    printf 'remote_scratch_dir=%s\n' "$CLOUD_REMOTE_SCRATCH_DIR"
    printf 'local_scratch_dir=%s\n' "$LOCAL_CLOUD_SCRATCH_DIR"
    printf 'delete=%s\n' "$DELETE"
    printf 'pull_experiment_models=%s\n' "${PULL_EXPERIMENT_MODELS:-0}"
    printf 'pull_model_adapters=%s\n' "$PULL_MODEL_ADAPTERS"
    printf 'pulled_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } >"$LOCAL_CLOUD_SCRATCH_DIR/_pull_manifest.txt"
fi

echo "Cloud scratch bundle ready: $LOCAL_CLOUD_SCRATCH_DIR"
