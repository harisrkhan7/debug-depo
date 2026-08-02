#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$CLOUD_DIR/common.sh"

MODE="${1:-}"
PERSISTENT_SIF_DIR="${PERSISTENT_SIF_DIR:-$CLOUD_PERSISTENT_ROOT/sifs}"

case "$MODE" in
  persist)
    source_dir="$DEBUG_DEPO_SIF_ROOT"
    destination_dir="$PERSISTENT_SIF_DIR"
    ;;
  restore)
    source_dir="$PERSISTENT_SIF_DIR"
    destination_dir="$DEBUG_DEPO_SIF_ROOT"
    ;;
  *)
    echo "Usage: bash cloud/sync_sifs.sh persist|restore" >&2
    exit 2
    ;;
esac

require_command rsync "Install rsync before syncing SIFs."
if [[ ! -d "$source_dir" ]]; then
  echo "SIF source directory does not exist: $source_dir" >&2
  exit 2
fi
if [[ "$source_dir" == "$destination_dir" ]]; then
  echo "SIF source and destination must be different: $source_dir" >&2
  exit 2
fi

mkdir -p "$destination_dir"
echo "Syncing SIFs ($MODE):"
echo "  source:      $source_dir/"
echo "  destination: $destination_dir/"
rsync -a --progress "$source_dir/" "$destination_dir/"

sif_count="$(find "$destination_dir" -type f -name '*.sif' -print | wc -l | tr -d '[:space:]')"
echo "SIF sync complete: $sif_count files in $destination_dir"
