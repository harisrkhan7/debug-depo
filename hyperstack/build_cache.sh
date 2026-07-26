#!/usr/bin/env bash
set -euo pipefail

HYPERSTACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HYPERSTACK_DIR/common.sh"

MODE="${1:-full}"
case "$MODE" in
  smoke|full) ;;
  *)
    echo "Usage: bash hyperstack/build_cache.sh smoke|full" >&2
    exit 2
    ;;
esac

require_positive_integer CACHE_BUILD_MAX_WORKERS "$CACHE_BUILD_MAX_WORKERS"
if ((CACHE_BUILD_MAX_WORKERS < 50 || CACHE_BUILD_MAX_WORKERS > 100)); then
  echo "CACHE_BUILD_MAX_WORKERS must be between 50 and 100; got $CACHE_BUILD_MAX_WORKERS." >&2
  exit 2
fi

# Apptainer converts each OCI image to SIF with mksquashfs. Unless constrained,
# every concurrent mksquashfs process may use all processors visible to the VM;
# Linux schedules those threads but does not divide the 192 H200-host vCPUs
# evenly between cache workers. The default 75-worker configuration caps
# compression at two processors per pull, leaving CPU capacity for extraction,
# filesystem I/O, the controller, and the operating system. Override the cap
# only after measuring:
#
#   CACHE_BUILD_MAX_WORKERS=75 \
#   APPTAINER_MKSQUASHFS_ARGS='-processors 1' \
#     bash hyperstack/run.sh build-cache full
#
# Omitting the processor cap may still complete, because pulls spend time
# downloading and extracting as well as compressing, but simultaneous
# compression phases can create excessive runnable threads and reduce total
# throughput through CPU scheduling and shared-storage contention.

SWESMITH_TASK_IDS_FILE="${SWESMITH_TASK_IDS_FILE:-data/splits/swesmith_cache_5500_instance_ids.txt}"
summary_dir="${CACHE_BUILD_SUMMARY_ROOT:-$DEBUG_DEPO_SCRATCH/cache-builds}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
export CACHE_BUILD_MODE="$MODE"
export CACHE_BUILD_DATASETS="${CACHE_BUILD_DATASETS:-both}"
export CACHE_BUILD_MAX_WORKERS
export SWESMITH_TASK_IDS_FILE
export CACHE_BUILD_SUMMARY_OUTPUT="${CACHE_BUILD_SUMMARY_OUTPUT:-$summary_dir/$MODE-$timestamp.json}"

cat <<MSG
HyperStack ephemeral Apptainer cache build
  mode:              $CACHE_BUILD_MODE
  datasets:          $CACHE_BUILD_DATASETS
  workers:           $CACHE_BUILD_MAX_WORKERS
  squashfs args:     $APPTAINER_MKSQUASHFS_ARGS
  ephemeral root:    $HYPERSTACK_EPHEMERAL_ROOT
  SWE-bench SIFs:    $SWEBENCH_APPTAINER_SIF_DIR
  SWE-smith SIFs:    $SWESMITH_APPTAINER_SIF_DIR
  SWE-smith IDs:     $SWESMITH_TASK_IDS_FILE
  summary:           $CACHE_BUILD_SUMMARY_OUTPUT
MSG

if [[ "${DRY_RUN:-0}" != "1" && "$MODE" == "full" && "${SKIP_STORAGE_CHECK:-0}" != "1" ]]; then
  free_gib="$(available_gib "$DEBUG_DEPO_CACHE_ROOT")"
  minimum_gib="${MIN_FULL_CACHE_FREE_GIB:-1000}"
  if ((free_gib < minimum_gib)); then
    cat >&2 <<MSG
Only ${free_gib} GiB is free under $DEBUG_DEPO_CACHE_ROOT.
A full build requires at least the conservative ${minimum_gib} GiB guard.
Use a HyperStack flavor with sufficient ephemeral storage, or deliberately set
SKIP_STORAGE_CHECK=1 after confirming the temporary disk will fit the images.
MSG
    exit 2
  fi
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  require_command apptainer
  require_project_environment
  require_separate_storage
fi
mkdir -p "$summary_dir"
cd "$DEBUG_DEPO_ROOT"
scripts/build_apptainer_cache.sh
