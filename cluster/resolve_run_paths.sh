#!/usr/bin/env bash

# Source this from a cluster submission wrapper after RUN_NAME has been set.
# It resolves the same scratch root used inside PBS jobs without creating the
# model/container cache directories managed by cluster/env/load.sh.

if [[ -z "${RUN_NAME:-}" ]]; then
  echo "RUN_NAME must be set before sourcing cluster/resolve_run_paths.sh." >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_NAME may contain only letters, numbers, dots, underscores, and dashes." >&2
  return 2 2>/dev/null || exit 2
fi

DEBUG_DEPO_SUBMISSION_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/env" && pwd)"
if [[ -f "$DEBUG_DEPO_SUBMISSION_CONFIG_DIR/local.sh" ]]; then
  source "$DEBUG_DEPO_SUBMISSION_CONFIG_DIR/local.sh"
fi
source "$DEBUG_DEPO_SUBMISSION_CONFIG_DIR/defaults.sh"

export RUN_NAME
export RUN_ROOT="${RUN_ROOT:-$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME}"
export CLUSTER_LOG_DIR="${CLUSTER_LOG_DIR:-$RUN_ROOT/cluster-logs}"

unset DEBUG_DEPO_SUBMISSION_CONFIG_DIR
