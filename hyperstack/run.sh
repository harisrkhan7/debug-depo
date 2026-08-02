#!/usr/bin/env bash
set -euo pipefail

HYPERSTACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-help}"
if (($#)); then
  shift
fi

case "$ACTION" in
  setup)
    exec bash "$HYPERSTACK_DIR/setup.sh" "$@"
    ;;
  preflight)
    exec bash "$HYPERSTACK_DIR/preflight.sh" "$@"
    ;;
  smoke)
    exec bash "$HYPERSTACK_DIR/smoke.sh" "$@"
    ;;
  push)
    exec bash "$HYPERSTACK_DIR/push_to_hyperstack.sh" "$@"
    ;;
  pull)
    exec bash "$HYPERSTACK_DIR/pull_from_hyperstack.sh" "$@"
    ;;
  build-cache)
    exec bash "$HYPERSTACK_DIR/build_cache.sh" "$@"
    ;;
  collect)
    exec bash "$HYPERSTACK_DIR/collect.sh" "$@"
    ;;
  evaluate)
    exec bash "$HYPERSTACK_DIR/evaluate.sh" "$@"
    ;;
  analyze)
    exec bash "$HYPERSTACK_DIR/analyze.sh" "$@"
    ;;
  pipeline)
    exec bash "$HYPERSTACK_DIR/pipeline.sh" "$@"
    ;;
  preference-data)
    exec bash "$HYPERSTACK_DIR/preference.sh" data "$@"
    ;;
  validate-data)
    exec bash "$HYPERSTACK_DIR/preference.sh" validate-data "$@"
    ;;
  dmpo)
    exec bash "$HYPERSTACK_DIR/preference.sh" dmpo "$@"
    ;;
  depo)
    exec bash "$HYPERSTACK_DIR/preference.sh" depo "$@"
    ;;
  train)
    exec bash "$HYPERSTACK_DIR/preference.sh" all "$@"
    ;;
  validate)
    exec bash "$HYPERSTACK_DIR/pipeline.sh" validation "$@"
    ;;
  validate-model)
    exec bash "$HYPERSTACK_DIR/validate_model.sh" "$@"
    ;;
  help|-h|--help)
    cat <<'MSG'
Usage: bash hyperstack/run.sh ACTION [ARGUMENT]

Actions:
  setup                         install runtime and project dependencies
  preflight                     verify GPUs, runtimes, paths, and capacity
  smoke verified|swesmith       bounded end-to-end all-GPU smoke pipeline
  push                          rsync this checkout to HyperStack
  pull                          copy all HyperStack scratch into scratch/hyperstack
  build-cache smoke|full        prebuild ephemeral task-image cache (50 workers)
  collect verified|swesmith     collect with one shard per detected GPU
  evaluate verified|swesmith    merge and evaluate trajectories
  analyze verified|swesmith     write deterministic analysis
  pipeline verified|swesmith    collect, evaluate, and analyze
  preference-data               build DMPO and DEPO data
  validate-data                 validate both preference datasets
  dmpo                          train/package DMPO on all detected GPUs
  depo                          train/package DEPO on all detected GPUs
  train                         build data, then train DMPO -> DEPO
  validate                      run the 500-task SWE-smith holdout pipeline
  validate-model dmpo|depo      evaluate a packaged model on Verified
MSG
    ;;
  *)
    echo "Unknown HyperStack action: $ACTION" >&2
    echo "Run: bash hyperstack/run.sh help" >&2
    exit 2
    ;;
esac
