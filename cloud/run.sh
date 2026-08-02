#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-help}"
if (($#)); then
  shift
fi

case "$ACTION" in
  setup)
    exec bash "$CLOUD_DIR/setup.sh" "$@"
    ;;
  prefetch-model)
    exec bash "$CLOUD_DIR/prefetch_model.sh" "$@"
    ;;
  preflight)
    exec bash "$CLOUD_DIR/preflight.sh" "$@"
    ;;
  storage)
    exec bash "$CLOUD_DIR/check_storage.sh" "$@"
    ;;
  smoke)
    exec bash "$CLOUD_DIR/smoke.sh" "$@"
    ;;
  push)
    exec bash "$CLOUD_DIR/push_to_cloud.sh" "$@"
    ;;
  pull)
    exec bash "$CLOUD_DIR/pull_from_cloud.sh" "$@"
    ;;
  build-cache)
    exec bash "$CLOUD_DIR/build_cache.sh" "$@"
    ;;
  collect)
    exec bash "$CLOUD_DIR/collect.sh" "$@"
    ;;
  evaluate)
    exec bash "$CLOUD_DIR/evaluate.sh" "$@"
    ;;
  analyze)
    exec bash "$CLOUD_DIR/analyze.sh" "$@"
    ;;
  pipeline)
    exec bash "$CLOUD_DIR/pipeline.sh" "$@"
    ;;
  trajectory-suite)
    exec bash "$CLOUD_DIR/trajectory_suite.sh" "$@"
    ;;
  preference-data)
    exec bash "$CLOUD_DIR/preference.sh" data "$@"
    ;;
  validate-data)
    exec bash "$CLOUD_DIR/preference.sh" validate-data "$@"
    ;;
  dmpo)
    exec bash "$CLOUD_DIR/preference.sh" dmpo "$@"
    ;;
  depo)
    exec bash "$CLOUD_DIR/preference.sh" depo "$@"
    ;;
  train)
    exec bash "$CLOUD_DIR/preference.sh" all "$@"
    ;;
  validate)
    exec bash "$CLOUD_DIR/validate.sh" "$@"
    ;;
  validate-model)
    exec bash "$CLOUD_DIR/validate_model.sh" "$@"
    ;;
  help|-h|--help)
    cat <<'MSG'
Usage: bash cloud/run.sh ACTION [ARGUMENT]

Actions:
  setup                         install dependencies and prefetch the model
  prefetch-model                serially populate the shared model cache
  preflight                     verify GPUs, runtimes, paths, and capacity
  storage                       verify durable and local filesystems
  smoke verified|swesmith       bounded end-to-end all-GPU smoke pipeline
  push                          rsync this checkout to the cloud VM
  pull                          copy cloud scratch into scratch/cloud
  build-cache smoke|full        prebuild local task-image cache (50 workers)
  collect verified|swesmith     collect with one shard per detected GPU
  evaluate verified|swesmith    merge and evaluate trajectories
  analyze verified|swesmith     write deterministic analysis
  pipeline verified|swesmith    collect, evaluate, and analyze
  trajectory-suite              collect 1K trajectories, then validate SFT at 100/200/500
  preference-data               build DMPO and DEPO data
  validate-data                 validate both preference datasets
  dmpo                          train/package DMPO on all detected GPUs
  depo                          train/package DEPO on all detected GPUs
  train                         build data, then train DMPO -> DEPO
  validate                      run one deterministic rollout per supplied SWE-smith task
  validate-model dmpo|depo      evaluate a packaged model on Verified
MSG
    ;;
  *)
    echo "Unknown Cloud action: $ACTION" >&2
    echo "Run: bash cloud/run.sh help" >&2
    exit 2
    ;;
esac
