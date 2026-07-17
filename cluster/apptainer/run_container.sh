#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-$ROOT_DIR/cluster/apptainer/debug_depo.sif}"

apptainer exec \
  --bind "$ROOT_DIR:$ROOT_DIR" \
  --pwd "$ROOT_DIR" \
  "$IMAGE" \
  "$@"
