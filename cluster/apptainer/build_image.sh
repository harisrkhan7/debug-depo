#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-$ROOT_DIR/cluster/apptainer/debug_depo.sif}"
DEF_FILE="${DEF_FILE:-$ROOT_DIR/cluster/apptainer/debug_depo.def}"

apptainer build "$IMAGE" "$DEF_FILE"
