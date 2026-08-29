#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXPERIMENT_ARM="${EXPERIMENT_ARM:-dmpo}"
export PREFERENCE_OBJECTIVE=dmpo
exec "$ROOT_DIR/scripts/train_preference.sh" "$@"
